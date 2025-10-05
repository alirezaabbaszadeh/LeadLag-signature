
import numpy as np
import pandas as pd
import itertools
from functools import partial
from typing import Dict, List, Tuple, Union, Optional, Literal, Any
from dataclasses import dataclass, field
from sklearn.feature_selection import mutual_info_classif
import warnings
from tqdm import tqdm
from scipy.stats import mode


try:
    import iisignature
    SIGNATURE_AVAILABLE = True
except ImportError:
    SIGNATURE_AVAILABLE = False

try:
    import dcor
    DCOR_AVAILABLE = True
except ImportError:
    DCOR_AVAILABLE = False

try:
    from sklearn.feature_selection import mutual_info_classif
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

try:
    from p_tqdm import p_map
    P_TQDM_AVAILABLE = True
except ImportError:
    P_TQDM_AVAILABLE = False



@dataclass
class LeaderFollowerConfig:
    """Configuration class for Leader-Follower detection methods."""
    
    method: Literal['percentile'] = 'percentile'
    
    # Percentile method parameters
    top_percentile: float = 50.0
    bottom_percentile: float = 50.0
    agg_func : str = 'sum'
    
    def __post_init__(self):
        """Validate and set default cluster assignments."""

        # Validate percentile parameters
        if self.method == 'percentile':
            if not (0 <= self.top_percentile <= 100):
                raise ValueError("top_percentile must be between 0 and 100")
            if not (0 <= self.bottom_percentile <= 100):
                raise ValueError("bottom_percentile must be between 0 and 100")
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'LeaderFollowerConfig':
        """
        Create LeaderFollowerConfig from dictionary input.
        
        Args:
            config_dict: Dictionary with method and method-specific configurations
            
        Returns:
            LeaderFollowerConfig: Configured instance
        """
        method = config_dict.get('method', 'percentile')
        # Extract method-specific parameters
        if method == 'percentile':
            return cls(
                method=method,
                agg_func = config_dict.get('agg_func', 'sum'),
                top_percentile=config_dict.get('top_percentile', 50.0),
                bottom_percentile=config_dict.get('bottom_percentile', 50.0)
            )
        
        else:
            raise ValueError(f"Unknown method: {method}")

@dataclass
class LeadLagConfig:
    """Configuration class for Lead-Lag analysis parameters with DTW support."""
    
    method: Literal['ccf_at_lag', 'ccf_auc', 'signature', 'ccf_at_max_lag', 'dtw'] = 'ccf_at_lag'
    correlation_method: Literal['pearson', 'kendall', 'spearman', 'distance', 'mutual_information', 'squared_pearson'] = 'pearson'
    lookback: Optional[int] = 252
    update_freq: Optional[int] = 1
    use_parallel: bool = True
    num_cpus: int = 7
    quantiles: int = 4
    show_progress: bool = True
    Scaling_Method: str = 'mean-centering'
    sig_method: str = 'levy'
    
    # Method-specific parameters
    lag: Optional[int] = None
    max_lag: Optional[int] = None
    
    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'LeadLagConfig':
        """Create LeadLagConfig from dictionary input with DTW support."""
        method = config_dict.get('method', 'ccf_at_lag')
        
        # Get method-specific config
        method_config = config_dict.get(method, {})
        
        # Extract common parameters
        common_params = {
            'method': method,
            'lookback': config_dict.get('lookback', 252),
            'update_freq': config_dict.get('update_freq', 1),
            'use_parallel': config_dict.get('use_parallel', True),
            'num_cpus': config_dict.get('num_cpus', 7),
            'show_progress': config_dict.get('show_progress', True),
            'Scaling_Method': config_dict.get('Scaling_Method', 'mean-centering'),
        }
        
        # Method-specific parameter extraction
        if method == 'ccf_at_lag':
            common_params['lag'] = method_config.get('lag', 1)
            common_params['correlation_method'] = method_config.get('correlation_method', 'pearson')
            common_params['quantiles'] = method_config.get('quantiles', 4)
            
        elif method == 'ccf_auc':
            common_params['max_lag'] = method_config.get('max_lag', 10)
            common_params['correlation_method'] = method_config.get('correlation_method', 'pearson')
            common_params['quantiles'] = method_config.get('quantiles', 4)
            
        elif method == 'ccf_at_max_lag':
            common_params['max_lag'] = method_config.get('max_lag', 10)
            common_params['correlation_method'] = method_config.get('correlation_method', 'pearson')
            common_params['quantiles'] = method_config.get('quantiles', 4)
            
        elif method == 'signature':
            common_params['correlation_method'] = method_config.get('correlation_method', 'pearson')
            common_params['quantiles'] = method_config.get('quantiles', 4)
            common_params['sig_method'] = method_config.get('sig_method', 'custom')
            
        return cls(**common_params)
    
    def __post_init__(self):
        """Validate configuration parameters with DTW support."""

        if self.method in ['ccf_at_lag'] and self.lag is None:
            raise ValueError("lag parameter is required for ccf_at_lag method")
            
        if self.method in ['ccf_auc', 'ccf_at_max_lag'] and self.max_lag is None:
            raise ValueError("max_lag parameter is required for ccf_auc and ccf_at_max_lag methods")
            
        if self.method == 'signature' and not SIGNATURE_AVAILABLE:
            raise ValueError("iisignature package is required for signature method")
            
        if self.correlation_method == 'distance' and not DCOR_AVAILABLE:
            raise ValueError("dcor package is required for distance correlation method")
            
        if self.use_parallel and not PARALLEL_AVAILABLE:
            self.use_parallel = False
            warnings.warn("Parallel processing disabled due to missing p_tqdm package")

class LeadLagAnalyzer():
    """
    Professional Lead-Lag Analysis class with optimized numpy vectorization.
    
    This class provides comprehensive lead-lag relationship analysis between financial instruments
    using various correlation methods and temporal analysis techniques.
    """
    def __init__(self, config: Union[LeadLagConfig, Dict], df_universe: pd.Series = None):
        """
        Initialize the LeadLagAnalyzer with configuration and universe data.
        
        Args:
            config: LeadLagConfig object or dictionary containing all analysis parameters
            df_universe: Series with DatetimeIndex showing which coins to use on each date
        """
        if isinstance(config, dict):
            self.config = LeadLagConfig.from_dict(config)
        else:
            self.config = config
        self.lead_lag_matrix_rolling = None
        self.df_universe = df_universe
        self._validate_config()
        self.selected_window_info = None
    
    def _validate_config(self):
        """Validate the configuration parameters."""
        if not isinstance(self.config, LeadLagConfig):
            raise TypeError("config must be a LeadLagConfig instance")
    
    def _validate_data(self, data: pd.DataFrame, allow_partial_nan: bool = True):
        """
        Validate input data structure with option to allow partial NaN values.
        
        Args:
            data: Input DataFrame to validate
            allow_partial_nan: Whether to allow some NaN values (useful when universe filtering is applied)
        """
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Input data must be a pandas DataFrame")
        if not isinstance(data.index, pd.DatetimeIndex):
            raise TypeError("DataFrame index must be DatetimeIndex")
        if data.shape[1] < 2:
            raise ValueError("DataFrame must have at least 2 columns")
        
        if not allow_partial_nan:
            if data.isnull().all().any():
                raise ValueError("Some columns contain only NaN values")
        else:
            # Check if we have enough non-NaN data for analysis
            non_nan_counts = (~data.isnull()).sum(axis=1)
            if (non_nan_counts < 2).all():
                raise ValueError("Insufficient non-NaN data for analysis (need at least 2 assets per time period)")

    def _compute_log_returns(self,price_df: pd.DataFrame):
        return np.log(price_df).ffill().diff().fillna(0)
    
    def _get_universe_coins_for_date(self, date: pd.Timestamp) -> list:
        """
        Get universe coins for a specific date.
        
        Args:
            date: Target date
            
        Returns:
            list: List of coin names for the date
        """
        if self.df_universe is None:
            return None
        
        # Convert date to pandas Timestamp if needed
        if not isinstance(date, pd.Timestamp):
            date = pd.Timestamp(date)
        
        # Find the most recent universe date <= target date
        universe_index = pd.to_datetime(self.df_universe.index)
        mask = universe_index <= date
        available_dates = universe_index[mask]
        
        if len(available_dates) == 0:
            return None
        
        latest_date = available_dates.max()
        universe_coins = self.df_universe.loc[latest_date]
        
        if isinstance(universe_coins, str):
            return [universe_coins]
        elif hasattr(universe_coins, '__iter__') and not isinstance(universe_coins, str):
            return list(universe_coins)
        else:
            return None
    
    def _preprocess_window_data(self, price_df: pd.DataFrame, window_start: pd.Timestamp, window_end: pd.Timestamp) -> pd.DataFrame:
        """
        Preprocess price data for a specific window by selecting universe coins and applying forward filling.
        
        Args:
            price_df: Full DataFrame with DatetimeIndex and coin names as columns
            window_start: Start date of the window
            window_end: End date of the window (target date)
        
        Returns:
            pd.DataFrame: Preprocessed price data for the window with only universe coins
        """
        # Convert dates to pandas Timestamps if needed
        if not isinstance(window_start, pd.Timestamp):
            window_start = pd.Timestamp(window_start)
        if not isinstance(window_end, pd.Timestamp):
            window_end = pd.Timestamp(window_end)
        
        if self.df_universe is None:
            # If no universe data, return window data with standard preprocessing
            price_index = pd.to_datetime(price_df.index)
            window_mask = (price_index >= window_start) & (price_index <= window_end)
            window_data = price_df.loc[window_mask].copy()
            return window_data.ffill()
        
        # Get universe coins for the target date (window_end)
        universe_coins = self._get_universe_coins_for_date(window_end)
        
        if universe_coins is None or len(universe_coins) == 0:
            # If no universe defined for this date, return empty DataFrame
            price_index = pd.to_datetime(price_df.index)
            window_mask = (price_index >= window_start) & (price_index <= window_end)
            return pd.DataFrame(index=price_df.index[window_mask])
        
        # Filter coins that exist in the price data
        available_coins = [coin for coin in universe_coins if coin in price_df.columns]
        
        if len(available_coins) == 0:
            # No universe coins available in price data
            price_index = pd.to_datetime(price_df.index)
            window_mask = (price_index >= window_start) & (price_index <= window_end)
            return pd.DataFrame(index=price_df.index[window_mask])
        
        # Select window and universe coins
        price_index = pd.to_datetime(price_df.index)
        window_mask = (price_index >= window_start) & (price_index <= window_end)
        preprocessed_data = price_df.loc[window_mask, available_coins].copy()
        
        # Apply forward filling only within this window
        preprocessed_data = preprocessed_data.ffill()
        
        return preprocessed_data
    
    def _compute_log_returns_for_window(self, price_df: pd.DataFrame, window_start: pd.Timestamp, window_end: pd.Timestamp) -> pd.DataFrame:
        """
        Compute log returns for a specific window with proper universe filtering and preprocessing.
        
        Args:
            price_df: Full DataFrame with DatetimeIndex and coin names as columns
            window_start: Start date of the window
            window_end: End date of the window (target date)
        
        Returns:
            pd.DataFrame: Log returns for the window with only universe coins
        """
        # Preprocess price data for this specific window
        processed_prices = self._preprocess_window_data(price_df, window_start, window_end)
        
        if processed_prices.empty or processed_prices.shape[1] == 0:
            return pd.DataFrame()
        
        # Compute log returns
        log_returns = np.log(processed_prices).diff().fillna(0)
        
        return log_returns
    
    def _standardize(self,array: np.ndarray , scaling_method : str):
        if array.ndim != 2:
            raise ValueError("Input array must be 2D (n_samples, n_features)")

        mean = np.mean(array, axis=0)
        std = np.std(array, axis=0, ddof=1)
        if scaling_method == 'by_std':
            return (array - mean)/std
        return array/std

    def analyze(self, 
                price_df: pd.DataFrame, 
                return_rolling: bool = False) -> Union[pd.DataFrame, np.ndarray]:
        """
        Main analysis method that returns either rolling lead-lag matrices or single matrix.
        Modified to handle universe data properly by selecting appropriate coins for each window.
        
        Args:
            price_df: DataFrame with DatetimeIndex and coin names as columns
            return_rolling: If True, returns rolling lead-lag matrices over time
                        If False, returns single lead-lag matrix for entire dataset
        
        Returns:
            Union[pd.DataFrame, pd.Series]: Lead-lag matrix or rolling matrices depending on return_rolling
        """
        # Basic validation of input data
        if not isinstance(price_df, pd.DataFrame):
            raise TypeError("Input data must be a pandas DataFrame")
        if not isinstance(price_df.index, pd.DatetimeIndex):
            raise TypeError("DataFrame index must be DatetimeIndex")
        
        if return_rolling:
            return self._compute_rolling_lead_lag_matrix(price_df)
        else:
            # For single matrix, pass the full price DataFrame
            return self._compute_single_lead_lag_matrix(price_df)

    def _compute_rolling_lead_lag_matrix(self, price_df: pd.DataFrame) -> pd.Series:
        """
        Compute rolling lead-lag matrices over time using optimized numpy operations.
        Modified to handle universe-filtered data properly by selecting data for each window.
        
        Args:
            price_df: Full price DataFrame
            
        Returns:
            pd.Series: Series of lead-lag matrices indexed by date
        """
        if self.config.lookback >= len(price_df):
            raise ValueError("lookback period must be less than data length")
        
        # Store all possible column names for reference
        self.column_names = price_df.columns.tolist()
        
        date_index = price_df.index
        n_data = len(price_df)
        
        # Initialize result dictionary
        result_dict = {}
        result_windows_dict = {}

        
        # Setup progress bar if available and requested
        update_freq = self.config.update_freq
        if self.config.show_progress and TQDM_AVAILABLE:
            time_iterator = tqdm(
                range(self.config.lookback, n_data, update_freq),
                desc="Computing rolling lead-lag matrices",
                unit="window"
            )
        else:
            time_iterator = range(self.config.lookback, n_data, update_freq)

    

        for i in time_iterator:
            current_date = date_index[i]
            window_start = date_index[i - self.config.lookback + 1]
            
            # Get log returns for this specific window with proper universe filtering
            window_log_returns = self._compute_log_returns_for_window(
                price_df, window_start, current_date
            )
            
            if window_log_returns.empty or window_log_returns.shape[1] < 2:
                # Skip this window if insufficient valid assets
                continue
            
            # Get valid columns and data
            valid_columns = window_log_returns.columns.tolist()
            window_data = window_log_returns.values
            n_assets = len(valid_columns)
            
            # Initialize lead-lag matrix for this window
            lead_lag_matrix = np.zeros((n_assets, n_assets))
            
            # Generate all unique pairs
            pairs = list(itertools.combinations(range(n_assets), 2))
            
            # Compute lead-lag for all pairs in this window
            for pair in pairs:
                i1, j1 = pair
                # Extract pair data from window
                pair_data = window_data[:, [i1, j1]]
                
                # Skip if either series is all NaN
                if np.all(np.isnan(pair_data[:, 0])) or np.all(np.isnan(pair_data[:, 1])):
                    continue
                
                # Compute lead-lag measure
                val = self._compute_lead_lag_measure_optimized(pair_data)
                
                if not np.isnan(val):
                    # Fill symmetric entries (antisymmetric matrix)
                    lead_lag_matrix[i1, j1] = val
                    lead_lag_matrix[j1, i1] = -val
            
            # Convert to DataFrame with valid columns only
            matrix_df = pd.DataFrame(
                lead_lag_matrix,
                index=valid_columns,
                columns=valid_columns
            )
            
            result_dict[current_date] = matrix_df
    
        # Create result series with DatetimeIndex
        result_series = pd.Series(result_dict)
        result_series.index = pd.DatetimeIndex(result_series.index)
        
        return result_series
    
    def _compute_single_lead_lag_matrix(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Compute single lead-lag matrix for the entire dataset.
        
        Args:
            data: DataFrame with log returns
            
        Returns:
            pd.DataFrame: Lead-lag matrix
        """
        # Store column names for output
        self.column_names = data.columns.tolist()
        
        # Convert to numpy for computation
        data_values = data.values
        n_assets = data_values.shape[1]
        lead_lag_matrix = np.zeros((n_assets, n_assets))
        
        # Generate all pairs
        pairs = list(itertools.combinations(range(n_assets), 2))
        
        # Compute lead-lag for all pairs
        for pair in pairs:
            i, j = pair
            pair_data = data_values[:, [i, j]]
            value = self._compute_lead_lag_measure_optimized(pair_data)
            
            # Fill antisymmetric matrix
            lead_lag_matrix[i, j] = value
            lead_lag_matrix[j, i] = -value
        
        # Convert to DataFrame
        result = pd.DataFrame(
            lead_lag_matrix,
            index=self.column_names,
            columns=self.column_names
        )
        
        return result
    
    def _compute_lead_lag_measure_optimized(self, data_pair: np.ndarray) -> float:
        """
        Extended method to include DTW-based lead-lag measures.
        """
        if self.config.method == 'ccf_at_lag':
            return self._ccf_at_lag_optimized(data_pair)
        elif self.config.method == 'ccf_auc':
            return self._ccf_auc_optimized(data_pair)
        elif self.config.method == 'ccf_at_max_lag':
            return self._ccf_at_max_lag_optimized(data_pair)
        elif self.config.method == 'signature':
            return self._signature_method_optimized(data_pair)
        else:
            raise NotImplementedError(f"Method {self.config.method} not implemented")
    
    def _ccf_at_lag_optimized(self, data_pair: np.ndarray) -> float:
        """Cross-correlation at specific lag using numpy arrays."""
        x, y = data_pair[:, 0], data_pair[:, 1]
        
        corr_xy = self._cross_correlation_optimized(x, y, self.config.lag)
        corr_yx = self._cross_correlation_optimized(y, x, self.config.lag)
        
        return corr_xy - corr_yx
    
    def _ccf_auc_optimized(self, data_pair: np.ndarray) -> float:
        """Cross-correlation area under curve method using numpy arrays."""
        x, y = data_pair[:, 0], data_pair[:, 1]
        
        lags = np.arange(1, self.config.max_lag + 1)
        lags = np.r_[-lags, lags]
        
        # Vectorized correlation computation
        correlations = np.array([
            self._cross_correlation_optimized(x, y, lag) for lag in lags
        ])
        
        # Split positive and negative lags
        pos_mask = lags > 0
        neg_mask = lags < 0
        
        A = np.abs(correlations[pos_mask]).sum()
        B = np.abs(correlations[neg_mask]).sum()
        
        if A + B == 0:
            return 0.0
        
        return np.sign(A - B) * max(A, B) / (A + B)
    
    def _ccf_at_max_lag_optimized(self, data_pair: np.ndarray) -> float:
        """Cross-correlation at maximum lag method using numpy arrays."""
        x, y = data_pair[:, 0], data_pair[:, 1]
        
        lags = np.arange(1, self.config.max_lag + 1)
        lags = np.r_[-lags, lags]
        
        # Vectorized correlation computation
        correlations = np.array([
            self._cross_correlation_optimized(x, y, lag) for lag in lags
        ])
        
        # Split positive and negative lags
        pos_mask = lags > 0
        neg_mask = lags < 0
        
        leadingness = np.abs(correlations[pos_mask]).max()
        laggingness = np.abs(correlations[neg_mask]).max()
        
        if leadingness > laggingness:
            return leadingness
        elif leadingness < laggingness:
            return -laggingness
        else:
            return 0.0
    
    def _signature_method_optimized(self, data_pair: np.ndarray) -> float:
        """Signature method for lead-lag analysis using numpy arrays."""
        if not SIGNATURE_AVAILABLE:
            raise ImportError("iisignature package required for signature method")
        
        # Convert to cumulative sum and normalize
        path = np.cumsum(data_pair, axis=0)
        path = self._standardize(data_pair , self.config.Scaling_Method)
        # path = path / np.std(path, axis=0 , ddof=1)
        
        signature = iisignature.sig(path, 2, 1)
        if self.config.sig_method == 'levy':
            return (signature[1][1] - signature[1][2]) * 0.5
        return signature[1][1] - signature[1][2]
    
    def _cross_correlation_optimized(self, x: np.ndarray, y: np.ndarray, lag: int) -> float:
        """
        Compute cross-correlation between two numpy arrays at given lag.
        
        Args:
            x: First time series (numpy array)
            y: Second time series (numpy array)
            lag: Lag to apply to first series
            
        Returns:
            float: Cross-correlation value
        """
        # Handle NaN values
        if np.any(np.isnan(x)) or np.any(np.isnan(y)):
            valid_mask = ~(np.isnan(x) | np.isnan(y))
            if not np.any(valid_mask):
                return np.nan
            x = x[valid_mask]
            y = y[valid_mask]
        
        # Apply lag using numpy operations
        if lag > 0:
            if lag >= len(x):
                return np.nan
            x_lagged = x[:-lag]
            y_aligned = y[lag:]
        elif lag < 0:
            if -lag >= len(y):
                return np.nan
            x_lagged = x[-lag:]
            y_aligned = y[:lag]
        else:
            x_lagged = x
            y_aligned = y
        
        if len(x_lagged) == 0 or len(y_aligned) == 0:
            return np.nan
        
        # Compute correlation based on method
        if self.config.correlation_method == 'pearson':
            return np.corrcoef(x_lagged, y_aligned)[0, 1]
        
        elif self.config.correlation_method in ['kendall', 'spearman']:
            # Use pandas for these correlation types
            combined_df = pd.DataFrame({'x': x_lagged, 'y': y_aligned})
            return combined_df.corr(method=self.config.correlation_method).iloc[0, 1]
        
        elif self.config.correlation_method == 'distance':
            if not DCOR_AVAILABLE:
                raise ImportError("dcor package required for distance correlation")
            return dcor.distance_correlation(x_lagged, y_aligned)
        
        elif self.config.correlation_method == 'mutual_information':
            return self._mutual_information_correlation_optimized(x_lagged, y_aligned)
        
        elif self.config.correlation_method == 'squared_pearson':
            return np.corrcoef(x_lagged**2, y_aligned**2)[0, 1]
        
        else:
            raise NotImplementedError(f"Correlation method {self.config.correlation_method} not implemented")
    
    def _mutual_information_correlation_optimized(self, x: np.ndarray, y: np.ndarray) -> float:
        """Compute mutual information correlation using numpy arrays."""
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn package required for mutual information")
        
        try:
            # Discretize using quantiles
            x_quantiles = pd.qcut(x, q=self.config.quantiles, labels=False, duplicates='drop')
            y_quantiles = pd.qcut(y, q=self.config.quantiles, labels=False, duplicates='drop')
            
            # Create mask for valid values
            valid_mask = ~(np.isnan(x_quantiles) | np.isnan(y_quantiles))
            
            if not np.any(valid_mask):
                return np.nan
            
            x_valid = x_quantiles[valid_mask]
            y_valid = y_quantiles[valid_mask]
            
            if len(x_valid) == 0:
                return np.nan
            
            # Compute mutual information
            return mutual_info_classif(
                x_valid.reshape(-1, 1),
                y_valid,
                discrete_features=True,
                random_state=0
            )[0]
        
        except Exception:
            return np.nan

    def apply_detector(self,config) -> dict:
        # Handle config input
        if isinstance(config, dict):
            config = LeaderFollowerConfig.from_dict(config)
        elif isinstance(config, LeaderFollowerConfig):
            config = config
        else:
            raise TypeError("method_config must be a dictionary or LeaderFollowerConfig instance")
        if self.lead_lag_matrix_rolling is None:
            raise ValueError("lead_lag_matrix_rolling is None. Please call the 'leader_follower_detector' method before using this method.")
        leaders_followers_dict = {}
        # Iterate through each date and its corresponding lead-lag matrix
        for date, lead_lag_matrix in self.lead_lag_matrix_rolling.items():
            if config.method == 'percentile':
                leaders, followers = self._identify_leaders_followers_percentile(
                    lead_lag_matrix, config
                )
            else:
                raise ValueError(f"Unknown method: {config.method}")
            
            temp_df = pd.DataFrame({
                'leaders': pd.Series(leaders),
                'followers': pd.Series(followers)
            })
            
            # Store this DataFrame in the dictionary with the date as the key
            leaders_followers_dict[date] = temp_df.dropna()
        return pd.Series(leaders_followers_dict)

    def leader_follower_detector(self, 
                            lead_lag_matrix_rolling: pd.Series, 
                            method_config: Union[Dict[str, Any], LeaderFollowerConfig]) -> pd.Series:
        """
        Detect leaders and followers for each date in the rolling lead-lag analysis.
        
        Args:
            lead_lag_matrix_rolling: Series of lead-lag matrices from rolling analysis
            method_config: Dictionary or LeaderFollowerConfig object specifying the method and parameters
            
        Returns:
            pd.Series: Series of DataFrames with leaders and followers for each date
        """
        # Handle config input
        if isinstance(method_config, dict):
            config = LeaderFollowerConfig.from_dict(method_config)
        elif isinstance(method_config, LeaderFollowerConfig):
            config = method_config
        else:
            raise TypeError("method_config must be a dictionary or LeaderFollowerConfig instance")
        
        # Create a dictionary to store the DataFrames with dates as keys
        # leaders_followers_dict = {}
        self.clustering_rolling=None
        self.lead_lag_matrix_rolling=lead_lag_matrix_rolling

        return self.apply_detector(config)

    def _identify_leaders_followers_percentile(self, 
                                            lead_lag_matrix: pd.DataFrame, 
                                            config: LeaderFollowerConfig) -> Tuple[pd.Index, pd.Index]:
        """
        Identify leaders and followers using percentile-based method.
        
        Args:
            lead_lag_matrix: Lead-lag matrix DataFrame
            config: Configuration object with percentile parameters
            
        Returns:
            Tuple[pd.Index, pd.Index]: Leaders and followers indices
        """
        return self.identify_quantiles(
            lead_lag_matrix, 
            upper_perc=config.top_percentile, 
            lower_perc=config.bottom_percentile,
            config = config,
        )

    def identify_quantiles(self, lead_lag_matrix: pd.DataFrame, 
                          upper_perc: float, 
                          lower_perc: float,
                          config) -> Tuple[pd.Index, pd.Index]:
        """
        Identify leaders and followers based on row sums and percentile thresholds.
        
        Args:
            lead_lag_matrix: Lead-lag matrix DataFrame
            upper_perc: Upper percentile threshold for leaders
            lower_perc: Lower percentile threshold for followers
            
        Returns:
            Tuple[pd.Index, pd.Index]: Leaders and followers indices
        """
        if config.agg_func == 'sum':
            row_sums = lead_lag_matrix.sum(axis=1)
        elif config.agg_func == 'mean':
            # row_sums = np.nanmean(lead_lag_matrix, axis=1)
            row_sums = lead_lag_matrix.mean(axis=1)

        else:
            raise ValueError(f"Invalid agg_func: '{config.agg_func}'. Supported values are 'sum' and 'mean'.")
        row_sums = row_sums[row_sums != 0]  # Eliminate tokens with no value (row-sum = 0)

        leaders_threshold = np.percentile(row_sums, upper_perc)
        followers_threshold = np.percentile(row_sums, lower_perc)

        leaders = row_sums[row_sums > leaders_threshold].index
        followers = row_sums[row_sums < followers_threshold].index

        return leaders, followers
 