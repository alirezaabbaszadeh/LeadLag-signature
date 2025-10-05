import pandas as pd
import numpy as np
from pandas.api.types import is_scalar
from typing import Tuple, Union, Literal

def resample_crypto_data(
    df: pd.DataFrame, 
    timeframe: str, 
    price_type: Literal['open', 'high', 'low', 'close'] = 'close',
    method: Literal['last', 'first', 'mean', 'median', 'min', 'max'] = None
) -> pd.DataFrame:
    """
    Resample cryptocurrency price data to a different timeframe.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with datetime index and coin names as columns.
        Values should represent the specified price_type.
        
    timeframe : str
        Target timeframe for resampling. Examples:
        - '1H' or '1h': 1 hour
        - '4H' or '4h': 4 hours  
        - '1D' or '1d': 1 day
        - '1W' or '1w': 1 week
        - '1M' or '1m': 1 month
        - '15T' or '15min': 15 minutes
        - '30T' or '30min': 30 minutes
        
    price_type : str, default 'close'
        The type of price data in the input dataframe.
        Options: 'open', 'high', 'low', 'close'
        
    method : str, optional
        Resampling method. If None, defaults based on price_type:
        - 'open': uses 'first' (first value in period)
        - 'high': uses 'max' (maximum value in period)  
        - 'low': uses 'min' (minimum value in period)
        - 'close': uses 'last' (last value in period)
        
        Manual options: 'first', 'last', 'mean', 'median', 'min', 'max'
    
    Returns:
    --------
    pd.DataFrame
        Resampled dataframe with the same structure as input but different timeframe.
        
    Examples:
    ---------
    # Resample hourly close prices to daily close prices
    daily_close = resample_crypto_data(hourly_df, '1D', 'close')
    
    # Resample hourly data to 4-hour open prices
    four_hour_open = resample_crypto_data(hourly_df, '4H', 'open')
    
    # Use custom method - get average price over each day
    daily_avg = resample_crypto_data(hourly_df, '1D', 'close', method='mean')
    """
    
    # Input validation
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("DataFrame index must be a DatetimeIndex")
    
    if df.empty:
        raise ValueError("Input DataFrame is empty")
    
    if price_type not in ['open', 'high', 'low', 'close']:
        raise ValueError("price_type must be one of: 'open', 'high', 'low', 'close'")
    
    # Create a copy to avoid modifying the original dataframe
    df_copy = df.copy()
    
    # Set default method based on price_type if not specified
    if method is None:
        method_mapping = {
            'open': 'first',
            'high': 'max', 
            'low': 'min',
            'close': 'last'
        }
        method = method_mapping[price_type]
    
    # Validate method
    valid_methods = ['first', 'last', 'mean', 'median', 'min', 'max']
    if method not in valid_methods:
        raise ValueError(f"method must be one of: {valid_methods}")
    
    # Perform resampling
    try:
        resampler = df_copy.resample(timeframe)
        
        # Apply the appropriate aggregation method
        if method == 'first':
            result_df = resampler.first()
        elif method == 'last':
            result_df = resampler.last()
        elif method == 'mean':
            result_df = resampler.mean()
        elif method == 'median':
            result_df = resampler.median()
        elif method == 'min':
            result_df = resampler.min()
        elif method == 'max':
            result_df = resampler.max()
            
    except Exception as e:
        raise ValueError(f"Error during resampling with timeframe '{timeframe}': {str(e)}")
    
    # Remove any rows with all NaN values (can happen with sparse data)
    result_df = result_df.dropna(how='all')
    
    # Ensure the result maintains the same column order as input
    result_df = result_df[df.columns]
    
    return result_df

def selected_uni(close_price, df_universe, maximum_coin=50, window_size=210):

    # Ensure datetime types
    close_price.index = pd.to_datetime(close_price.index)
    df_universe['date'] = pd.to_datetime(df_universe['date'], format='%Y-%m')
    
    # Generate prev_month column for close_price
    df_close_month = pd.DataFrame({
        'prev_month': (close_price.index - pd.DateOffset(months=1)).to_period('M')
    }, index=close_price.index)
    
    # Prepare df_universe
    df_universe['month'] = df_universe['date'].dt.to_period('M')
    df_universe['symbols_list'] = df_universe['symbols'].str.split(', ')
    
    # Merge
    merged = df_close_month.merge(
        df_universe[['month', 'symbols_list']],
        left_on='prev_month', right_on='month',
        how='left'
    ).drop(columns=['month', 'prev_month'])
    
    def trim_coins(symbols):
        """Trim coin list based on maximum_coin and ensure even count"""
        if not isinstance(symbols, list):
            return []
        
        seen = set()
        unique_symbols = []
        for symbol in symbols:
            if symbol not in seen:
                seen.add(symbol)
                unique_symbols.append(symbol)
        
        count = len(unique_symbols)
        if count < 2:
            return unique_symbols
        if count > maximum_coin:
            unique_symbols = unique_symbols[:maximum_coin]
            count = maximum_coin
        if count % 2 == 1:
            count -= 1
        return unique_symbols[:count]
    
    # Apply trimming
    merged['symbols_list'] = merged['symbols_list'].apply(trim_coins)
    merged['symbols_list'] = merged['symbols_list'].apply(lambda x: np.nan if x == [] else x)
    
    # Create initial series
    result_series = pd.Series(merged['symbols_list'].values, index=close_price.index, name='coins_list')
    
    def filter_nan_coins(date, coins):
        """Filter out coins with too many NaN values in the window"""
        if not isinstance(coins, list):
            return np.nan
        
        try:
            date_idx = close_price.index.get_loc(date)
        except KeyError:
            return np.nan
            
        start_idx = max(0, date_idx - window_size)
        window_df = close_price.iloc[start_idx:date_idx]
        
        if window_df.empty:
            return coins
        
        bad_coins = set()
        for col in coins:
            if col not in close_price.columns:
                bad_coins.add(col)
                continue
                
            series = window_df[col]
            # Only condition: more than 10 NaNs in the window
            if series.isna().sum() > 10:
                bad_coins.add(col)
        
        filtered = [c for c in coins if c not in bad_coins]
        
        seen = set()
        unique_filtered = []
        for coin in filtered:
            if coin not in seen:
                seen.add(coin)
                unique_filtered.append(coin)
        
        if len(unique_filtered) % 2 == 1 and len(unique_filtered) > 1:
            unique_filtered = unique_filtered[:-1]
        
        return unique_filtered if unique_filtered else np.nan
    
    # Apply filtering
    result_series = pd.Series(
        [filter_nan_coins(date, coins) for date, coins in zip(result_series.index, result_series)],
        index=result_series.index,
        name='coins_list'
    ).dropna()
    result_series.index = pd.to_datetime(result_series.index)
    return result_series


def preprocess_ffill(df_close: pd.DataFrame,
                     df_universe: Union[pd.Series, pd.DataFrame],
                     end_date: str = None) -> Tuple[pd.DataFrame, Union[pd.Series, pd.DataFrame]]:
    """
    Forward-fill NaN in df_close only for coins present in the universe each day.
    Also drop rows before first date with coins in universe.
    If end_date (str 'YYYY-MM-DD') is given, keep only dates < end_date in both dataframes.

    Returns:
        (df_filled, df_universe_filtered)
    """
    if isinstance(df_universe, pd.DataFrame):
        df_universe = df_universe.iloc[:, 0]

    df_close = df_close.sort_index()
    df_universe = df_universe.sort_index()

    if end_date is not None:
        end_date_ts = pd.to_datetime(end_date)
        df_close = df_close.loc[df_close.index < end_date_ts]
        df_universe = df_universe.loc[df_universe.index < end_date_ts]

    def parse_cell(v):
        if is_scalar(v):
            if pd.isna(v):
                return []
            if isinstance(v, str):
                s = v.strip()
                if not s:
                    return []
                return [c.strip() for c in s.split(',') if c.strip()]
            return []
        try:
            seq = list(v)
        except Exception:
            return []
        out = []
        for item in seq:
            if pd.isna(item):
                continue
            if isinstance(item, str):
                item = item.strip()
                if not item:
                    continue
                if ',' in item:
                    parts = [p.strip() for p in item.split(',') if p.strip()]
                    out.extend(parts)
                else:
                    out.append(item)
            else:
                out.append(str(item))
        return out

    # Find first date with coins
    first_univ_date = None
    for dt in df_universe.index:
        coins = parse_cell(df_universe.loc[dt])
        if coins:
            first_univ_date = dt
            break

    if first_univ_date is None:
        # Empty output, but return filtered universe anyway
        return df_close.iloc[0:0].copy(), df_universe.copy()

    # Filter df_close to dates >= first_univ_date
    df_close = df_close.loc[df_close.index >= first_univ_date]

    if df_close.shape[0] == 0:
        return df_close, df_universe

    universe_by_date = {}
    for dt in df_close.index:
        if dt in df_universe.index:
            universe_by_date[dt] = parse_cell(df_universe.loc[dt])
        else:
            universe_by_date[dt] = []

    df_filled = df_close.copy()
    last_valid = {}

    for dt in df_close.index:
        coins_today = universe_by_date.get(dt, [])
        row = df_close.loc[dt]

        for col in df_close.columns:
            val = row[col]
            if not pd.isna(val):
                last_valid[col] = val

        for c in coins_today:
            if c not in df_close.columns:
                continue
            if pd.isna(df_filled.at[dt, c]) and c in last_valid:
                df_filled.at[dt, c] = last_valid[c]

    return df_filled, df_universe
