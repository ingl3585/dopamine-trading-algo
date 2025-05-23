# utils/io_utils.py

import os
import time
import logging
import pandas as pd

log = logging.getLogger(__name__)

def safe_read_csv(file_path, max_retries=3):
    """Safely read CSV with Ichimoku/EMA feature validation"""
    for attempt in range(max_retries):
        try:
            if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
                log.debug(f"CSV file not found or empty: {file_path}")
                return None

            # Read CSV
            df = pd.read_csv(file_path)
            if len(df) == 0:
                log.debug("CSV file contains no data")
                return None

            # Validate column structure for Ichimoku/EMA features
            expected_columns = [
                "ts", "close", "normalized_volume", 
                "tenkan_kijun_signal", "price_cloud_signal", "future_cloud_signal",
                "ema_cross_signal", "tenkan_momentum", "kijun_momentum", 
                "lwpe", "reward"
            ]
            
            # Check if we have the right number of columns
            if df.shape[1] == len(expected_columns):
                df.columns = expected_columns
                log.debug(f"Applied Ichimoku/EMA column names to CSV")
            elif df.shape[1] == 11:  # Expected for new format
                df.columns = expected_columns
            else:
                log.warning(f"Unexpected column count: {df.shape[1]}, expected {len(expected_columns)}")
                # Try to handle legacy format
                if df.shape[1] == 6:  # Old ATR format
                    log.info("Converting legacy ATR format to Ichimoku/EMA format")
                    return _convert_legacy_format(df)

            # Validate feature data
            feature_columns = expected_columns[1:-1]  # Exclude timestamp and reward
            numeric_df = df[feature_columns].apply(pd.to_numeric, errors='coerce')
            
            if numeric_df.isna().any().any():
                log.warning(f"CSV contains non-numeric data in attempt {attempt+1}")
                if attempt == max_retries - 1:
                    # Clean the data on final attempt
                    df = df.dropna()
                    if len(df) == 0:
                        return None
                    log.info(f"Cleaned CSV data, {len(df)} rows remaining")
                    return df
                continue

            # Validate signal ranges
            signal_columns = [
                "tenkan_kijun_signal", "price_cloud_signal", "future_cloud_signal",
                "ema_cross_signal", "tenkan_momentum", "kijun_momentum"
            ]
            
            for col in signal_columns:
                if col in df.columns:
                    # Check for values outside expected range
                    out_of_range = ~df[col].isin([-1, 0, 1])
                    if out_of_range.any():
                        log.warning(f"Signal column {col} has values outside [-1,0,1] range")
                        # Clip to valid range
                        df[col] = df[col].clip(-1, 1)

            # Validate LWPE range
            if 'lwpe' in df.columns:
                lwpe_out_of_range = (df['lwpe'] < 0) | (df['lwpe'] > 1)
                if lwpe_out_of_range.any():
                    log.warning(f"LWPE values out of [0,1] range: {lwpe_out_of_range.sum()} rows")
                    df['lwpe'] = df['lwpe'].clip(0, 1)

            log.info(f"Successfully loaded CSV with {len(df)} rows and Ichimoku/EMA features")
            return df

        except Exception as e:
            log.warning(f"CSV read error (attempt {attempt+1}): {e}")
            time.sleep(1)

    log.error(f"Failed to read CSV after {max_retries} attempts")
    return None

def _convert_legacy_format(legacy_df):
    """Convert legacy ATR format to new Ichimoku/EMA format"""
    try:
        log.info("Converting legacy format - adding placeholder Ichimoku/EMA features")
        
        # Legacy columns: [ts, close, volume, atr, lwpe, reward]
        legacy_columns = ["ts", "close", "volume", "atr", "lwpe", "reward"]
        if len(legacy_df.columns) == 6:
            legacy_df.columns = legacy_columns
        
        # Create new DataFrame with Ichimoku/EMA structure
        new_df = pd.DataFrame()
        
        # Keep existing data
        new_df["ts"] = legacy_df["ts"]
        new_df["close"] = legacy_df["close"]
        new_df["normalized_volume"] = legacy_df["volume"]  # Assume already normalized
        
        # Add placeholder Ichimoku/EMA signals (neutral)
        new_df["tenkan_kijun_signal"] = 0
        new_df["price_cloud_signal"] = 0
        new_df["future_cloud_signal"] = 0
        new_df["ema_cross_signal"] = 0
        new_df["tenkan_momentum"] = 0
        new_df["kijun_momentum"] = 0
        
        # Keep LWPE and reward
        new_df["lwpe"] = legacy_df["lwpe"]
        new_df["reward"] = legacy_df["reward"]
        
        log.info(f"Converted {len(new_df)} rows from legacy to Ichimoku/EMA format")
        return new_df
        
    except Exception as e:
        log.error(f"Legacy format conversion failed: {e}")
        return None

def clean_feature_file(file_path, max_lines=10000, max_size_mb=10):
    """Clean feature file with Ichimoku/EMA awareness"""
    try:
        if not os.path.exists(file_path):
            return
            
        file_size = os.path.getsize(file_path)
        if file_size > max_size_mb * 1024 * 1024:
            log.info(f"Feature file size ({file_size/1024/1024:.1f}MB) exceeds limit, cleaning...")
            
            with open(file_path, 'r') as f:
                lines = f.readlines()

            if len(lines) > 1:
                # Keep header and recent data
                keep_lines = max(max_lines, 1000)
                header = lines[0]
                recent_lines = lines[-keep_lines:]
                new_lines = [header] + recent_lines

                with open(file_path, 'w') as f:
                    f.writelines(new_lines)
                    
                log.info(f"Trimmed feature file to {len(new_lines)} lines "
                        f"(saved {len(lines) - len(new_lines)} lines)")
            
    except Exception as e:
        log.warning(f"File cleaning error: {e}")

def validate_feature_file_format(file_path):
    """Validate that feature file has correct Ichimoku/EMA format"""
    try:
        if not os.path.exists(file_path):
            return True  # New file is OK
        
        # Read just the header
        with open(file_path, 'r') as f:
            header = f.readline().strip()
        
        expected_columns = [
            "ts", "close", "normalized_volume", 
            "tenkan_kijun_signal", "price_cloud_signal", "future_cloud_signal",
            "ema_cross_signal", "tenkan_momentum", "kijun_momentum", 
            "lwpe", "reward"
        ]
        
        header_columns = header.split(',')
        
        if len(header_columns) != len(expected_columns):
            log.warning(f"Feature file has {len(header_columns)} columns, "
                       f"expected {len(expected_columns)} for Ichimoku/EMA format")
            return False
        
        # Check column names match
        for i, (expected, actual) in enumerate(zip(expected_columns, header_columns)):
            if expected != actual.strip():
                log.warning(f"Column {i} mismatch: expected '{expected}', got '{actual.strip()}'")
                return False
        
        log.info("Feature file format validation passed")
        return True
        
    except Exception as e:
        log.warning(f"Feature file validation error: {e}")
        return False

def backup_feature_file(file_path):
    """Create backup of feature file before major changes"""
    try:
        if not os.path.exists(file_path):
            return None
        
        timestamp = int(time.time())
        backup_path = f"{file_path}.backup_{timestamp}"
        
        # Copy file
        import shutil
        shutil.copy2(file_path, backup_path)
        
        log.info(f"Created feature file backup: {backup_path}")
        return backup_path
        
    except Exception as e:
        log.warning(f"Feature file backup failed: {e}")
        return None

def migrate_feature_file_format(file_path):
    """Migrate feature file from legacy ATR format to Ichimoku/EMA format"""
    try:
        if not os.path.exists(file_path):
            log.info("No existing feature file to migrate")
            return True
        
        # Create backup first
        backup_path = backup_feature_file(file_path)
        if not backup_path:
            log.error("Failed to create backup, aborting migration")
            return False
        
        # Read existing data
        df = safe_read_csv(file_path)
        if df is None:
            log.warning("Could not read existing feature file for migration")
            return False
        
        # Check if already in new format
        if df.shape[1] == 11:  # New format
            log.info("Feature file already in Ichimoku/EMA format")
            return True
        
        # Convert to new format
        if df.shape[1] == 6:  # Legacy ATR format
            converted_df = _convert_legacy_format(df)
            if converted_df is not None:
                # Save converted data
                converted_df.to_csv(file_path, index=False)
                log.info(f"Successfully migrated feature file to Ichimoku/EMA format")
                return True
        
        log.warning(f"Unknown feature file format with {df.shape[1]} columns")
        return False
        
    except Exception as e:
        log.error(f"Feature file migration failed: {e}")
        return False