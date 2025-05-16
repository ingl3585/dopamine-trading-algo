# utils/io_utils.py

import os
import time
import logging
import pandas as pd

log = logging.getLogger(__name__)

def safe_read_csv(file_path, max_retries=3):
    for attempt in range(max_retries):
        try:
            if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
                return None

            df = pd.read_csv(file_path, header=None)
            if len(df) == 0:
                return None

            numeric_df = df.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')
            if numeric_df.isna().any().any():
                log.warning(f"CSV contains non-numeric data in attempt {attempt+1}")
                if attempt == max_retries - 1:
                    df = df.dropna()
                    if len(df) == 0:
                        return None
                    return df
                continue

            return df

        except Exception as e:
            log.warning(f"CSV read error (attempt {attempt+1}): {e}")
            time.sleep(1)

    return None

def clean_feature_file(file_path, max_lines=10000, max_size_mb=10):
    try:
        if os.path.getsize(file_path) > max_size_mb * 1024 * 1024:
            with open(file_path, 'r') as f:
                lines = f.readlines()

            keep_lines = max(max_lines, 1000)
            new_lines = [lines[0]] + lines[-keep_lines:]

            with open(file_path, 'w') as f:
                f.writelines(new_lines)
            log.info(f"Trimmed feature file to {len(new_lines)} lines")
    except Exception as e:
        log.error(f"File cleaning error: {e}")
