# utils/feature_logger.py

import os
import logging
import pandas as pd

log = logging.getLogger(__name__)

class FeatureLogger:
    def __init__(self, file_path, batch_size):
        self.file_path = file_path
        self.batch_size = batch_size
        self.rows = []

    def append(self, row):
        self.rows.append(row)

    def flush(self):
        if not self.rows:
            return
        try:
            df = pd.DataFrame(self.rows, columns=["ts", "close", "volume", "atr", "lwpe", "reward"])
            df.to_csv(self.file_path, mode='a', header=not os.path.exists(self.file_path), index=False)
            self.rows.clear()
            log.info("Flushed %d feature rows to disk", len(df))
        except Exception as e:
            log.warning(f"Flush failed: {e}")
