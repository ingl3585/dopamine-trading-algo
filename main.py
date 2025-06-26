# main.py

import argparse
import logging
import shutil
import signal
import sys
from pathlib import Path

from trading_system import TradingSystem

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/trading.log'),
            logging.StreamHandler()
        ]
    )

def reset_workspace():
    import os
    import stat
    
    def handle_remove_readonly(func, path, exc):
        if os.path.exists(path):
            os.chmod(path, stat.S_IWRITE)
            func(path)
    
    dirs = ["models", "data", "logs"]
    for d in dirs:
        path = Path(d)
        if path.exists():
            print(f"Removing {d}/...")
            try:
                shutil.rmtree(path, onerror=handle_remove_readonly)
            except PermissionError as e:
                print(f"Warning: Could not remove {d}/ - {e}")
                # Try to clear contents instead
                for item in path.iterdir():
                    try:
                        if item.is_file():
                            item.chmod(0o777)
                            item.unlink()
                        elif item.is_dir():
                            shutil.rmtree(item, onerror=handle_remove_readonly)
                    except Exception as ex:
                        print(f"Warning: Could not remove {item} - {ex}")
        
        path.mkdir(parents=True, exist_ok=True)
        print(f"Created {d}/")
    
    print("Reset complete!")

def main():
    parser = argparse.ArgumentParser(description="Black Box Trading System")
    parser.add_argument("--reset", action="store_true", help="Reset workspace")
    args = parser.parse_args()

    if args.reset:
        reset_workspace()
        print("Workspace reset complete")
        return

    # Ensure directories exist
    for d in ["models", "data", "logs"]:
        Path(d).mkdir(parents=True, exist_ok=True)

    setup_logging()
    
    system = TradingSystem()
    
    def shutdown_handler(signum, frame):
        system.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    try:
        system.start()
    except Exception as e:
        logging.error(f"System failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()