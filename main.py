# main.py

import argparse

from utils.logging_config import setup_logging
from core.runner import Runner

setup_logging()

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--reset", action="store_true", help="Force full model retraining")
    return parser.parse_args()

def main():
    args = parse_args()
    runner = Runner(args)
    runner.run()

if __name__ == "__main__":
    main()  