# main.py

from config import Config
from model.agent import RLAgent
from utils.tcp_bridge import TCPBridge
from utils.tick_processor import TickProcessor
from utils.portfolio import Portfolio

import os, time, logging
import pandas as pd
import threading
import argparse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[logging.StreamHandler()]
)

log = logging.getLogger(__name__)

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Force full model retraining")
    args = parser.parse_args()
    cfg    = Config()
    portfolio = Portfolio(cfg.MAX_SIZE)

    tick_processor = TickProcessor(host="localhost")
    tick_thread = threading.Thread(target=tick_processor.accept_connections, daemon=True)
    tick_thread.start()
    if not tick_processor.wait_until_ready():
        raise RuntimeError("Tick processor failed to start")
    
    tcp = TCPBridge("localhost", 5556, 5557)
    agent  = RLAgent(cfg)

    os.makedirs(os.path.dirname(cfg.FEATURE_FILE), exist_ok=True)
    rows, last_price, trained = [], None, False
    last_sent_ts = -1 

    def handle_feat(feat, live):
        log.debug(f"Received features: {feat} | live={live}")
        nonlocal rows, last_price, trained, last_sent_ts

        close = feat[0]
        atr = feat[2] if len(feat) > 2 else 0.01
        price_change = 0.0 if last_price is None else close - last_price
        reward = 0.0 if last_price is None else agent.calculate_improved_reward(price_change, atr)
        last_price = close
        lwpe = feat[3] if len(feat) > 3 else 0.5
        log.debug(f"LWPE = {lwpe:.4f}")
        rows.append([time.time(), *feat, reward])

        if live == 0:
            return

        if not trained or args.reset:
            log.info("Initial backfill training…")
            df = pd.DataFrame(rows, columns=["ts", "close", "volume", "atr", "lwpe", "reward"])
            agent.train(df, epochs=3)
            agent.save_model()
            rows.clear()
            trained = True
            return

        action, conf = agent.predict_single(feat)
        if action == 1:
            reward -= 0.01
        rows[-1][-1] = reward

        agent.push_sample(feat, action, reward)

        now_ts = int(time.time())
        if now_ts == last_sent_ts:
            return
        last_sent_ts = now_ts

        current_pos = portfolio.get_current_position()
        portfolio.update_position(current_pos)
        desired_size = int(conf * cfg.BASE_SIZE)
        adjusted_size = portfolio.adjust_size(action, desired_size)

        sig = {
            "action": action,
            "confidence": conf,
            "size": max(cfg.MIN_SIZE, adjusted_size) if adjusted_size > 0 else 0,
            "timestamp": now_ts
        }
        tcp.send_signal(sig)
        log.info("Sent signal %s", sig)

        if len(rows) >= cfg.BATCH_SIZE:
            df = pd.DataFrame(rows, columns=["ts", "close", "volume", "atr", "lwpe", "reward"])
            agent.train(df, epochs=1)
            agent.save_model()
            df.to_csv(cfg.FEATURE_FILE,
                      mode="a",
                      header=not os.path.exists(cfg.FEATURE_FILE),
                      index=False)
            rows.clear()

    tcp.on_features = handle_feat

    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        tcp.close()
        log.info("Session terminated by user")
    finally:
        if rows:
            df = pd.DataFrame(rows, columns=["ts", "close", "volume", "atr", "lwpe", "reward"])
            header = not os.path.exists(cfg.FEATURE_FILE)
            df.to_csv(cfg.FEATURE_FILE, mode='a', header=header, index=False)
            agent.train(df, epochs=1)
            agent.save_model()

if __name__ == "__main__":
    main()
