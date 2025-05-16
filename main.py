# main.py

from config import Config
from model.agent import RLAgent
from utils.tcp_bridge import TCPBridge
from utils.io_utils import safe_read_csv, clean_feature_file

import os, time, logging
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[logging.StreamHandler()]
)

log = logging.getLogger(__name__)

def main():
    cfg    = Config()
    agent  = RLAgent(cfg)
    tcp    = TCPBridge()

    os.makedirs(os.path.dirname(cfg.FEATURE_FILE), exist_ok=True)
    rows, last_price, trained = [], None, False
    last_sent_ts = -1 

    def handle_feat(feat, live):
        nonlocal rows, last_price, trained, last_sent_ts

        close = feat[0]
        atr = feat[2] if len(feat) > 2 else 0.01
        price_change = 0.0 if last_price is None else close - last_price
        reward = 0.0 if last_price is None else agent.calculate_improved_reward(price_change, atr)
        last_price = close
        rows.append([time.time(), *feat, reward])

        if live == 0:
            return

        if not trained:
            log.info("Initial backfill training…")
            df = pd.DataFrame(rows, columns=["ts", "close", "volume", "atr", "reward"])
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

        sig = {
            "action": action,
            "confidence": conf,
            "size": max(cfg.MIN_SIZE,
                        int(conf * (cfg.BASE_SIZE if action != 1 else cfg.CONS_SIZE))),
            "timestamp": now_ts
        }
        tcp.send_signal(sig)
        log.info("Sent signal %s", sig)

        if len(rows) >= cfg.BATCH_SIZE:
            df = pd.DataFrame(rows, columns=["ts", "close", "volume", "atr", "reward"])
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
            df = pd.DataFrame(rows, columns=["ts", "close", "volume", "atr", "reward"])
            header = not os.path.exists(cfg.FEATURE_FILE)
            df.to_csv(cfg.FEATURE_FILE, mode='a', header=header, index=False)
            agent.train(df, epochs=1)
            agent.save_model()

if __name__ == "__main__":
    main()
