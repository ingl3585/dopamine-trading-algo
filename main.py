# main.py

from config import Config
from model.agent import RLAgent
from utils.tcp_bridge import TCPBridge
from utils.tick_processor import TickProcessor
from utils.portfolio import Portfolio
from utils.market import MarketUtils

import os
import time
import logging
import numpy as np
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
    log.info("TCP bridge established, waiting for first feature...")
    agent  = RLAgent(cfg)

    os.makedirs(os.path.dirname(cfg.FEATURE_FILE), exist_ok=True)
    rows, last_price, trained = [], None, False
    last_sent_ts = -1 

    prev_lwpe = None
    price_history = []
    MAX_HISTORY = 100

    def handle_feat(feat, live):
        log.info(f"Feature received: close={feat[0]:.2f}, volume={feat[1]:.2f}, atr={feat[2]:.4f}")
        nonlocal rows, last_price, trained, last_sent_ts, price_history, prev_lwpe

        close = feat[0]
        atr = feat[2] if len(feat) > 2 else 0.01
        lwpe = feat[3] if len(feat) > 3 else 0.5
        delta_lwpe = 0.0 if prev_lwpe is None else lwpe - prev_lwpe
        prev_lwpe = lwpe

        price_history.append(close)
        if len(price_history) > MAX_HISTORY:
            price_history = price_history[-MAX_HISTORY:]

        regime = MarketUtils.detect_regime(price_history)
        volatility = MarketUtils.forecast_volatility(price_history)

        price_change = 0.0 if last_price is None else close - last_price
        reward = 0.0 if last_price is None else agent.calculate_improved_reward(price_change, atr)
        last_price = close

        log.debug(f"LWPE = {lwpe:.4f}, Regime = {regime}, Volatility = {volatility:.4f}")
        full_feat = [close, feat[1], atr, lwpe, delta_lwpe, volatility, regime]
        rows.append([time.time(), *full_feat, reward])

        if live == 0:
            return

        log.info("Checking if model needs initial training...")
        if not trained or args.reset:
            log.info("Initial backfill training")
            df = pd.DataFrame(rows, columns=["ts", "close", "volume", "atr", "lwpe", "delta_lwpe", "volatility", "regime", "reward"])
            log.info(f"Starting model training with {len(df)} samples")
            agent.train(df, epochs=3)
            agent.save_model()
            rows.clear()
            trained = True
            args.reset = False
            return

        action, conf = agent.predict_signal(full_feat)

        vol_damp = np.clip(np.log1p(volatility), 0, 1.0)
        regime_scale = {0: 1.0, 1: 0.5}.get(regime, 1.0)
        scaled_conf = conf * regime_scale * (1.0 - 0.7 * vol_damp)

        log.info(f"Raw conf={conf:.3f}, regime={regime}, volatility={volatility:.4f}, scaled_conf={scaled_conf:.3f}")
        conf = max(0.0, min(1.0, scaled_conf))

        if action == 1:
            reward -= 0.01
        rows[-1][-1] = reward
        agent.push_sample(full_feat, action, reward)

        now_ts = int(time.time())
        if now_ts == last_sent_ts:
            return
        last_sent_ts = now_ts

        portfolio.update_position(tcp._current_position)
        log.info(f"Updated position from NinjaTrader: {tcp._current_position}")
        desired_size = int(conf * cfg.BASE_SIZE)
        adjusted_size = portfolio.adjust_size(action, desired_size)
        if adjusted_size <= 0:
            action = 0
            conf = 0.0

        sig = {
            "action": action,
            "confidence": conf,
            "size": 0,
            "timestamp": now_ts
        }

        log.info(f"[Position Check] Current pos: {portfolio.position}, Action: {action}, Desired size: {desired_size}, Adjusted size: {adjusted_size}")
        if portfolio.can_execute(action, adjusted_size):
            sig["size"] = max(cfg.MIN_SIZE, adjusted_size) if adjusted_size > 0 else 0
        else:
            log.info("Position cap reached, skipping trade")

        tcp.send_signal(sig)
        log.info("Sent signal %s", sig)

        if len(rows) >= cfg.BATCH_SIZE:
            df = pd.DataFrame(rows, columns=["ts", "close", "volume", "atr", "lwpe", "delta_lwpe", "volatility", "regime", "reward"])
            log.info(f"Starting model training with {len(df)} samples")
            agent.train(df, epochs=1)
            agent.save_model()
            df.to_csv(cfg.FEATURE_FILE,
                    mode="a",
                    header=not os.path.exists(cfg.FEATURE_FILE),
                    index=False)
            rows.clear()

    tcp.on_features = handle_feat
    log.info("Feature handler assigned and ready to receive data")

    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        tcp.close()
        log.info("Session terminated by user")
    finally:
        if rows:
            df = pd.DataFrame(rows, columns=["ts", "close", "volume", "atr", "lwpe", "delta_lwpe", "volatility", "regime", "reward"])
            header = not os.path.exists(cfg.FEATURE_FILE)
            df.to_csv(cfg.FEATURE_FILE, mode='a', header=header, index=False)
            log.info(f"Starting model training with {len(df)} samples")
            agent.train(df, epochs=1)
            agent.save_model()

if __name__ == "__main__":
    main()
