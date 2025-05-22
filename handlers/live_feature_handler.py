# handlers/live_feature_handler.py

import logging

from model.trainer import Trainer
from model.reward import RewardCalculator
from services.feature_processor import FeatureProcessor
from handlers.live_signal_handler import LiveSignalHandler

log = logging.getLogger(__name__)

class LiveFeatureHandler:
    def __init__(self, cfg, agent, portfolio, logger, tcp, args):
        self.rewarder = RewardCalculator()
        self.processor = FeatureProcessor(agent, self.rewarder)
        self.trainer = Trainer(cfg, agent, logger, args)
        self.dispatcher = LiveSignalHandler(cfg, agent, portfolio, tcp, logger)
        self.step_counter = 0

    def handle_live_feature(self, feat, live):
        self.step_counter += 1
        
        # Process features and get prediction
        row, action, conf, close = self.processor.process_and_predict(feat)
        self.trainer.append(row)
        self.trainer.last_price = close

        # Handle historical data (live=0)
        if live == 0:
            return

        # Initial training on historical data
        if self.trainer.should_train_initial():
            log.info("Performing initial training before live trading")
            self.trainer.perform_initial_training()
            return

        # Check if ready for trading
        if not self.trainer.is_ready_for_trading():
            log.debug("Not ready for trading yet")
            return

        # Dispatch trading signal
        self.dispatcher.dispatch_signal(action, conf)

        # Online learning - train frequently on recent experiences
        if self.trainer.should_train_online():
            self.trainer.train_online()

        # Periodic batch training on logged data
        if self.trainer.should_train_batch():
            log.info("Performing batch training on accumulated data")
            self.trainer.train_batch()

        # Periodic logging
        if self.step_counter % 100 == 0:
            log.info(f"Processed {self.step_counter} live features, "
                    f"buffer size: {len(self.trainer.agent.experience_buffer)}")