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

        row, action, conf, close = self.processor.process_and_predict(feat)
        self.trainer.append(row)
        self.trainer.last_price = close

        if live == 0:
            return

        if self.trainer.should_train_initial():
            log.info("Performing initial training before live trading")
            self.trainer.perform_initial_training()
            return

        if not self.trainer.is_ready_for_trading():
            log.debug("Not ready for trading yet")
            return

        self.dispatcher.dispatch_signal(action, conf)

        if self.trainer.should_train_online():
            self.trainer.train_online()

        if self.trainer.should_train_batch():
            log.info("Performing batch training on accumulated data")
            self.trainer.train_batch()

        if self.step_counter % 100 == 0:
            log.info(f"Processed {self.step_counter} live features, "
                    f"buffer size: {len(self.trainer.agent.experience_buffer)}")