# handlers/live_feature_handler.py

from model.trainer import Trainer
from model.reward import RewardCalculator
from services.feature_processor import FeatureProcessor
from handlers.live_signal_handler import LiveSignalHandler

class LiveFeatureHandler:
    def __init__(self, cfg, agent, portfolio, logger, tcp, args):
        self.rewarder = RewardCalculator()
        self.processor = FeatureProcessor(agent, self.rewarder)
        self.trainer = Trainer(cfg, agent, logger, args)
        self.dispatcher = LiveSignalHandler(cfg, agent, portfolio, tcp, logger)

    def handle_live_feature(self, feat, live):
        row, action, conf, close = self.processor.process_and_predict(feat)
        self.trainer.append(row)
        self.trainer.last_price = close

        if live == 0:
            return

        if self.trainer.should_train_initial():
            self.trainer.perform_initial_training()
            return

        self.dispatcher.dispatch_signal(action, conf)

        if self.trainer.should_train_batch():
            self.trainer.train_batch()