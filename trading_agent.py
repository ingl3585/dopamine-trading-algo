# trading_agent.py

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import time
import logging

from collections import deque
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

from intelligence_engine import Features
from data_processor import MarketData
from meta_learner import MetaLearner
from adaptive_network import AdaptiveTradingNetwork, FeatureLearner, StateEncoder
from enhanced_neural import SelfEvolvingNetwork, FewShotLearner
from real_time_adaptation import RealTimeAdaptationEngine

logger = logging.getLogger(__name__)

@dataclass
class Decision:
    action: str
    confidence: float
    size: float
    stop_price: float = 0.0
    target_price: float = 0.0
    primary_tool: str = 'unknown'
    exploration: bool = False
    intelligence_data: Dict = None
    state_features: List = None
    # Enhanced decision data
    adaptation_strategy: str = 'conservative'
    uncertainty_estimate: float = 0.5
    few_shot_prediction: float = 0.0
    regime_awareness: Dict = None


class TradingAgent:
    def __init__(self, intelligence, portfolio):
        self.intelligence = intelligence
        self.portfolio = portfolio
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Enhanced meta-learning system
        self.meta_learner = MetaLearner(state_dim=64)  # Increased state dimension
        
        # Enhanced neural architecture with self-evolution
        initial_sizes = self.meta_learner.architecture_evolver.current_sizes
        self.network = SelfEvolvingNetwork(
            input_size=64,
            initial_sizes=initial_sizes,
            evolution_frequency=500
        ).to(self.device)

        self.target_network = SelfEvolvingNetwork(
            input_size=64,
            initial_sizes=initial_sizes
        ).to(self.device)
        
        # Enhanced feature learning with catastrophic forgetting prevention
        self.feature_learner = FeatureLearner(
            raw_feature_dim=100, 
            learned_feature_dim=64,
        ).to(self.device)
        self.state_encoder = StateEncoder()
        
        # Few-shot learning capability
        self.few_shot_learner = FewShotLearner(feature_dim=64).to(self.device)
        
        # Real-time adaptation integration
        self.adaptation_engine = RealTimeAdaptationEngine(model_dim=64)
        
        # Enhanced optimizer with learning rate scheduling
        self.optimizer = optim.AdamW(
            list(self.network.parameters()) + 
            list(self.feature_learner.parameters()) + 
            list(self.few_shot_learner.parameters()) +
            list(self.meta_learner.subsystem_weights.parameters()) +
            list(self.meta_learner.exploration_strategy.parameters()),
            lr=0.001,
            weight_decay=1e-5
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.8, patience=50
        )
        
        # Enhanced experience replay with prioritization
        self.experience_buffer = deque(maxlen=20000)  # Increased buffer size
        self.priority_buffer = deque(maxlen=5000)  # High priority experiences
        
        # Catastrophic forgetting prevention
        self.previous_task_buffer = deque(maxlen=1000)
        self.importance_weights = {}
        
        # Enhanced statistics and tracking
        self.total_decisions = 0
        self.successful_trades = 0
        self.total_pnl = 0.0
        self.last_trade_time = 0.0
        self.regime_transitions = 0
        self.adaptation_events = 0
        
        # Performance tracking for different strategies
        self.strategy_performance = {
            'conservative': deque(maxlen=100),
            'aggressive': deque(maxlen=100),
            'momentum': deque(maxlen=100),
            'mean_reversion': deque(maxlen=100),
            'adaptive': deque(maxlen=100)
        }
        
        # Model ensemble for uncertainty quantification
        self.ensemble_predictions = deque(maxlen=10)
    
    def decide(self, features: Features, market_data: MarketData) -> Decision:
        self.total_decisions += 1
        
        # Check for architecture evolution
        if self.meta_learner.should_evolve_architecture():
            self._evolve_architecture()
        
        # Enhanced state representation with microstructure features
        meta_context = self._get_enhanced_meta_context(market_data, features)
        raw_state = self._create_enhanced_state(market_data, features, meta_context)
        
        # Learn features with catastrophic forgetting prevention
        learned_state = self.feature_learner(raw_state.unsqueeze(0).to(self.device))
        
        # Few-shot learning prediction
        few_shot_prediction = self.few_shot_learner(learned_state)
        
        # Enhanced subsystem contributions with explicit dtype
        subsystem_signals = torch.tensor([
            float(features.dna_signal),
            float(features.temporal_signal),
            float(features.immune_signal),
            float(features.microstructure_signal),  # Enhanced with microstructure
            float(features.regime_adjusted_signal)
        ], dtype=torch.float64, device=self.device)
        
        subsystem_weights = self.meta_learner.get_subsystem_weights()
        # Expand weights to match signals if needed
        if len(subsystem_weights) < len(subsystem_signals):
            additional_weights = torch.ones(len(subsystem_signals) - len(subsystem_weights), dtype=torch.float64, device=self.device) * 0.1
            subsystem_weights = torch.cat([subsystem_weights, additional_weights])
        
        weighted_signal = torch.sum(subsystem_signals * subsystem_weights[:len(subsystem_signals)])
        
        # Real-time adaptation decision
        adaptation_context = {
            'volatility': features.volatility,
            'trend_strength': abs(features.price_momentum),
            'volume_regime': features.volume_momentum + 0.5,
            'time_of_day': features.time_of_day,
            'regime_confidence': features.regime_confidence
        }
        
        adaptation_decision = self.adaptation_engine.get_adaptation_decision(
            learned_state.squeeze(), adaptation_context
        )
        
        # Enhanced trading constraints with regime awareness
        if not self._should_consider_trading(market_data, meta_context, features):
            return Decision('hold', 0, 0, regime_awareness=adaptation_context)
        
        # Enhanced neural network decision with ensemble
        ensemble_outputs = []
        
        with torch.no_grad():
            # Main network prediction
            main_outputs = self.network(learned_state)
            ensemble_outputs.append(main_outputs)
            
            # Target network prediction for ensemble
            target_outputs = self.target_network(learned_state)
            ensemble_outputs.append(target_outputs)
            
            # Combine ensemble predictions
            combined_outputs = {}
            for key in main_outputs.keys():
                combined_outputs[key] = torch.mean(torch.stack([out[key] for out in ensemble_outputs]), dim=0)
            
            # Get action probabilities with temperature scaling
            temperature = 1.0 + features.volatility * 2.0  # Higher temperature for higher volatility
            action_logits = combined_outputs['action_logits'] / temperature
            action_probs = F.softmax(action_logits, dim=-1).detach().cpu().numpy()[0]
            confidence = float(combined_outputs['confidence'].detach().cpu().numpy()[0])
            
            # Enhanced exploration decision with adaptation strategy
            should_explore = self.meta_learner.should_explore(
                learned_state.squeeze(), 
                meta_context
            ) or adaptation_decision.get('emergency_mode', False)
            
            # Strategy selection based on adaptation engine
            selected_strategy = adaptation_decision.get('strategy_name', 'conservative')
            
            if should_explore or adaptation_decision.get('emergency_mode', False):
                # Enhanced exploration with strategy awareness
                action_idx = self._strategic_exploration(
                    weighted_signal, selected_strategy, features, adaptation_decision
                )
                exploration = True
            else:
                # Exploitation with uncertainty consideration
                uncertainty = adaptation_decision.get('uncertainty', 0.5)
                if uncertainty > 0.7:  # High uncertainty, be more conservative
                    action_probs[0] *= 1.5  # Boost hold probability
                    action_probs = action_probs / np.sum(action_probs)  # Renormalize
                
                action_idx = np.argmax(action_probs)
                exploration = False
            
            # Enhanced confidence threshold with regime awareness
            base_threshold = self.meta_learner.get_parameter('confidence_threshold')
            regime_adjustment = (1.0 - features.regime_confidence) * 0.2
            volatility_adjustment = features.volatility * 0.3
            confidence_threshold = base_threshold + regime_adjustment + volatility_adjustment
            
            if confidence < confidence_threshold and not exploration:
                action_idx = 0
            
            # Enhanced sizing and risk parameters
            position_size = float(combined_outputs['position_size'].detach().cpu().numpy()[0])
            risk_params = combined_outputs['risk_params'].detach().cpu().numpy()[0]
            
            # Uncertainty-adjusted position sizing
            uncertainty_factor = 1.0 - adaptation_decision.get('uncertainty', 0.5)
            position_size *= uncertainty_factor
            
            # Regime-aware risk management
            use_stop = risk_params[0] > 0.5
            stop_distance = risk_params[1]
            use_target = risk_params[2] > 0.5
            target_distance = risk_params[3]
            
            # Adjust risk parameters based on regime
            if features.regime_confidence < 0.5:  # Uncertain regime
                stop_distance *= 0.8  # Tighter stops
                target_distance *= 1.2  # Wider targets
        
        # Convert to decision
        actions = ['hold', 'buy', 'sell']
        action = actions[action_idx]
        
        if action == 'hold':
            return Decision(
                'hold', confidence, 0,
                adaptation_strategy=selected_strategy,
                uncertainty_estimate=adaptation_decision.get('uncertainty', 0.5),
                few_shot_prediction=float(few_shot_prediction.squeeze()),
                regime_awareness=adaptation_context
            )
        
        # Enhanced stop and target calculation with regime awareness
        stop_price, target_price = self._calculate_enhanced_levels(
            action, market_data, use_stop, stop_distance, use_target, target_distance, features
        )
        
        # Determine primary tool with enhanced attribution
        primary_tool = self._get_enhanced_primary_tool(subsystem_signals, subsystem_weights, features)
        
        # Store enhanced intelligence data
        intelligence_data = {
            'subsystem_signals': subsystem_signals.detach().cpu().numpy().tolist(),
            'subsystem_weights': subsystem_weights.detach().cpu().numpy().tolist(),
            'weighted_signal': float(weighted_signal),
            'adaptation_decision': adaptation_decision,
            'few_shot_prediction': float(few_shot_prediction.squeeze()),
            'ensemble_uncertainty': self._calculate_ensemble_uncertainty(ensemble_outputs),
            'regime_context': adaptation_context,
            'current_patterns': getattr(features, 'current_patterns', {})
        }
        
        self.last_trade_time = market_data.timestamp
        
        # Store few-shot learning example
        self.few_shot_learner.add_support_example(learned_state.squeeze(), weighted_signal.detach().item())
        
        return Decision(
            action=action,
            confidence=confidence,
            size=position_size,
            stop_price=stop_price,
            target_price=target_price,
            primary_tool=primary_tool,
            exploration=exploration,
            intelligence_data=intelligence_data,
            state_features=learned_state.squeeze().detach().cpu().numpy().tolist(),
            adaptation_strategy=selected_strategy,
            uncertainty_estimate=adaptation_decision.get('uncertainty', 0.5),
            few_shot_prediction=float(few_shot_prediction.squeeze()),
            regime_awareness=adaptation_context
        )
    
    def _strategic_exploration(self, weighted_signal: torch.Tensor, strategy: str, 
                             features: Features, adaptation_decision: Dict) -> int:
        """Enhanced exploration based on selected strategy"""
        
        if strategy == 'conservative':
            # Conservative exploration - only trade with strong signals
            if abs(weighted_signal) > 0.3:
                return 1 if weighted_signal > 0 else 2
            return 0
        
        elif strategy == 'aggressive':
            # Aggressive exploration - trade on weaker signals
            if abs(weighted_signal) > 0.1:
                return 1 if weighted_signal > 0 else 2
            return 0
        
        elif strategy == 'momentum':
            # Momentum strategy - follow trends
            if features.price_momentum > 0.01:
                return 1
            elif features.price_momentum < -0.01:
                return 2
            return 0
        
        elif strategy == 'mean_reversion':
            # Mean reversion - trade against extremes
            if features.price_position > 0.8:
                return 2  # Sell at highs
            elif features.price_position < 0.2:
                return 1  # Buy at lows
            return 0
        
        else:  # adaptive
            # Adaptive strategy based on current conditions
            if adaptation_decision.get('emergency_mode', False):
                return 0  # Hold in emergency
            elif features.regime_confidence > 0.7:
                # High confidence regime - follow signals
                return 1 if weighted_signal > 0.1 else (2 if weighted_signal < -0.1 else 0)
            else:
                # Low confidence regime - be conservative
                return 1 if weighted_signal > 0.2 else (2 if weighted_signal < -0.2 else 0)
    
    def _calculate_enhanced_levels(self, action: str, market_data: MarketData, 
                                 use_stop: bool, stop_distance: float,
                                 use_target: bool, target_distance: float,
                                 features: Features) -> tuple:
        """Enhanced stop and target calculation with regime awareness"""
        
        # Base calculation from meta-learner
        stop_distance_factor = self.meta_learner.get_parameter('stop_distance_factor')
        target_distance_factor = self.meta_learner.get_parameter('target_distance_factor')
        
        # Regime adjustments
        regime_multiplier = 1.0
        if features.regime_confidence < 0.5:
            regime_multiplier = 0.8  # Tighter levels in uncertain regimes
        elif features.volatility > 0.03:
            regime_multiplier = 1.3  # Wider levels in high volatility
        
        stop_price = 0.0
        target_price = 0.0
        
        if use_stop:
            adjusted_distance = stop_distance_factor * (1 + stop_distance) * regime_multiplier
            if action == 'buy':
                stop_price = market_data.price * (1 - adjusted_distance)
            else:
                stop_price = market_data.price * (1 + adjusted_distance)
        
        if use_target:
            adjusted_distance = target_distance_factor * (1 + target_distance) * regime_multiplier
            if action == 'buy':
                target_price = market_data.price * (1 + adjusted_distance)
            else:
                target_price = market_data.price * (1 - adjusted_distance)
        
        return stop_price, target_price
    
    def _calculate_ensemble_uncertainty(self, ensemble_outputs: List[Dict]) -> float:
        """Calculate uncertainty from ensemble predictions"""
        if len(ensemble_outputs) < 2:
            return 0.5
        
        # Calculate variance in action predictions
        action_logits = [out['action_logits'] for out in ensemble_outputs]
        action_probs = [F.softmax(logits, dim=-1) for logits in action_logits]
        
        # Stack and calculate variance
        prob_stack = torch.stack(action_probs)
        prob_variance = torch.var(prob_stack, dim=0)
        uncertainty = torch.mean(prob_variance).detach().item()
        
        return min(1.0, uncertainty * 5.0)  # Scale to [0, 1]
    
    def _create_enhanced_state(self, market_data: MarketData, features: Features, 
                             meta_context: Dict) -> torch.Tensor:
        """Create enhanced state representation"""
        
        # Base state from encoder
        base_state = self.state_encoder.create_full_state(market_data, features, meta_context)
        
        # Enhanced features with explicit float conversion
        enhanced_features = torch.tensor([
            # Microstructure features
            float(features.microstructure_signal),
            float(features.regime_adjusted_signal),
            float(features.smart_money_flow),
            float(features.liquidity_depth),
            float(features.regime_confidence),
            
            # Regime and adaptation features
            float(features.adaptation_quality),
            float(meta_context.get('regime_transitions', 0.0)),
            float(meta_context.get('adaptation_events', 0.0)),
            
            # Account context
            float(min(1.0, market_data.account_balance / 50000)),
            float(market_data.margin_utilization),
            float(market_data.buying_power_ratio),
            
            # Time-based features
            float(np.sin(2 * np.pi * features.time_of_day)),  # Cyclical time
            float(np.cos(2 * np.pi * features.time_of_day)),
            
            # Volatility regime features
            float(min(1.0, features.volatility / 0.05)),  # Normalized volatility
            float(np.tanh(features.price_momentum * 10)),  # Bounded momentum
            
            # Pattern strength indicators
            float(features.pattern_score),
            float(features.confidence),
            
            # Cross-timeframe features (if available)
            float(getattr(features, 'tf_5m_momentum', 0.0)),
            float(getattr(features, 'tf_15m_momentum', 0.0))
        ], dtype=torch.float64, device=self.device)
        
        # Combine base state with enhanced features
        full_state = torch.cat([base_state, enhanced_features])
        
        # Pad or truncate to exactly 100 features
        if len(full_state) < 100:
            padding = torch.zeros(100 - len(full_state), dtype=torch.float64, device=self.device)
            full_state = torch.cat([full_state, padding])
        else:
            full_state = full_state[:100]
        
        return full_state
    
    def _get_enhanced_meta_context(self, market_data: MarketData, features: Features) -> Dict[str, float]:
        """Enhanced meta context with regime and adaptation awareness"""
        portfolio_summary = self.portfolio.get_summary()
        
        base_context = {
            'recent_performance': np.tanh(portfolio_summary.get('daily_pnl', 0) / (market_data.account_balance * 0.01)),
            'consecutive_losses': portfolio_summary.get('consecutive_losses', 0),
            'position_count': portfolio_summary.get('pending_orders', 0),
            'trades_today': portfolio_summary.get('total_trades', 0),
            'time_since_last_trade': 0.0 if self.last_trade_time == 0 else (np.log(1 + (time.time() - self.last_trade_time) / 3600)),
            'learning_efficiency': self.meta_learner.get_learning_efficiency(),
            'architecture_generation': self.meta_learner.architecture_evolver.generations
        }
        
        # Enhanced context
        enhanced_context = {
            'regime_confidence': features.regime_confidence,
            'microstructure_strength': abs(features.microstructure_signal),
            'adaptation_quality': features.adaptation_quality,
            'regime_transitions': self.regime_transitions / max(1, self.total_decisions),
            'adaptation_events': self.adaptation_events / max(1, self.total_decisions),
            'volatility_regime': min(1.0, features.volatility / 0.05),
            'liquidity_regime': features.liquidity_depth,
            'smart_money_activity': abs(features.smart_money_flow)
        }
        
        return {**base_context, **enhanced_context}
    
    def _should_consider_trading(self, market_data: MarketData, meta_context: Dict, 
                               features: Features) -> bool:
        """Enhanced trading constraints with regime awareness"""
        
        # Base constraints from meta-learner
        loss_tolerance = self.meta_learner.get_parameter('loss_tolerance_factor')
        max_loss = market_data.account_balance * loss_tolerance
        if market_data.daily_pnl <= -max_loss:
            return False
        
        consecutive_limit = self.meta_learner.get_parameter('consecutive_loss_tolerance')
        if meta_context['consecutive_losses'] >= consecutive_limit:
            return False
        
        frequency_limit = self.meta_learner.get_parameter('trade_frequency_base')
        time_since_last = market_data.timestamp - self.last_trade_time
        if time_since_last < (300 / frequency_limit):
            return False
        
        # Enhanced regime-based constraints
        if features.regime_confidence < 0.3:  # Very uncertain regime
            return False
        
        if features.volatility > 0.08:  # Extreme volatility
            return False
        
        if features.liquidity_depth < 0.2:  # Poor liquidity
            return False
        
        # Adaptation engine emergency mode
        if meta_context.get('adaptation_events', 0) > 0.1:  # Too many adaptation events
            return False
        
        return True
    
    def _get_enhanced_primary_tool(self, signals: torch.Tensor, weights: torch.Tensor, 
                                 features: Features) -> str:
        """Enhanced primary tool identification"""
        weighted_signals = torch.abs(signals * weights[:len(signals)])
        tool_names = ['dna', 'temporal', 'immune', 'microstructure', 'regime']
        
        if torch.sum(weighted_signals) == 0:
            return 'basic'
        
        primary_idx = torch.argmax(weighted_signals)
        
        # Add context about why this tool was primary
        primary_tool = tool_names[min(primary_idx, len(tool_names) - 1)]
        
        # Enhanced attribution
        if features.regime_confidence < 0.5:
            primary_tool += '_uncertain'
        elif features.volatility > 0.05:
            primary_tool += '_highvol'
        elif abs(features.smart_money_flow) > 0.3:
            primary_tool += '_smartmoney'
        
        return primary_tool
    
    def learn_from_trade(self, trade):
        """Enhanced learning with catastrophic forgetting prevention"""
        if not hasattr(trade, 'intelligence_data'):
            return
        
        # Update statistics
        if trade.pnl > 0:
            self.successful_trades += 1
        self.total_pnl += trade.pnl
        
        # Track strategy performance
        strategy = getattr(trade, 'adaptation_strategy', 'conservative')
        if strategy in self.strategy_performance:
            self.strategy_performance[strategy].append(trade.pnl)
        
        # Enhanced trade data for meta-learning
        trade_data = {
            'pnl': trade.pnl,
            'account_balance': getattr(trade.market_data, 'account_balance', 25000),
            'hold_time': trade.exit_time - trade.entry_time,
            'was_exploration': getattr(trade, 'exploration', False),
            'subsystem_contributions': torch.tensor(trade.intelligence_data.get('subsystem_signals', [0,0,0,0,0]), dtype=torch.float64),
            'subsystem_agreement': self._calculate_enhanced_subsystem_agreement(trade.intelligence_data),
            'confidence': getattr(trade, 'confidence', 0.5),
            'primary_tool': getattr(trade, 'primary_tool', 'unknown'),
            'stop_used': getattr(trade, 'stop_used', False),
            'target_used': getattr(trade, 'target_used', False),
            'adaptation_strategy': getattr(trade, 'adaptation_strategy', 'conservative'),
            'uncertainty_estimate': getattr(trade, 'uncertainty_estimate', 0.5),
            'regime_confidence': trade.intelligence_data.get('regime_context', {}).get('regime_confidence', 0.5)
        }
        
        # Compute enhanced reward
        reward = self.meta_learner.compute_reward(trade_data)
        
        # Store experience with priority
        if hasattr(trade, 'state_features') and trade.state_features:
            experience = {
                'state_features': trade.state_features,
                'action': ['hold', 'buy', 'sell'].index(trade.action),
                'reward': reward,
                'done': True,
                'trade_data': trade_data,
                'uncertainty': trade_data['uncertainty_estimate'],
                'regime_confidence': trade_data['regime_confidence']
            }
            
            # Prioritize unusual or high-impact experiences
            if abs(reward) > 0.5 or trade_data['uncertainty_estimate'] > 0.7:
                self.priority_buffer.append(experience)
            else:
                self.experience_buffer.append(experience)
        
        # Catastrophic forgetting prevention
        if len(self.experience_buffer) > 1000:
            # Store some experiences as previous task data
            sample_size = min(50, len(self.experience_buffer) // 20)
            sampled_experiences = np.random.choice(
                list(self.experience_buffer), size=sample_size, replace=False
            )
            for exp in sampled_experiences:
                self.previous_task_buffer.append(exp)
                # Add to network's previous task data with device specification
                state_tensor = torch.tensor(exp['state_features'], dtype=torch.float64, device=self.device)
                target_tensor = torch.tensor([exp['reward']], dtype=torch.float64, device=self.device)
                self.network.add_previous_task_data(state_tensor, target_tensor)
        
        # Meta-learning update
        self.meta_learner.learn_from_outcome(trade_data)
        
        # Real-time adaptation update
        adaptation_context = trade.intelligence_data.get('regime_context', {})
        adaptation_context['predicted_confidence'] = trade_data['confidence']
        self.adaptation_engine.update_from_outcome(reward, adaptation_context)
        
        # Network performance tracking
        self.network.record_performance(reward)
        
        # Train networks if enough experience
        if len(self.experience_buffer) >= 64 or len(self.priority_buffer) >= 32:
            self._train_enhanced_networks()
        
        # Update learning rate based on performance
        recent_rewards = [exp['reward'] for exp in list(self.experience_buffer)[-20:]]
        if len(recent_rewards) >= 10:
            avg_reward = np.mean(recent_rewards)
            self.scheduler.step(avg_reward)
        
        # Periodic parameter adaptation
        if self.total_decisions % 50 == 0:
            self.meta_learner.adapt_parameters()
    
    def _calculate_enhanced_subsystem_agreement(self, intelligence_data: Dict) -> float:
        """Enhanced subsystem agreement calculation"""
        signals = intelligence_data.get('subsystem_signals', [0, 0, 0, 0, 0])
        
        if not signals or all(s == 0 for s in signals):
            return 0.5
        
        # Enhanced agreement calculation with uncertainty
        positive_signals = sum(1 for s in signals if s > 0.1)
        negative_signals = sum(1 for s in signals if s < -0.1)
        neutral_signals = sum(1 for s in signals if abs(s) <= 0.1)
        total_signals = len([s for s in signals if abs(s) > 0.05])  # Lower threshold
        
        if total_signals == 0:
            return 0.5
        
        # Directional agreement
        directional_agreement = max(positive_signals, negative_signals) / len(signals)
        
        # Magnitude agreement
        signal_magnitudes = [abs(s) for s in signals if abs(s) > 0.05]
        if len(signal_magnitudes) > 1:
            magnitude_std = np.std(signal_magnitudes)
            magnitude_agreement = 1.0 / (1.0 + magnitude_std)
        else:
            magnitude_agreement = 1.0
        
        # Combined agreement
        return directional_agreement * 0.7 + magnitude_agreement * 0.3
    
    def _train_enhanced_networks(self):
        """Enhanced network training with catastrophic forgetting prevention"""
        total_experiences = len(self.experience_buffer) + len(self.priority_buffer)
        if total_experiences < 32:
            return
        
        # Sample from both buffers
        regular_sample_size = min(24, len(self.experience_buffer))
        priority_sample_size = min(8, len(self.priority_buffer))
        
        batch = []
        if regular_sample_size > 0:
            batch.extend(np.random.choice(list(self.experience_buffer), size=regular_sample_size, replace=False))
        if priority_sample_size > 0:
            batch.extend(np.random.choice(list(self.priority_buffer), size=priority_sample_size, replace=False))
        
        # Add some previous task data to prevent catastrophic forgetting
        if len(self.previous_task_buffer) > 0:
            prev_task_sample_size = min(8, len(self.previous_task_buffer))
            batch.extend(np.random.choice(list(self.previous_task_buffer), size=prev_task_sample_size, replace=False))
        
        if len(batch) < 16:
            return
        
        # Prepare tensors with explicit dtype conversion
        states = torch.tensor([exp['state_features'] for exp in batch],
                            dtype=torch.float64, device=self.device)
        actions = torch.tensor([exp['action'] for exp in batch],
                             dtype=torch.long, device=self.device)
        rewards = torch.tensor([exp['reward'] for exp in batch],
                             dtype=torch.float64, device=self.device)
        uncertainties = torch.tensor([exp.get('uncertainty', 0.5) for exp in batch],
                                   dtype=torch.float64, device=self.device)
        
        # Enhanced feature learning
        learned_features = self.feature_learner(states)
        
        # Forward pass through main network
        outputs = self.network(learned_features)
        
        # Enhanced loss calculation
        # Policy loss with uncertainty weighting
        action_logits = outputs['action_logits']
        action_probs = F.log_softmax(action_logits, dim=-1)
        selected_probs = action_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Weight by inverse uncertainty (more certain predictions get higher weight)
        uncertainty_weights = 1.0 / (uncertainties + 0.1)
        policy_loss = -(selected_probs * rewards * uncertainty_weights).mean()
        
        # Value losses with uncertainty consideration
        confidence_target = torch.abs(rewards).unsqueeze(1) * uncertainty_weights.unsqueeze(1)
        confidence_loss = F.mse_loss(outputs['confidence'], confidence_target)
        
        # Position size loss (reward-weighted with uncertainty)
        size_target = torch.clamp(torch.abs(rewards) * 2.0 * uncertainty_weights, 0.5, 3.0).unsqueeze(1)
        size_loss = F.mse_loss(outputs['position_size'], size_target)
        
        # Risk parameter loss
        risk_target = torch.sigmoid(rewards.unsqueeze(1).expand(-1, 4))  # Target based on reward
        risk_loss = F.mse_loss(outputs['risk_params'], risk_target)
        
        # Few-shot learning loss
        few_shot_predictions = self.few_shot_learner(learned_features)
        few_shot_loss = F.mse_loss(few_shot_predictions.squeeze(), rewards.unsqueeze(1))
        
        # Regularization for catastrophic forgetting prevention
        regularization_loss = 0.0
        if hasattr(self, 'importance_weights') and self.importance_weights:
            for name, param in self.network.named_parameters():
                if name in self.importance_weights:
                    regularization_loss += torch.sum(self.importance_weights[name] * (param ** 2))
        
        # Total loss with adaptive weighting
        total_loss = (policy_loss +
                     0.1 * confidence_loss +
                     0.05 * size_loss +
                     0.03 * risk_loss +
                     0.02 * few_shot_loss +
                     0.001 * regularization_loss)
        
        # Backward pass with gradient clipping
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # Calculate importance weights for catastrophic forgetting prevention
        self._update_importance_weights()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            list(self.network.parameters()) +
            list(self.feature_learner.parameters()) +
            list(self.few_shot_learner.parameters()),
            1.0
        )
        
        self.optimizer.step()
        
        # Update target network periodically
        if self.total_decisions % 200 == 0:
            self.target_network.load_state_dict(self.network.state_dict())
    
    def _update_importance_weights(self):
        """Update importance weights for catastrophic forgetting prevention"""
        for name, param in self.network.named_parameters():
            if param.grad is not None:
                # Fisher Information approximation
                importance = param.grad.data.clone().pow(2)
                
                if name in self.importance_weights:
                    # Exponential moving average of importance
                    self.importance_weights[name] = (0.9 * self.importance_weights[name] +
                                                   0.1 * importance)
                else:
                    self.importance_weights[name] = importance
    
    def _evolve_architecture(self):
        """Enhanced architecture evolution"""
        new_sizes = self.meta_learner.evolve_architecture()
        
        # Get current performance for evolution decision
        recent_performance = [exp['reward'] for exp in list(self.experience_buffer)[-50:]]
        if len(recent_performance) >= 20:
            avg_performance = np.mean(recent_performance)
            self.network.record_performance(avg_performance)
        
        # Evolve main network
        old_state = self.network.state_dict()
        self.network.evolve_architecture(new_sizes)
        self.target_network.evolve_architecture(new_sizes)
        
        # Update optimizer with new parameters
        self.optimizer = optim.AdamW(
            list(self.network.parameters()) +
            list(self.feature_learner.parameters()) +
            list(self.few_shot_learner.parameters()) +
            list(self.meta_learner.subsystem_weights.parameters()) +
            list(self.meta_learner.exploration_strategy.parameters()),
            lr=self.optimizer.param_groups[0]['lr'],  # Preserve current learning rate
            weight_decay=1e-5
        )
        
        # Update scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.8, patience=50
        )
        
        logger.info(f"Architecture evolved to: {new_sizes}")
    
    def save_model(self, filepath: str):
        """Enhanced model saving"""
        import os
        # Ensure directory exists with proper error handling
        dir_path = os.path.dirname(filepath)
        if dir_path and not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
        
        torch.save({
            'network_state': self.network.state_dict(),
            'target_network_state': self.target_network.state_dict(),
            'feature_learner_state': self.feature_learner.state_dict(),
            'few_shot_learner_state': self.few_shot_learner.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict(),
            'importance_weights': self.importance_weights,
            'total_decisions': self.total_decisions,
            'successful_trades': self.successful_trades,
            'total_pnl': self.total_pnl,
            'last_trade_time': self.last_trade_time,
            'regime_transitions': self.regime_transitions,
            'adaptation_events': self.adaptation_events,
            'strategy_performance': {k: list(v) for k, v in self.strategy_performance.items()},
            'network_evolution_stats': self.network.get_evolution_stats()
        }, filepath)
        
        # Save meta-learner and adaptation engine separately
        meta_filepath = filepath.replace('.pt', '_meta.pt')
        adaptation_filepath = filepath.replace('.pt', '_adaptation.pt')
        
        # Ensure directories exist for additional files with proper error handling
        for path in [meta_filepath, adaptation_filepath]:
            dir_path = os.path.dirname(path)
            if dir_path and not os.path.exists(dir_path):
                os.makedirs(dir_path, exist_ok=True)
        
        self.meta_learner.save_state(meta_filepath)
        
        # Save adaptation engine state
        adaptation_stats = self.adaptation_engine.get_comprehensive_stats()
        torch.save(adaptation_stats, adaptation_filepath)
    
    def load_model(self, filepath: str):
        """Enhanced model loading"""
        try:
            checkpoint = torch.load(filepath, map_location=self.device)
            
            self.network.load_state_dict(checkpoint['network_state'])
            self.target_network.load_state_dict(checkpoint['target_network_state'])
            self.feature_learner.load_state_dict(checkpoint['feature_learner_state'])
            
            if 'few_shot_learner_state' in checkpoint:
                self.few_shot_learner.load_state_dict(checkpoint['few_shot_learner_state'])
            
            self.optimizer.load_state_dict(checkpoint['optimizer_state'])
            
            if 'scheduler_state' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler_state'])
            
            if 'importance_weights' in checkpoint:
                self.importance_weights = checkpoint['importance_weights']
            
            self.total_decisions = checkpoint.get('total_decisions', 0)
            self.successful_trades = checkpoint.get('successful_trades', 0)
            self.total_pnl = checkpoint.get('total_pnl', 0.0)
            self.last_trade_time = checkpoint.get('last_trade_time', 0.0)
            self.regime_transitions = checkpoint.get('regime_transitions', 0)
            self.adaptation_events = checkpoint.get('adaptation_events', 0)
            
            # Load strategy performance
            if 'strategy_performance' in checkpoint:
                for strategy, performance in checkpoint['strategy_performance'].items():
                    self.strategy_performance[strategy] = deque(performance, maxlen=100)
            
            # Load meta-learner
            self.meta_learner.load_state(filepath.replace('.pt', '_meta.pt'))
            
            logger.info("Enhanced model loaded successfully")
            
        except FileNotFoundError:
            logger.info("No existing model found, starting fresh")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
    
    def get_stats(self) -> dict:
        """Enhanced statistics"""
        base_stats = {
            'total_decisions': self.total_decisions,
            'successful_trades': self.successful_trades,
            'success_rate': self.successful_trades / max(1, self.total_decisions),
            'total_pnl': self.total_pnl,
            'experience_size': len(self.experience_buffer),
            'priority_experience_size': len(self.priority_buffer),
            'learning_efficiency': self.meta_learner.get_learning_efficiency(),
            'architecture_generation': self.meta_learner.architecture_evolver.generations,
            'current_sizes': self.meta_learner.architecture_evolver.current_sizes,
            'subsystem_weights': self.meta_learner.get_subsystem_weights().detach().cpu().numpy().tolist(),
            'regime_transitions': self.regime_transitions,
            'adaptation_events': self.adaptation_events
        }
        
        # Strategy performance stats
        strategy_stats = {}
        for strategy, performance in self.strategy_performance.items():
            if performance:
                strategy_stats[strategy] = {
                    'avg_pnl': np.mean(list(performance)),
                    'win_rate': sum(1 for p in performance if p > 0) / len(performance),
                    'total_trades': len(performance),
                    'sharpe_ratio': np.mean(list(performance)) / (np.std(list(performance)) + 1e-8)
                }
        
        # Network evolution stats
        evolution_stats = self.network.get_evolution_stats()
        
        # Adaptation engine stats
        adaptation_stats = self.adaptation_engine.get_comprehensive_stats()
        
        # Key parameters
        key_parameters = {
            'confidence_threshold': self.meta_learner.get_parameter('confidence_threshold'),
            'position_size_factor': self.meta_learner.get_parameter('position_size_factor'),
            'loss_tolerance_factor': self.meta_learner.get_parameter('loss_tolerance_factor'),
            'stop_preference': self.meta_learner.get_parameter('stop_preference'),
            'target_preference': self.meta_learner.get_parameter('target_preference'),
            'current_learning_rate': self.optimizer.param_groups[0]['lr']
        }
        
        return {
            **base_stats,
            'strategy_performance': strategy_stats,
            'network_evolution': evolution_stats,
            'adaptation_engine': adaptation_stats,
            'key_parameters': key_parameters,
            'few_shot_support_size': len(self.few_shot_learner.support_features),
            'catastrophic_forgetting_protection': len(self.importance_weights)
        }