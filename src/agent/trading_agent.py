# trading_agent.py

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import time
import logging

from collections import deque
from queue import Queue
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

from src.shared.types import Features
from src.market_analysis.data_processor import MarketData
from src.agent.meta_learner import MetaLearner
from src.neural.adaptive_network import AdaptiveTradingNetwork, FeatureLearner, StateEncoder
from src.neural.enhanced_neural import (
    SelfEvolvingNetwork, FewShotLearner, ActorCriticLoss, 
    TradingOptimizer, create_enhanced_network
)
from src.agent.real_time_adaptation import RealTimeAdaptationEngine
from src.agent.confidence import ConfidenceManager

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
        
        # Enhanced neural architecture with self-evolution and memory management
        initial_sizes = self.meta_learner.architecture_evolver.current_sizes
        self.network = create_enhanced_network(
            input_size=64,
            initial_sizes=initial_sizes,
            enable_few_shot=True,
            memory_efficient=True,
            max_memory_gb=2.0
        ).to(self.device)

        self.target_network = create_enhanced_network(
            input_size=64,
            initial_sizes=initial_sizes,
            enable_few_shot=False,
            memory_efficient=True,
            max_memory_gb=1.0
        ).to(self.device)
        
        # Enhanced feature learning with catastrophic forgetting prevention
        self.feature_learner = FeatureLearner(
            raw_feature_dim=64, 
            learned_feature_dim=64,
        ).to(self.device)
        self.state_encoder = StateEncoder()
        
        # Few-shot learning capability
        self.few_shot_learner = FewShotLearner(feature_dim=64).to(self.device)
        
        # Real-time adaptation integration
        self.adaptation_engine = RealTimeAdaptationEngine(model_dim=64)
        
        # Enhanced loss function and optimizer system
        self.loss_function = ActorCriticLoss(
            confidence_weight=0.3,
            position_weight=0.2,
            risk_weight=0.1,
            entropy_weight=0.01
        ).to(self.device)
        
        # Advanced multi-component optimizer
        self.optimizer = TradingOptimizer(
            networks=[self.network, self.feature_learner, self.few_shot_learner],
            base_lr=0.001
        )
        
        # Legacy optimizer for compatibility during transition
        self.unified_optimizer = optim.AdamW(
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
            self.unified_optimizer, mode='max', factor=0.8, patience=50
        )
        
        # Thread-safe experience replay with prioritization  
        self.experience_buffer = Queue(maxsize=20000)  # Thread-safe buffer
        self.priority_buffer = Queue(maxsize=5000)  # Thread-safe priority buffer
        
        # Catastrophic forgetting prevention (thread-safe)
        self.previous_task_buffer = Queue(maxsize=1000)
        
        # Queue helper buffers for deque-like operations
        self._experience_list = []
        self._priority_list = []
        self._previous_task_list = []
        self.importance_weights = {}
        
        # Enhanced statistics and tracking
        self.total_decisions = 0
        self.successful_trades = 0
        self.total_pnl = 0.0
        self.last_trade_time = 0.0
        self.regime_transitions = 0
        self.adaptation_events = 0
        self.current_strategy = 'conservative'  # Default strategy
        self.exploration_rate = 0.1  # Default exploration rate
        self.successful_adaptations = 0  # Track successful adaptations
        
        # Performance tracking for different strategies
        self.strategy_performance = {
            'conservative': deque(maxlen=100),
            'aggressive': deque(maxlen=100),
            'momentum': deque(maxlen=100),
            'mean_reversion': deque(maxlen=100),
            'adaptive': deque(maxlen=100)
        }
        
        # NEW: Centralized confidence management system
        self.confidence_manager = ConfidenceManager(
            initial_confidence=0.6,
            min_confidence=0.15,
            max_confidence=0.95,
            debug_mode=True
        )
        
        # DEPRECATED: Legacy confidence tracking (to be removed)
        # Keep temporarily for compatibility during transition
        self.recent_position_rejections = 0
        self.position_rejection_timestamps = deque(maxlen=50)
        self.last_position_rejection_time = 0.0
        self.last_processed_violation_time = 0.0
        self.confidence_recovery_factor = 1.0
        self.last_successful_trade_time = 0.0
        self.confidence_violations = deque(maxlen=20)
        self.recovery_boost_active = False
        
        # Model ensemble for uncertainty quantification
        self.ensemble_predictions = deque(maxlen=10)
    
    # Queue helper methods for deque-like operations
    def _buffer_size(self, buffer_type):
        """Get buffer size"""
        if buffer_type == 'experience':
            return len(self._experience_list)
        elif buffer_type == 'priority':
            return len(self._priority_list)
        elif buffer_type == 'previous_task':
            return len(self._previous_task_list)
        return 0
    
    def _buffer_append(self, item, buffer_type):
        """Add item to buffer with maxsize handling"""
        if buffer_type == 'experience':
            self._experience_list.append(item)
            if len(self._experience_list) > 20000:
                self._experience_list.pop(0)
            try:
                self.experience_buffer.put_nowait(item)
            except:
                pass  # Queue full, list maintains the data
        elif buffer_type == 'priority':
            self._priority_list.append(item)
            if len(self._priority_list) > 5000:
                self._priority_list.pop(0)
            try:
                self.priority_buffer.put_nowait(item)
            except:
                pass  # Queue full, list maintains the data
        elif buffer_type == 'previous_task':
            self._previous_task_list.append(item)
            if len(self._previous_task_list) > 1000:
                self._previous_task_list.pop(0)
            try:
                self.previous_task_buffer.put_nowait(item)
            except:
                pass  # Queue full, list maintains the data
    
    def _buffer_recent(self, n, buffer_type):
        """Get recent N items from buffer"""
        if buffer_type == 'experience':
            return self._experience_list[-n:] if self._experience_list else []
        elif buffer_type == 'priority':
            return self._priority_list[-n:] if self._priority_list else []
        elif buffer_type == 'previous_task':
            return self._previous_task_list[-n:] if self._previous_task_list else []
        return []
    
    def _buffer_all(self, buffer_type):
        """Get all items from buffer as list"""
        if buffer_type == 'experience':
            return self._experience_list.copy()
        elif buffer_type == 'priority':
            return self._priority_list.copy()
        elif buffer_type == 'previous_task':
            return self._previous_task_list.copy()
        return []

    def decide(self, features: Features, market_data: MarketData) -> Decision:
        self.total_decisions += 1
        
        # DOPAMINE PHASE: PRE-TRADE ANTICIPATION
        # Trigger anticipation response before making decision
        anticipation_context = {
            'confidence': getattr(features, 'confidence', 0.5),
            'expected_outcome': self._estimate_expected_outcome(features, market_data)
        }
        dopamine_anticipation = self.intelligence.dopamine_subsystem.process_trading_event(
            'anticipation', self._create_dopamine_market_data(market_data), anticipation_context
        )
        
        # Check for architecture evolution
        if self.meta_learner.should_evolve_architecture():
            self._evolve_architecture()
        
        # Enhanced state representation with microstructure features and dopamine state
        meta_context = self._get_enhanced_meta_context(market_data, features)
        meta_context['dopamine_state'] = dopamine_anticipation.state.value
        meta_context['dopamine_urgency'] = dopamine_anticipation.urgency_factor
        
        raw_state = self._create_enhanced_state(market_data, features, meta_context)
        
        # Learn features with catastrophic forgetting prevention
        learned_state = self.feature_learner(raw_state.unsqueeze(0).to(self.device))
        
        # Few-shot learning prediction
        few_shot_prediction = self.few_shot_learner(learned_state)
        
        # Enhanced subsystem contributions with explicit dtype and dopamine influence
        base_dopamine_signal = float(features.dopamine_signal)
        dopamine_amplified_signal = base_dopamine_signal * dopamine_anticipation.position_size_modifier
        
        subsystem_signals = torch.tensor([
            float(features.dna_signal),
            float(features.temporal_signal),
            float(features.immune_signal),
            float(features.microstructure_signal),  # Enhanced with microstructure
            dopamine_amplified_signal,  # Dopamine-amplified signal based on psychological state
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
        
        # CRITICAL FIX: Always process neural network and confidence recovery BEFORE any early returns
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
            raw_confidence = float(combined_outputs['confidence'].detach().cpu().numpy()[0])
            
            # COMPREHENSIVE DEBUG: Log all confidence stages
            logger.info(f"CONFIDENCE TRACE: raw_from_network={raw_confidence:.6f}")
            
            # DEBUG: Log raw confidence from neural network
            if raw_confidence < 0.1:
                logger.error(f"CRITICAL: Neural network outputting raw_confidence={raw_confidence:.6f} - BoundedSigmoid may be failing!")
                logger.error(f"Network confidence tensor: {combined_outputs['confidence']}")
                logger.error(f"Network output keys: {list(combined_outputs.keys())}")
            
            # NEW: Use centralized confidence manager
            market_context = {
                'volatility': features.volatility,
                'price_momentum': features.price_momentum,
                'regime_confidence': features.regime_confidence
            }
            confidence = self.confidence_manager.process_neural_output(raw_confidence, market_context)
            
            # CONFIDENCE DEBUG: Log final confidence after processing
            logger.info(f"CONFIDENCE TRACE: after_recovery={confidence:.6f}")
            if confidence != raw_confidence:
                logger.info(f"Confidence processed: {raw_confidence:.3f} -> {confidence:.3f}")
            if confidence < 0.15:
                logger.error(f"CRITICAL: Final confidence={confidence:.6f} after processing - check confidence manager!")
        
        # Enhanced trading constraints with regime awareness - NOW uses recovered confidence
        if not self._should_consider_trading(market_data, meta_context, features):
            # CRITICAL FIX: Use recovered confidence instead of hardcoded 0
            logger.info(f"CONFIDENCE TRACE: early_hold_decision={confidence:.6f}")
            return Decision('hold', confidence, 0, regime_awareness=adaptation_context)
        
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
        
        # Enhanced sizing and risk parameters with dopamine influence
        raw_position_size = float(combined_outputs['position_size'].detach().cpu().numpy()[0])
        risk_params = combined_outputs['risk_params'].detach().cpu().numpy()[0]
        
        # Ensure position size is always positive
        base_position_size = max(0.1, abs(raw_position_size))
        
        # DOPAMINE-DRIVEN POSITION SIZING
        # Apply dopamine psychological modifiers to position sizing
        dopamine_position_modifier = dopamine_anticipation.position_size_modifier
        dopamine_risk_modifier = dopamine_anticipation.risk_tolerance_modifier
        
        # Adjust position size based on dopamine state
        position_size = base_position_size * dopamine_position_modifier
        
        # Uncertainty-adjusted position sizing
        uncertainty_factor = 1.0 - adaptation_decision.get('uncertainty', 0.5)
        position_size *= max(0.1, uncertainty_factor)  # Ensure factor is positive
        
        # Apply dopamine urgency factor to position sizing
        urgency_adjustment = 0.8 + (dopamine_anticipation.urgency_factor * 0.4)  # 0.8 to 1.2 range
        position_size *= urgency_adjustment
        
        # Regime-aware risk management
        use_stop = risk_params[0] > 0.5
        stop_distance = risk_params[1]
        use_target = risk_params[2] > 0.5
        target_distance = risk_params[3]
        
        # Adjust risk parameters based on regime and dopamine state
        if features.regime_confidence < 0.5:  # Uncertain regime
            stop_distance *= 0.8  # Tighter stops
        
        # Dopamine-influenced risk parameters
        stop_distance *= dopamine_risk_modifier
        target_distance *= dopamine_risk_modifier
        
        # Convert to decision
        actions = ['hold', 'buy', 'sell']
        action = actions[action_idx]
        
        if action == 'hold':
            # COMPREHENSIVE DEBUG: Log confidence before Decision creation
            logger.info(f"CONFIDENCE TRACE: creating_hold_decision={confidence:.6f}")
            decision = Decision(
                'hold', confidence, 0,
                adaptation_strategy=selected_strategy,
                uncertainty_estimate=adaptation_decision.get('uncertainty', 0.5),
                few_shot_prediction=float(few_shot_prediction.squeeze()),
                regime_awareness=adaptation_context
            )
            # COMPREHENSIVE DEBUG: Log confidence after Decision creation
            logger.info(f"CONFIDENCE TRACE: hold_decision_created={decision.confidence:.6f}")
            return decision
        
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
        
        # DOPAMINE PHASE: EXECUTION 
        # Process execution phase for dopamine response
        execution_context = {
            'action': action,
            'position_size': position_size,
            'confidence': confidence
        }
        dopamine_execution = self.intelligence.dopamine_subsystem.process_trading_event(
            'execution', self._create_dopamine_market_data(market_data), execution_context
        )
        
        # Store few-shot learning example
        self.few_shot_learner.add_support_example(learned_state.squeeze(), weighted_signal.detach().item())
        
        decision = Decision(
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
        
        # Store dopamine context with decision for later reflection
        decision.dopamine_anticipation = dopamine_anticipation
        decision.dopamine_execution = dopamine_execution
        
        return decision
    
    def _strategic_exploration(self, weighted_signal: torch.Tensor, strategy: str, 
                             features: Features, adaptation_decision: Dict) -> int:
        """Enhanced exploration based on selected strategy"""
        
        if strategy == 'conservative':
            # Conservative exploration - adaptive signal threshold
            if abs(weighted_signal) > 0.01:  # Much lower threshold for discovery
                return 1 if weighted_signal > 0 else 2
            return 0
        
        elif strategy == 'aggressive':
            # Aggressive exploration - very low threshold
            if abs(weighted_signal) > 0.005:  # Very low threshold for discovery
                return 1 if weighted_signal > 0 else 2
            return 0
        
        elif strategy == 'momentum':
            # Momentum strategy - adaptive momentum threshold
            if features.price_momentum > 0.001:  # Much lower threshold
                return 1
            elif features.price_momentum < -0.001:  # Much lower threshold
                return 2
            return 0
        
        elif strategy == 'mean_reversion':
            # Mean reversion - adaptive position thresholds
            if features.price_position > 0.6:  # Lower threshold for discovery
                return 2  # Sell at highs
            elif features.price_position < 0.4:  # Higher threshold for discovery
                return 1  # Buy at lows
            return 0
        
        else:  # adaptive
            # Adaptive strategy based on current conditions
            if adaptation_decision.get('emergency_mode', False):
                return 0  # Hold in emergency
            elif features.regime_confidence > 0.7:
                # High confidence regime - follow signals with low threshold
                return 1 if weighted_signal > 0.005 else (2 if weighted_signal < -0.005 else 0)
            else:
                # Low confidence regime - slightly higher threshold
                return 1 if weighted_signal > 0.01 else (2 if weighted_signal < -0.01 else 0)
    
    def _calculate_enhanced_levels(self, action: str, market_data: MarketData,
                                 use_stop: bool, stop_distance: float,
                                 use_target: bool, target_distance: float,
                                 features: Features) -> tuple:
        """Intelligent stop and target calculation based on learning"""
        
        # Let the algorithm learn whether to use stops/targets at all
        learned_stop_preference = self.meta_learner.get_parameter('stop_preference')
        learned_target_preference = self.meta_learner.get_parameter('target_preference')
        
        # Intelligent decision on whether to use stops/targets
        # Based on recent performance with/without them
        stop_effectiveness = self._evaluate_stop_effectiveness()
        target_effectiveness = self._evaluate_target_effectiveness()
        
        # Dynamic decision making
        should_use_stop = (use_stop and learned_stop_preference > 0.3 and stop_effectiveness > 0.0)
        should_use_target = (use_target and learned_target_preference > 0.3 and target_effectiveness > 0.0)
        
        stop_price = 0.0
        target_price = 0.0
        
        if should_use_stop:
            # Intelligent stop placement based on market conditions
            stop_price = self._calculate_intelligent_stop(action, market_data, features, stop_distance)
        
        if should_use_target:
            # Intelligent target placement based on market conditions
            target_price = self._calculate_intelligent_target(action, market_data, features, target_distance)
        
        return stop_price, target_price
    
    def _evaluate_stop_effectiveness(self) -> float:
        """Evaluate how effective stops have been recently"""
        if self._buffer_size('experience') < 20:
            return 0.5  # Neutral when insufficient data
        
        recent_experiences = self._buffer_recent(50, 'experience')
        
        stop_used_outcomes = []
        no_stop_outcomes = []
        
        for exp in recent_experiences:
            trade_data = exp.get('trade_data', {})
            if isinstance(trade_data, dict):
                stop_used = trade_data.get('stop_used', False)
                reward = exp.get('reward', 0)
                
                if stop_used:
                    stop_used_outcomes.append(reward)
                else:
                    no_stop_outcomes.append(reward)
        
        if not stop_used_outcomes or not no_stop_outcomes:
            return 0.5
        
        avg_with_stop = np.mean(stop_used_outcomes)
        avg_without_stop = np.mean(no_stop_outcomes)
        
        # Return relative effectiveness (-1 to 1)
        return np.tanh((avg_with_stop - avg_without_stop) * 5)
    
    def _evaluate_target_effectiveness(self) -> float:
        """Evaluate how effective targets have been recently"""
        if self._buffer_size('experience') < 20:
            return 0.5  # Neutral when insufficient data
        
        recent_experiences = self._buffer_recent(50, 'experience')
        
        target_used_outcomes = []
        no_target_outcomes = []
        
        for exp in recent_experiences:
            trade_data = exp.get('trade_data', {})
            if isinstance(trade_data, dict):
                target_used = trade_data.get('target_used', False)
                reward = exp.get('reward', 0)
                
                if target_used:
                    target_used_outcomes.append(reward)
                else:
                    no_target_outcomes.append(reward)
        
        if not target_used_outcomes or not no_target_outcomes:
            return 0.5
        
        avg_with_target = np.mean(target_used_outcomes)
        avg_without_target = np.mean(no_target_outcomes)
        
        # Return relative effectiveness (-1 to 1)
        return np.tanh((avg_with_target - avg_without_target) * 5)
    
    def _calculate_intelligent_stop(self, action: str, market_data: MarketData,
                                  features: Features, base_distance: float) -> float:
        """Calculate intelligent stop placement"""
        
        # Base distance from meta-learner
        base_stop_distance = self.meta_learner.get_parameter('stop_distance_factor')
        
        # Volatility-based adjustment
        vol_adjustment = 1.0 + (features.volatility * 10)  # Scale with volatility
        
        # Regime-based adjustment
        regime_adjustment = 1.0
        if features.regime_confidence < 0.4:
            regime_adjustment = 0.7  # Tighter stops in uncertain regimes
        elif features.volatility > 0.04:
            regime_adjustment = 1.5  # Wider stops in high volatility
        
        # Time-based adjustment (wider stops during volatile hours)
        time_adjustment = 1.0
        if 0.35 < features.time_of_day < 0.65:  # Market open hours
            time_adjustment = 1.2
        
        # Combined adjustment
        final_distance = base_stop_distance * (1 + base_distance) * vol_adjustment * regime_adjustment * time_adjustment
        final_distance = min(0.05, max(0.005, final_distance))  # Reasonable bounds
        
        if action == 'buy':
            return market_data.price * (1 - final_distance)
        else:
            return market_data.price * (1 + final_distance)
    
    def _calculate_intelligent_target(self, action: str, market_data: MarketData,
                                    features: Features, base_distance: float) -> float:
        """Calculate intelligent target placement"""
        
        # Base distance from meta-learner
        base_target_distance = self.meta_learner.get_parameter('target_distance_factor')
        
        # Trend strength adjustment
        trend_adjustment = 1.0 + abs(features.price_momentum) * 5  # Wider targets in strong trends
        
        # Confidence adjustment
        confidence_adjustment = 0.5 + features.confidence  # Wider targets with higher confidence
        
        # Pattern strength adjustment
        pattern_adjustment = 1.0 + features.pattern_score * 0.5
        
        # Combined adjustment
        final_distance = base_target_distance * (1 + base_distance) * trend_adjustment * confidence_adjustment * pattern_adjustment
        final_distance = min(0.15, max(0.01, final_distance))  # Reasonable bounds
        
        if action == 'buy':
            return market_data.price * (1 + final_distance)
        else:
            return market_data.price * (1 - final_distance)
    
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
            
            # CRITICAL: Position limit awareness - current position context
            float(getattr(market_data, 'total_position_size', 0) / 10.0),  # Normalized position
            float(abs(getattr(market_data, 'total_position_size', 0)) / int(self.meta_learner.get_parameter('max_contracts_limit'))),  # Position ratio
            float(min(1.0, self.recent_position_rejections / 10.0)),  # Recent rejection ratio
            
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
        if time_since_last < (1 / frequency_limit):  # Adaptive timing, no hardcoded minimum
            return False
        
        # Let AI discover regime-based constraints through economic feedback
        # Removed hardcoded regime blocks to enable neuromorphic boundary learning
        
        # Adaptation engine emergency mode
        if meta_context.get('adaptation_events', 0) > 0.1:  # Too many adaptation events
            return False
        
        # Let AI learn position limit boundaries through economic violation feedback
        # Removed hardcoded position blocking to enable neuromorphic boundary discovery
        
        return True
    
    def _get_enhanced_primary_tool(self, signals: torch.Tensor, weights: torch.Tensor, 
                                 features: Features) -> str:
        """Enhanced primary tool identification"""
        weighted_signals = torch.abs(signals * weights[:len(signals)])
        tool_names = ['dna', 'temporal', 'immune', 'microstructure', 'dopamine', 'regime']
        
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
    
    # REMOVED: _apply_confidence_recovery method
    # This functionality has been moved to the centralized ConfidenceManager
    
    def learn_from_trade(self, trade):
        """Enhanced learning with comprehensive error handling and debugging"""
        try:
            logger.info(f"=== LEARNING FROM TRADE START ===")
            logger.info(f"Trade PnL: {trade.pnl}, Entry: {trade.entry_price}, Exit: {trade.exit_price}")
            
            # DOPAMINE PHASE: REALIZATION
            # Process trade exit and P&L realization for dopamine response
            realization_market_data = {
                'unrealized_pnl': 0.0,  # Position closed
                'daily_pnl': getattr(trade, 'pnl', 0.0),
                'realized_pnl': getattr(trade, 'pnl', 0.0),
                'open_positions': 0.0,  # Position closed
                'current_price': getattr(trade, 'exit_price', 0.0),
                'trade_duration': getattr(trade, 'duration', 0.0)
            }
            
            realization_context = {
                'expected_outcome': getattr(trade, 'expected_outcome', 0.0),
                'confidence': getattr(trade, 'confidence', 0.5),
                'action': getattr(trade, 'action', 'unknown')
            }
            
            dopamine_realization = self.intelligence.dopamine_subsystem.process_trading_event(
                'realization', realization_market_data, realization_context
            )
            
            if not hasattr(trade, 'intelligence_data'):
                logger.warning("Trade missing intelligence_data - creating minimal data")
                trade.intelligence_data = {'subsystem_signals': [0,0,0,0,0,0]}
            
            # NEW: Update confidence manager with trade outcome and dopamine integration
            trade_context = {
                'entry_price': trade.entry_price,
                'exit_price': trade.exit_price,
                'strategy': getattr(trade, 'adaptation_strategy', 'conservative'),
                'tool_used': getattr(trade, 'primary_tool', 'unknown')
            }
            self.confidence_manager.handle_trade_outcome(trade.pnl, trade_context, dopamine_realization)
            
            # LEGACY: Update statistics (for compatibility)
            if trade.pnl > 0:
                self.successful_trades += 1
                self.last_successful_trade_time = time.time()
                # Boost confidence recovery factor for successful trades
                self.confidence_recovery_factor = min(1.2, self.confidence_recovery_factor + 0.05)
                success_rate = self.successful_trades / max(1, self.total_decisions)
                logger.info(f"SUCCESSFUL TRADE: #{self.successful_trades}, Success rate: {success_rate:.1%}")
            elif trade.pnl < 0:
                # Slight reduction in recovery factor for losses (but not major impact)
                self.confidence_recovery_factor = max(0.8, self.confidence_recovery_factor - 0.01)
                logger.info(f"LOSING TRADE: PnL={trade.pnl:.2f}")
            self.total_pnl += trade.pnl
            
            # Track strategy performance
            strategy = getattr(trade, 'adaptation_strategy', 'conservative')
            if strategy in self.strategy_performance:
                self.strategy_performance[strategy].append(trade.pnl)
                logger.info(f"Strategy {strategy} performance updated")
            
            # Enhanced trade data for meta-learning
            try:
                account_balance = getattr(trade.market_data, 'account_balance', 1000) if hasattr(trade, 'market_data') else 1000
                
                trade_data = {
                    'pnl': float(trade.pnl),
                    'account_balance': float(account_balance),
                    'hold_time': float(trade.exit_time - trade.entry_time),
                    'was_exploration': bool(getattr(trade, 'exploration', False)),
                    'subsystem_contributions': torch.tensor([0,0,0,0,0,0], dtype=torch.float64),  # Safe default
                    'subsystem_agreement': 0.5,  # Safe default
                    'confidence': float(getattr(trade, 'confidence', 0.5)),
                    'primary_tool': str(getattr(trade, 'primary_tool', 'unknown')),
                    'stop_used': bool(getattr(trade, 'stop_used', False)),
                    'target_used': bool(getattr(trade, 'target_used', False)),
                    'adaptation_strategy': str(getattr(trade, 'adaptation_strategy', 'conservative')),
                    'uncertainty_estimate': float(getattr(trade, 'uncertainty_estimate', 0.5)),
                    'regime_confidence': 0.5  # Safe default
                }
                
                # Safely extract subsystem contributions
                if hasattr(trade, 'intelligence_data') and trade.intelligence_data:
                    signals = trade.intelligence_data.get('subsystem_signals', [0,0,0,0,0,0])
                    if isinstance(signals, (list, tuple)) and len(signals) >= 3:
                        trade_data['subsystem_contributions'] = torch.tensor(signals[:6] + [0]*(6-len(signals)), dtype=torch.float64)
                        trade_data['subsystem_agreement'] = self._calculate_enhanced_subsystem_agreement(trade.intelligence_data)
                        trade_data['regime_confidence'] = trade.intelligence_data.get('regime_context', {}).get('regime_confidence', 0.5)
                
                logger.info(f"Trade data prepared: PnL={trade_data['pnl']}, Confidence={trade_data['confidence']}")
                
            except Exception as e:
                logger.error(f"Error preparing trade data: {e}")
                # Use minimal safe trade data
                trade_data = {
                    'pnl': float(trade.pnl),
                    'account_balance': 1000.0,
                    'hold_time': 60.0,
                    'was_exploration': False,
                    'subsystem_contributions': torch.tensor([0,0,0,0,0,0], dtype=torch.float64),
                    'subsystem_agreement': 0.5,
                    'confidence': 0.5,
                    'primary_tool': 'unknown',
                    'stop_used': False,
                    'target_used': False,
                    'adaptation_strategy': 'conservative',
                    'uncertainty_estimate': 0.5,
                    'regime_confidence': 0.5
                }
            
            # Meta-learning update with error handling
            try:
                logger.info("Starting meta-learner update...")
                old_total_updates = self.meta_learner.total_updates
                old_successful_adaptations = self.meta_learner.successful_adaptations
                
                # Compute enhanced reward
                reward = self.meta_learner.compute_reward(trade_data)
                logger.info(f"Computed reward: {reward}")
                
                # Meta-learning update
                self.meta_learner.learn_from_outcome(trade_data)
                
                new_total_updates = self.meta_learner.total_updates
                new_successful_adaptations = self.meta_learner.successful_adaptations
                
                logger.info(f"Meta-learner updates: {old_total_updates} -> {new_total_updates}")
                logger.info(f"Successful adaptations: {old_successful_adaptations} -> {new_successful_adaptations}")
                
                if new_total_updates > old_total_updates:
                    logger.info("Meta-learner successfully updated!")
                else:
                    logger.warning("Meta-learner update failed - no increment in total_updates")
                    
            except Exception as e:
                logger.error(f"Meta-learner update failed: {e}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
            
            # Real-time adaptation update with error handling
            try:
                logger.info("Starting adaptation engine update...")
                adaptation_context = trade.intelligence_data.get('regime_context', {}) if hasattr(trade, 'intelligence_data') and trade.intelligence_data else {}
                adaptation_context['predicted_confidence'] = trade_data['confidence']
                self.adaptation_engine.update_from_outcome(reward, adaptation_context)
                logger.info("Adaptation engine updated!")
            except Exception as e:
                logger.error(f"Adaptation engine update failed: {e}")
            
            # Network performance tracking
            try:
                self.network.record_performance(reward)
                logger.info(f"Network performance recorded: {reward}")
            except Exception as e:
                logger.error(f"Network performance recording failed: {e}")
            
            # Store experience with priority
            try:
                if hasattr(trade, 'state_features') and trade.state_features:
                    experience = {
                        'state_features': trade.state_features,
                        'action': ['hold', 'buy', 'sell'].index(trade.action) if trade.action in ['hold', 'buy', 'sell'] else 0,
                        'reward': reward,
                        'done': True,
                        'trade_data': {k: (v.detach().cpu().numpy().tolist() if hasattr(v, 'detach') else v) 
                                    for k, v in trade_data.items()},  # Convert tensors for storage
                        'uncertainty': trade_data['uncertainty_estimate'],
                        'regime_confidence': trade_data['regime_confidence']
                    }
                    
                    # Prioritize unusual or high-impact experiences
                    if abs(reward) > 0.5 or trade_data['uncertainty_estimate'] > 0.7:
                        self._buffer_append(experience, 'priority')
                        logger.info("Experience added to priority buffer")
                    else:
                        self._buffer_append(experience, 'experience')
                        logger.info("Experience added to regular buffer")
                        
                else:
                    logger.warning("Trade missing state_features - skipping experience storage")
                    
            except Exception as e:
                logger.error(f"Experience storage failed: {e}")
            
            # Train networks if enough experience
            try:
                if self._buffer_size('experience') >= 64 or self._buffer_size('priority') >= 32:
                    logger.info("Starting network training...")
                    self._train_enhanced_networks()
                    logger.info("Network training completed!")
            except Exception as e:
                logger.error(f"Network training failed: {e}")
            
            # Update learning rate based on performance
            try:
                recent_rewards = [exp['reward'] for exp in self._buffer_recent(20, 'experience')]
                if len(recent_rewards) >= 10:
                    avg_reward = np.mean(recent_rewards)
                    self.scheduler.step(avg_reward)
                    logger.info(f"Learning rate updated based on avg reward: {avg_reward}")
            except Exception as e:
                logger.error(f"Learning rate update failed: {e}")
            
            # Periodic parameter adaptation
            try:
                if self.total_decisions % 50 == 0:
                    logger.info("Running periodic parameter adaptation...")
                    self.meta_learner.adapt_parameters()
                    logger.info("Parameter adaptation completed!")
            except Exception as e:
                logger.error(f"Parameter adaptation failed: {e}")
            
            # Log final learning stats
            try:
                current_efficiency = self.meta_learner.get_learning_efficiency()
                logger.info(f"Current learning efficiency: {current_efficiency:.2%}")
                logger.info(f"Total meta-learner updates: {self.meta_learner.total_updates}")
                logger.info(f"Successful adaptations: {self.meta_learner.successful_adaptations}")
            except Exception as e:
                logger.error(f"Error getting learning stats: {e}")
            
            # DOPAMINE PHASE: REFLECTION
            # Process post-trade reflection for learning and psychological adjustment
            reflection_context = {
                'learned_something': True,
                'trade_outcome': 'profit' if trade.pnl > 0 else 'loss',
                'strategy_effectiveness': getattr(trade, 'adaptation_strategy', 'unknown'),
                'confidence_outcome': getattr(trade, 'confidence', 0.5)
            }
            
            dopamine_reflection = self.intelligence.dopamine_subsystem.process_trading_event(
                'reflection', realization_market_data, reflection_context
            )
            
            # Log comprehensive dopamine learning
            logger.info(f"DOPAMINE LEARNING: Realization signal={dopamine_realization.signal:.3f}, "
                       f"Reflection signal={dopamine_reflection.signal:.3f}, "
                       f"State={dopamine_reflection.state.value}, "
                       f"Tolerance={dopamine_reflection.tolerance_level:.3f}, "
                       f"Addiction={dopamine_reflection.addiction_risk:.3f}")
            
            logger.info(f"=== LEARNING FROM TRADE COMPLETE ===")
            
        except Exception as e:
            logger.error(f"CRITICAL: Overall learn_from_trade failed: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
    
    def _calculate_enhanced_subsystem_agreement(self, intelligence_data: Dict) -> float:
        """Enhanced subsystem agreement calculation"""
        signals = intelligence_data.get('subsystem_signals', [0, 0, 0, 0, 0, 0])
        
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
        total_experiences = self._buffer_size('experience') + self._buffer_size('priority')
        if total_experiences < 32:
            return
        
        # Sample from both buffers
        regular_sample_size = min(24, self._buffer_size('experience'))
        priority_sample_size = min(8, self._buffer_size('priority'))
        
        batch = []
        if regular_sample_size > 0:
            batch.extend(np.random.choice(self._buffer_all('experience'), size=regular_sample_size, replace=False))
        if priority_sample_size > 0:
            batch.extend(np.random.choice(self._buffer_all('priority'), size=priority_sample_size, replace=False))
        
        # Add some previous task data to prevent catastrophic forgetting
        if self._buffer_size('previous_task') > 0:
            prev_task_sample_size = min(8, self._buffer_size('previous_task'))
            batch.extend(np.random.choice(self._buffer_all('previous_task'), size=prev_task_sample_size, replace=False))
        
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
        self.unified_optimizer.zero_grad()
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
        
        self.unified_optimizer.step()
        
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
        recent_performance = [exp['reward'] for exp in self._buffer_recent(50, 'experience')]
        if len(recent_performance) >= 20:
            avg_performance = np.mean(recent_performance)
            self.network.record_performance(avg_performance)
        
        # Evolve main network
        old_state = self.network.state_dict()
        self.network.evolve_architecture(new_sizes)
        self.target_network.evolve_architecture(new_sizes)
        
        # Update optimizer with new parameters
        self.unified_optimizer = optim.AdamW(
            list(self.network.parameters()) +
            list(self.feature_learner.parameters()) +
            list(self.few_shot_learner.parameters()) +
            list(self.meta_learner.subsystem_weights.parameters()) +
            list(self.meta_learner.exploration_strategy.parameters()),
            lr=self.unified_optimizer.param_groups[0]['lr'],  # Preserve current learning rate
            weight_decay=1e-5
        )
        
        # Update scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.unified_optimizer, mode='max', factor=0.8, patience=50
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
            'optimizer_state': self.unified_optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict(),
            'importance_weights': self.importance_weights,
            'total_decisions': self.total_decisions,
            'successful_trades': self.successful_trades,
            'total_pnl': self.total_pnl,
            'last_trade_time': self.last_trade_time,
            'regime_transitions': self.regime_transitions,
            'adaptation_events': self.adaptation_events,
            'strategy_performance': {k: list(v) for k, v in self.strategy_performance.items()},
            'network_evolution_stats': self.network.get_evolution_stats(),
            'experience_list': self._experience_list[-1000:],  # Save last 1000 experiences
            'priority_list': self._priority_list[-500:],  # Save last 500 priority experiences
            'previous_task_list': self._previous_task_list[-200:]  # Save last 200 previous tasks
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
        """Enhanced model loading with architecture compatibility check"""
        try:
            # Handle PyTorch 2.6+ security requirements
            try:
                checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
            except Exception as weights_error:
                # Fallback for older PyTorch versions or if weights_only fails
                logger.warning(f"weights_only=False failed, trying alternative loading: {weights_error}")
                checkpoint = torch.load(filepath, map_location=self.device)
            
            # Check for architecture compatibility (BoundedSigmoid vs Sigmoid)
            try:
                self.network.load_state_dict(checkpoint['network_state'])
                self.target_network.load_state_dict(checkpoint['target_network_state'])
                self.feature_learner.load_state_dict(checkpoint['feature_learner_state'])
            except Exception as arch_error:
                logger.warning(f"Architecture mismatch detected (old model vs new BoundedSigmoid): {arch_error}")
                logger.warning("Starting with fresh neural networks due to architecture change")
                # Don't load network states - keep fresh networks with BoundedSigmoid
                return
            
            if 'few_shot_learner_state' in checkpoint:
                self.few_shot_learner.load_state_dict(checkpoint['few_shot_learner_state'])
            
            self.unified_optimizer.load_state_dict(checkpoint['optimizer_state'])
            
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
            
            # Load buffer lists
            if 'experience_list' in checkpoint:
                self._experience_list = checkpoint['experience_list']
            if 'priority_list' in checkpoint:
                self._priority_list = checkpoint['priority_list']
            if 'previous_task_list' in checkpoint:
                self._previous_task_list = checkpoint['previous_task_list']
            
            # Load meta-learner
            self.meta_learner.load_state(filepath.replace('.pt', '_meta.pt'))
            
            logger.info("Enhanced model loaded successfully")
            
        except FileNotFoundError:
            logger.info("No existing model found, starting fresh with BoundedSigmoid confidence (min=0.1)")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
    
    def get_stats(self) -> dict:
        """Enhanced statistics"""
        base_stats = {
            'total_decisions': self.total_decisions,
            'successful_trades': self.successful_trades,
            'success_rate': self.successful_trades / max(1, self.total_decisions),
            'total_pnl': self.total_pnl,
            'experience_size': self._buffer_size('experience'),
            'priority_experience_size': self._buffer_size('priority'),
            'learning_efficiency': self.meta_learner.get_learning_efficiency(),
            'architecture_generation': self.meta_learner.architecture_evolver.generations,
            'current_sizes': self.meta_learner.architecture_evolver.current_sizes,
            'subsystem_weights': self.meta_learner.get_subsystem_weights().detach().cpu().numpy().tolist(),
            'regime_transitions': self.regime_transitions,
            'adaptation_events': self.adaptation_events,
            'recent_rewards': list(self.meta_learner.outcome_history)[-5:] if hasattr(self.meta_learner, 'outcome_history') else [],
            'current_strategy': self.current_strategy,
            'exploration_rate': self.exploration_rate,
            'meta_learner_updates': getattr(self.meta_learner, 'total_updates', 0),
            'successful_adaptations': self.successful_adaptations
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
            'current_learning_rate': self.unified_optimizer.param_groups[0]['lr']
        }
        
        return {
            **base_stats,
            'strategy_performance': strategy_stats,
            'network_evolution': evolution_stats,
            'adaptation_engine': adaptation_stats,
            'key_parameters': key_parameters,
            'few_shot_support_size': len(self.few_shot_learner.support_features),
            'catastrophic_forgetting_protection': len(self.importance_weights),
            'buffer_sizes': {
                'experience': self._buffer_size('experience'),
                'priority': self._buffer_size('priority'),
                'previous_task': self._buffer_size('previous_task')
            }
        }
    
    def get_agent_context(self) -> Dict:
        """Get comprehensive agent context for LLM"""
        # Get ensemble agreement from recent predictions
        ensemble_agreement = 0.5
        if len(self.ensemble_predictions) > 1:
            predictions = list(self.ensemble_predictions)
            if len(predictions) >= 2:
                agreement = 1.0 - np.std([p.get('confidence', 0.5) for p in predictions])
                ensemble_agreement = max(0.0, min(1.0, agreement))
        
        return {
            'neural_confidence': ensemble_agreement,
            'learning_phase': getattr(self.meta_learner, 'current_phase', 'exploitation'),
            'adaptation_trigger': getattr(self, 'last_adaptation_reason', 'none'),
            'reward_prediction_error': getattr(self, 'last_reward_error', 0.0),
            'exploration_vs_exploitation': self.exploration_rate,
            'current_strategy': self.current_strategy,
            'recent_performance_trend': self._calculate_performance_trend(),
            'network_evolution_status': self.network.get_evolution_stats(),
            'meta_learning_efficiency': self.meta_learner.get_learning_efficiency(),
            'adaptation_events_rate': self.adaptation_events / max(1, self.total_decisions),
            'regime_transitions_rate': self.regime_transitions / max(1, self.total_decisions)
        }
    
    def get_decision_reasoning(self, decision) -> Dict:
        """Get detailed decision reasoning context"""
        if not hasattr(decision, 'intelligence_data') or not decision.intelligence_data:
            return {}
        
        intel = decision.intelligence_data
        subsystem_signals = intel.get('subsystem_signals', [0,0,0,0,0,0])
        subsystem_weights = intel.get('subsystem_weights', [0,0,0,0,0,0])
        
        # Calculate weighted contributions
        weighted_contributions = {}
        tool_names = ['dna', 'temporal', 'immune', 'microstructure', 'dopamine', 'regime']
        for i, (signal, weight, name) in enumerate(zip(subsystem_signals, subsystem_weights, tool_names)):
            if i < len(subsystem_signals) and i < len(subsystem_weights):
                weighted_contributions[name] = float(signal * weight)
        
        return {
            'subsystem_contributions': weighted_contributions,
            'risk_override_active': decision.confidence < 0.3,
            'regime_alignment': intel.get('regime_context', {}).get('regime_confidence', 0.5),
            'pattern_match_strength': intel.get('ensemble_uncertainty', 0.5),
            'temporal_window_used': intel.get('adaptation_decision', {}).get('strategy_name', 'unknown'),
            'uncertainty_level': decision.uncertainty_estimate,
            'exploration_mode': decision.exploration,
            'primary_signal_strength': max([abs(s) for s in subsystem_signals] + [0])
        }
    
    def get_learning_context(self) -> Dict:
        """Get system learning and adaptation context"""
        recent_experiences = self._buffer_recent(20, 'experience')
        recent_rewards = [exp.get('reward', 0) for exp in recent_experiences]
        
        performance_trend = 'stable'
        if len(recent_rewards) >= 10:
            first_half = np.mean(recent_rewards[:len(recent_rewards)//2])
            second_half = np.mean(recent_rewards[len(recent_rewards)//2:])
            if second_half > first_half + 0.1:
                performance_trend = 'improving'
            elif second_half < first_half - 0.1:
                performance_trend = 'declining'
        
        return {
            'recent_performance_trend': performance_trend,
            'strategy_effectiveness': self._get_strategy_effectiveness(),
            'market_adaptation_speed': self.adaptation_events / max(1, self.total_decisions * 0.1),
            'subsystem_health': self._get_subsystem_health(),
            'learning_efficiency': self.meta_learner.get_learning_efficiency(),
            'architecture_evolution_stage': self.meta_learner.architecture_evolver.generations,
            'confidence_recovery_status': self.confidence_recovery_factor,
            'exploration_strategy': self.meta_learner.exploration_strategy.get_current_strategy() if hasattr(self.meta_learner.exploration_strategy, 'get_current_strategy') else 'standard'
        }
    
    def _calculate_performance_trend(self) -> str:
        """Calculate recent performance trend"""
        recent_experiences = self._buffer_recent(20, 'experience')
        if len(recent_experiences) < 10:
            return 'insufficient_data'
        
        rewards = [exp.get('reward', 0) for exp in recent_experiences]
        first_half = np.mean(rewards[:len(rewards)//2])
        second_half = np.mean(rewards[len(rewards)//2:])
        
        if second_half > first_half + 0.1:
            return 'improving'
        elif second_half < first_half - 0.1:
            return 'declining'
        return 'stable'
    
    def _get_strategy_effectiveness(self) -> Dict:
        """Get effectiveness scores for different strategies"""
        effectiveness = {}
        for strategy, performance in self.strategy_performance.items():
            if len(performance) > 0:
                avg_pnl = np.mean(list(performance))
                win_rate = sum(1 for p in performance if p > 0) / len(performance)
                effectiveness[strategy] = {
                    'avg_pnl': avg_pnl,
                    'win_rate': win_rate,
                    'trade_count': len(performance),
                    'score': avg_pnl * win_rate  # Combined effectiveness score
                }
        return effectiveness
    
    def _get_subsystem_health(self) -> Dict:
        """Get health status of each subsystem"""
        recent_experiences = self._buffer_recent(50, 'experience')
        subsystem_performance = {'dna': [], 'temporal': [], 'immune': [], 'microstructure': [], 'dopamine': []}
        
        for exp in recent_experiences:
            trade_data = exp.get('trade_data', {})
            if isinstance(trade_data, dict):
                primary_tool = trade_data.get('primary_tool', 'unknown')
                reward = exp.get('reward', 0)
                
                # Extract base tool name (remove modifiers like '_uncertain')
                base_tool = primary_tool.split('_')[0]
                if base_tool in subsystem_performance:
                    subsystem_performance[base_tool].append(reward)
        
        health_status = {}
        for subsystem, rewards in subsystem_performance.items():
            if len(rewards) > 0:
                avg_reward = np.mean(rewards)
                consistency = 1.0 - np.std(rewards) if len(rewards) > 1 else 1.0
                health_status[subsystem] = {
                    'avg_performance': avg_reward,
                    'consistency': max(0, consistency),
                    'usage_count': len(rewards),
                    'health_score': avg_reward * consistency
                }
            else:
                health_status[subsystem] = {
                    'avg_performance': 0.0,
                    'consistency': 0.5,
                    'usage_count': 0,
                    'health_score': 0.0
                }
        
        return health_status

    def learn_from_rejection(self, rejection_type: str, rejection_data: Dict, reward: float):
        """Learn from order rejections to prevent future repeated failures"""
        try:
            import time
            current_time = time.time()
            
            if rejection_type == 'position_limit':
                # Prevent duplicate processing of the same violation
                if current_time - self.last_processed_violation_time < 0.1:  # Within 0.1 second = duplicate
                    logger.debug("Skipping duplicate position limit violation in trading agent")
                    return
                
                # NEW: Use centralized confidence manager for rejection handling
                self.confidence_manager.handle_position_rejection(rejection_data)
                
                # LEGACY: Update position rejection tracking (for compatibility)
                self.recent_position_rejections += 1
                self.position_rejection_timestamps.append(current_time)
                self.last_position_rejection_time = current_time
                self.last_processed_violation_time = current_time
                self.confidence_violations.append(current_time)
                self.confidence_recovery_factor = max(0.1, self.confidence_recovery_factor - 0.001)
                
                # Adaptive memory window for violation tracking
                cutoff_time = current_time - 60  # Much shorter adaptive window
                while (self.position_rejection_timestamps and 
                       self.position_rejection_timestamps[0] < cutoff_time):
                    self.position_rejection_timestamps.popleft()
                
                # Update recent count based on cleaned timestamps
                self.recent_position_rejections = len(self.position_rejection_timestamps)
                
                # Create learning experience for immediate behavioral change
                rejection_experience = {
                    'rejection_type': rejection_type,
                    'current_position': rejection_data.get('current_position', 0),
                    'max_contracts': rejection_data.get('max_contracts', 10),
                    'tool_used': rejection_data.get('primary_tool', 'unknown'),
                    'exploration_mode': rejection_data.get('exploration_mode', False),
                    'rejection_count': self.recent_position_rejections,
                    'negative_reward': reward,
                    'timestamp': current_time
                }
                
                # Store high-priority negative experience
                self._buffer_append(rejection_experience, 'priority')
                
                # Update meta-learner parameters to be more conservative
                if hasattr(self.meta_learner, 'learn_from_negative_feedback'):
                    self.meta_learner.learn_from_negative_feedback(rejection_data, reward)
                
                # SINGLE PENALTY SYSTEM: Apply moderate penalty once (no cascading)
                # Calculate violation severity with adaptive learning
                violation_severity = rejection_data.get('violation_severity', 1.0)
                base_penalty = -0.01  # Much lighter penalty for neuromorphic discovery
                final_penalty = base_penalty * violation_severity
                
                logger.warning(f"Position limit violation: Applied single penalty: {final_penalty:.3f}")
                
                # Store rejection experience in separate buffer to prevent confidence contamination
                rejection_experience = {
                    'rejection_type': rejection_type,
                    'timestamp': current_time,
                    'tool_used': rejection_data.get('primary_tool', 'unknown'),
                    'exploration_mode': rejection_data.get('exploration_mode', False),
                    'position_context': rejection_data.get('current_position', 0),
                    'single_penalty': final_penalty
                }
                
                # Store in separate rejection buffer (doesn't affect main confidence training)
                self._buffer_append(rejection_experience, 'previous_task')  # Isolated storage
                
                # Apply SINGLE penalty to meta-learner (no reward engine cascading)
                if hasattr(self.meta_learner, 'outcome_history'):
                    # Apply the penalty only once, with isolation from confidence training
                    isolated_penalty = final_penalty * 0.3  # Reduce impact on confidence even further
                    self.meta_learner.outcome_history.append(isolated_penalty)
                
                # Update exploration strategy to avoid position limits
                if hasattr(self.adaptation_engine, 'update_exploration_constraints'):
                    self.adaptation_engine.update_exploration_constraints('position_limit', rejection_data)
                
                logger.info(f"Learning from {rejection_type} rejection: "
                           f"recent_rejections={self.recent_position_rejections}, "
                           f"reward={reward:.1f}, tool={rejection_data.get('primary_tool', 'unknown')}")
                
        except Exception as e:
            logger.error(f"Error in learn_from_rejection: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
    
    def get_confidence_health(self) -> Dict:
        """
        Get comprehensive confidence health status from the confidence manager
        
        Returns:
            Dict: Confidence health information including status, scores, and metrics
        """
        return self.confidence_manager.get_confidence_health()
    
    def get_confidence_debug_info(self) -> Dict:
        """
        Get detailed confidence debug information
        
        Returns:
            Dict: Detailed debug information from confidence manager
        """
        return self.confidence_manager.get_debug_info()
    
    def get_current_confidence(self) -> float:
        """
        Get the current confidence value
        
        Returns:
            float: Current confidence level
        """
        return self.confidence_manager.get_current_confidence()
    
    def _estimate_expected_outcome(self, features: Features, market_data: MarketData) -> float:
        """Estimate expected outcome for dopamine anticipation phase"""
        
        # Simple expected outcome based on signal strength and confidence
        signal_strength = abs(features.overall_signal)
        base_expectation = signal_strength * features.confidence * 0.1
        
        # Add regime confidence influence
        regime_adjustment = features.regime_confidence * 0.05
        
        # Direction-aware expectation
        if features.overall_signal > 0:
            return base_expectation + regime_adjustment
        else:
            return -(base_expectation + regime_adjustment)
    
    def _create_dopamine_market_data(self, market_data: MarketData) -> Dict:
        """Create market data dictionary for dopamine subsystem"""
        
        return {
            'unrealized_pnl': getattr(market_data, 'unrealized_pnl', 0.0),
            'daily_pnl': getattr(market_data, 'daily_pnl', 0.0),
            'open_positions': getattr(market_data, 'open_positions', 0.0),
            'current_price': market_data.prices_1m[-1] if market_data.prices_1m else 0.0,
            'trade_duration': getattr(market_data, 'trade_duration', 0.0)
        }