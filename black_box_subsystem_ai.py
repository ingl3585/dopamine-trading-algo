# black_box_subsystem_ai.py - The missing orchestrator that learns strategic tool usage

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from collections import deque, defaultdict
import logging
from dataclasses import dataclass
import random

log = logging.getLogger(__name__)

@dataclass
class MarketContext:
    """Encapsulates current market state for tool selection"""
    volatility: float
    trend_strength: float
    momentum: float
    time_of_day: float
    volume_profile: float
    recent_performance: float
    
class ToolSelector(nn.Module):
    """Neural network that learns when to use which subsystem tools"""
    
    def __init__(self, context_size: int = 10, hidden_size: int = 64):
        super().__init__()
        
        # Market context encoder
        self.context_encoder = nn.Sequential(
            nn.Linear(context_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        
        # Tool confidence predictor (for each of the 4 subsystems)
        self.tool_confidence = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 4)  # DNA, Micro, Temporal, Immune
        )
        
        # Tool combination predictor
        self.combination_value = nn.Sequential(
            nn.Linear(hidden_size + 4, 32),  # context + tool confidences
            nn.ReLU(),
            nn.Linear(32, 6)  # 6 possible pairs
        )
        
        # Market regime classifier
        self.regime_classifier = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 4)  # trending, volatile, sideways, reversal
        )
    
    def forward(self, market_context):
        # Encode market context
        context_features = self.context_encoder(market_context)
        
        # Predict tool confidences
        tool_confs = torch.sigmoid(self.tool_confidence(context_features))
        
        # Predict combination values
        combo_input = torch.cat([context_features, tool_confs], dim=-1)
        combo_values = torch.sigmoid(self.combination_value(combo_input))
        
        # Classify market regime
        regime_probs = F.softmax(self.regime_classifier(context_features), dim=-1)
        
        return {
            'tool_confidences': tool_confs,
            'combination_values': combo_values,
            'regime_probs': regime_probs,
            'context_features': context_features
        }

class DecisionIntegrator(nn.Module):
    """Integrates subsystem outputs with learned tool usage weights"""
    
    def __init__(self, subsystem_features_size: int = 16, hidden_size: int = 64):
        super().__init__()
        
        # Process subsystem outputs
        self.subsystem_processor = nn.Sequential(
            nn.Linear(subsystem_features_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        
        # Attention mechanism for subsystem weighting
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=4, batch_first=True)
        
        # Final decision maker
        self.decision_head = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),  # attention output + tool selector output
            nn.ReLU(),
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 3)  # buy/sell/hold
        )
        
        # Risk management heads
        self.risk_heads = nn.ModuleDict({
            'use_stop': nn.Sequential(nn.Linear(hidden_size * 2, 16), nn.ReLU(), nn.Linear(16, 1)),
            'stop_size': nn.Sequential(nn.Linear(hidden_size * 2, 16), nn.ReLU(), nn.Linear(16, 1)),
            'use_target': nn.Sequential(nn.Linear(hidden_size * 2, 16), nn.ReLU(), nn.Linear(16, 1)),
            'target_size': nn.Sequential(nn.Linear(hidden_size * 2, 16), nn.ReLU(), nn.Linear(16, 1))
        })
        
        # Overall confidence
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_size * 2, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
    
    def forward(self, subsystem_features, tool_weights, context_features):
        # Process subsystem features
        processed_subsystems = self.subsystem_processor(subsystem_features)
        
        # Reshape for attention (treat each subsystem as a sequence element)
        subsystem_seq = processed_subsystems.view(processed_subsystems.shape[0], 4, -1)
        tool_weights_seq = tool_weights.unsqueeze(-1).expand(-1, -1, subsystem_seq.shape[-1])
        
        # Apply attention with tool weights as importance
        attended_output, attention_weights = self.attention(
            subsystem_seq * tool_weights_seq,
            subsystem_seq,
            subsystem_seq
        )
        
        # Global pool attention output
        pooled_subsystems = attended_output.mean(dim=1)
        
        # Combine with context
        combined_features = torch.cat([pooled_subsystems, context_features], dim=-1)
        
        # Generate all outputs
        action_logits = self.decision_head(combined_features)
        
        risk_outputs = {}
        for name, head in self.risk_heads.items():
            risk_outputs[name] = torch.sigmoid(head(combined_features))
        
        confidence = torch.sigmoid(self.confidence_head(combined_features))
        
        return {
            'action_logits': action_logits,
            'risk_management': risk_outputs,
            'confidence': confidence,
            'attention_weights': attention_weights,
            'combined_features': combined_features
        }

class BlackBoxSubsystemAI:
    """
    Main black box AI that learns to strategically orchestrate your existing subsystems
    
    Key Learning Objectives:
    1. Market regime recognition (trending vs volatile vs sideways vs reversal)
    2. Tool selection based on market conditions
    3. Tool combination strategies
    4. Risk management per tool type
    5. Exit timing optimization
    """
    
    def __init__(self, intelligence_engine):
        self.intelligence_engine = intelligence_engine
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Neural networks
        self.tool_selector = ToolSelector().to(self.device)
        self.decision_integrator = DecisionIntegrator().to(self.device)
        
        # Optimizers
        self.tool_optimizer = torch.optim.Adam(self.tool_selector.parameters(), lr=1e-4)
        self.decision_optimizer = torch.optim.Adam(self.decision_integrator.parameters(), lr=1e-4)
        
        # Experience tracking
        self.experience_buffer = deque(maxlen=10000)
        self.tool_performance = {
            'dna': deque(maxlen=200),
            'micro': deque(maxlen=200), 
            'temporal': deque(maxlen=200),
            'immune': deque(maxlen=200)
        }
        
        # Tool usage patterns
        self.regime_tool_usage = defaultdict(lambda: defaultdict(list))  # regime -> tool -> outcomes
        self.tool_combinations = defaultdict(list)  # combination -> outcomes
        
        # Learning state
        self.learning_active = False
        self.step_count = 0
        self.recent_performance = deque(maxlen=50)
        
        # Tool discovery learning
        self.exploration_rate = 0.3
        self.exploration_decay = 0.995
        self.min_exploration = 0.05
        
        log.info("Black Box Subsystem AI initialized")
        log.info("Learning objectives: Strategic tool usage, regime adaptation, risk management")
    
    def extract_market_context(self, prices: List[float], volumes: List[float], 
                             timestamp: datetime) -> np.ndarray:
        """Extract market context features for tool selection"""
        
        if len(prices) < 20:
            return np.zeros(10, dtype=np.float32)
        
        recent_prices = prices[-20:]
        recent_volumes = volumes[-20:] if len(volumes) >= 20 else [1000] * 20
        
        # Calculate context features
        price_changes = np.diff(recent_prices) / recent_prices[:-1]
        
        volatility = np.std(price_changes)
        trend_strength = abs(np.mean(price_changes)) / (volatility + 1e-8)
        momentum = (recent_prices[-1] - recent_prices[-5]) / recent_prices[-5] if len(recent_prices) >= 5 else 0
        
        # Time features
        hour = timestamp.hour / 24.0
        minute = timestamp.minute / 60.0
        time_of_day = hour + minute / 60.0
        
        # Volume profile
        avg_volume = np.mean(recent_volumes)
        volume_spike = (recent_volumes[-1] - avg_volume) / (avg_volume + 1)
        
        # Recent performance context
        recent_perf = np.mean(list(self.recent_performance)) if self.recent_performance else 0.0
        
        # Price position in recent range
        price_high = max(recent_prices)
        price_low = min(recent_prices)
        price_position = (recent_prices[-1] - price_low) / (price_high - price_low + 1e-8)
        
        # Range expansion/contraction
        early_range = max(recent_prices[:10]) - min(recent_prices[:10])
        recent_range = max(recent_prices[-10:]) - min(recent_prices[-10:])
        range_change = (recent_range - early_range) / (early_range + 1e-8)
        
        return np.array([
            volatility,
            trend_strength, 
            momentum,
            time_of_day,
            volume_spike,
            recent_perf,
            price_position,
            range_change,
            len(recent_prices) / 100.0,  # data sufficiency
            1.0  # bias term
        ], dtype=np.float32)
    
    def extract_subsystem_features(self, intelligence_result: Dict) -> np.ndarray:
        """Extract features from your existing subsystems"""
        
        features = []
        
        # DNA system features
        dna_signal = intelligence_result['subsystem_signals'].get('dna', 0.0)
        dna_confidence = intelligence_result['subsystem_scores'].get('dna', 0.0)
        dna_patterns_found = intelligence_result.get('similar_patterns_count', 0) / 10.0  # normalize
        dna_sequence_quality = min(intelligence_result.get('dna_sequence_length', 0) / 20.0, 1.0)
        
        features.extend([dna_signal, dna_confidence, dna_patterns_found, dna_sequence_quality])
        
        # Micro pattern features
        micro_signal = intelligence_result['subsystem_signals'].get('micro', 0.0)
        micro_confidence = intelligence_result['subsystem_scores'].get('micro', 0.0)
        micro_pattern_strength = abs(micro_signal) * micro_confidence
        micro_pattern_reliability = 1.0 if intelligence_result.get('micro_pattern_id') else 0.0
        
        features.extend([micro_signal, micro_confidence, micro_pattern_strength, micro_pattern_reliability])
        
        # Temporal features
        temporal_signal = intelligence_result['subsystem_signals'].get('temporal', 0.0)
        temporal_confidence = intelligence_result['subsystem_scores'].get('temporal', 0.0)
        temporal_timing_quality = temporal_confidence  # how good is the timing
        temporal_session_relevance = 1.0 if abs(temporal_signal) > 0.1 else 0.0
        
        features.extend([temporal_signal, temporal_confidence, temporal_timing_quality, temporal_session_relevance])
        
        # Immune system features
        immune_signal = intelligence_result['subsystem_signals'].get('immune', 0.0)
        immune_confidence = intelligence_result['subsystem_scores'].get('immune', 0.0)
        danger_detected = 1.0 if intelligence_result.get('is_dangerous_pattern') else 0.0
        beneficial_detected = 1.0 if intelligence_result.get('is_beneficial_pattern') else 0.0
        
        features.extend([immune_signal, immune_confidence, danger_detected, beneficial_detected])
        
        return np.array(features, dtype=np.float32)
    
    def make_complete_decision(self, market_obs: np.ndarray, prices: List[float], 
                             volumes: List[float], current_price: float, 
                             timestamp: datetime, in_position: bool = False) -> Dict:
        """
        Make complete trading decision using strategic subsystem orchestration
        """
        
        # Get intelligence from your existing subsystems
        intelligence_result = self.intelligence_engine.process_market_data(prices, volumes, timestamp)
        
        # Extract context and subsystem features
        market_context = self.extract_market_context(prices, volumes, timestamp)
        subsystem_features = self.extract_subsystem_features(intelligence_result)
        
        with torch.no_grad():
            # Convert to tensors
            context_tensor = torch.tensor(market_context, dtype=torch.float32, device=self.device).unsqueeze(0)
            subsystem_tensor = torch.tensor(subsystem_features, dtype=torch.float32, device=self.device).unsqueeze(0)
            
            # Tool selection decision
            tool_outputs = self.tool_selector(context_tensor)
            tool_confidences = tool_outputs['tool_confidences'][0]
            regime_probs = tool_outputs['regime_probs'][0]
            
            # Apply exploration to tool selection
            if random.random() < self.exploration_rate and self.learning_active:
                # Exploration: occasionally try different tool combinations
                exploration_bonus = torch.randn_like(tool_confidences) * 0.2
                tool_confidences = torch.clamp(tool_confidences + exploration_bonus, 0, 1)
            
            # Decision integration
            decision_outputs = self.decision_integrator(
                subsystem_tensor, 
                tool_confidences.unsqueeze(0),
                tool_outputs['context_features']
            )
            
            # Extract outputs
            action_logits = decision_outputs['action_logits'][0]
            action_probs = F.softmax(action_logits, dim=0)
            action = torch.argmax(action_probs).item()
            
            confidence = float(decision_outputs['confidence'][0])
            
            # Risk management
            risk_mgmt = decision_outputs['risk_management']
            use_stop = float(risk_mgmt['use_stop'][0]) > 0.5
            stop_size_pct = float(risk_mgmt['stop_size'][0]) * 3.0  # 0-3%
            use_target = float(risk_mgmt['use_target'][0]) > 0.5  
            target_size_pct = float(risk_mgmt['target_size'][0]) * 5.0  # 0-5%
            
            # Calculate actual prices
            stop_price = None
            target_price = None
            
            if use_stop and action != 0:
                if action == 1:  # Long
                    stop_price = current_price * (1 - stop_size_pct / 100)
                else:  # Short
                    stop_price = current_price * (1 + stop_size_pct / 100)
            
            if use_target and action != 0:
                if action == 1:  # Long
                    target_price = current_price * (1 + target_size_pct / 100)
                else:  # Short
                    target_price = current_price * (1 - target_size_pct / 100)
            
            # Tool analysis
            tool_names = ['dna', 'micro', 'temporal', 'immune']
            tool_trust = {tool_names[i]: float(tool_confidences[i]) for i in range(4)}
            
            # Determine primary and secondary tools
            sorted_tools = sorted(tool_trust.items(), key=lambda x: x[1], reverse=True)
            primary_tool = sorted_tools[0][0]
            secondary_tool = sorted_tools[1][0] if len(sorted_tools) > 1 and sorted_tools[1][1] > 0.3 else None
            
            # Market regime
            regime_names = ['trending', 'volatile', 'sideways', 'reversal']
            regime = regime_names[torch.argmax(regime_probs).item()]
            
            # Generate reasoning
            reasoning = self._generate_ai_reasoning(
                primary_tool, secondary_tool, regime, 
                tool_trust[primary_tool], confidence, action
            )
            
            # Should exit (if in position)
            should_exit = in_position and (
                (confidence < 0.3) or  # Low confidence
                (intelligence_result.get('is_dangerous_pattern', False)) or  # Danger detected
                (action == 0)  # Hold signal while in position
            )
            
            return {
                'action': action,
                'confidence': confidence,
                'use_stop': use_stop,
                'stop_price': stop_price,
                'stop_distance_pct': stop_size_pct,
                'use_target': use_target,
                'target_price': target_price,
                'target_distance_pct': target_size_pct,
                'should_exit': should_exit,
                'position_size': 1.0,  # Always 1 contract for now
                
                # Tool orchestration info
                'primary_tool': primary_tool,
                'secondary_tool': secondary_tool,
                'tool_trust': tool_trust,
                'market_regime': regime,
                'reasoning': reasoning,
                
                # Raw data for learning
                'raw_market_obs': market_obs.copy(),
                'raw_subsystem_features': subsystem_features.copy(),
                'raw_intelligence_result': intelligence_result,
                'tool_outputs': {k: v.cpu().numpy() if torch.is_tensor(v) else v for k, v in tool_outputs.items()},
                'decision_outputs': {k: v.cpu().numpy() if torch.is_tensor(v) else v for k, v in decision_outputs.items()}
            }
    
    def _generate_ai_reasoning(self, primary_tool: str, secondary_tool: str, 
                              regime: str, primary_trust: float, confidence: float, action: int) -> str:
        """Generate human-readable AI reasoning"""
        
        action_names = ['Hold', 'Buy', 'Sell']
        action_name = action_names[action]
        
        parts = []
        parts.append(f"Action: {action_name}")
        parts.append(f"Primary Tool: {primary_tool.upper()} (trust: {primary_trust:.2f})")
        
        if secondary_tool:
            parts.append(f"Secondary: {secondary_tool.upper()}")
        
        parts.append(f"Market Regime: {regime}")
        parts.append(f"AI Confidence: {confidence:.2f}")
        
        # Add regime-specific reasoning
        if regime == 'trending' and primary_tool == 'dna':
            parts.append("DNA patterns work well in trends")
        elif regime == 'volatile' and primary_tool == 'immune':
            parts.append("Immune system protects in volatility")
        elif regime == 'reversal' and primary_tool == 'micro':
            parts.append("Micro patterns catch reversals")
        elif primary_tool == 'temporal':
            parts.append("Timing-based entry")
        
        return " | ".join(parts)
    
    def store_experience(self, decision_data: Dict, reward: float, 
                        next_market_obs: np.ndarray, next_subsystem_features: np.ndarray, done: bool):
        """Store experience for learning with tool performance tracking"""
        
        experience = {
            'market_context': decision_data.get('raw_market_obs', np.zeros(10)),
            'subsystem_features': decision_data.get('raw_subsystem_features', np.zeros(16)),
            'action': decision_data['action'],
            'reward': reward,
            'primary_tool': decision_data['primary_tool'],
            'tool_trust': decision_data['tool_trust'],
            'market_regime': decision_data['market_regime'],
            'confidence': decision_data['confidence'],
            'next_market_context': next_market_obs,
            'next_subsystem_features': next_subsystem_features,
            'done': done
        }
        
        self.experience_buffer.append(experience)
        
        # Track tool performance
        primary_tool = decision_data['primary_tool']
        tool_success = 1.0 if reward > 0.01 else 0.0
        
        self.tool_performance[primary_tool].append(tool_success)
        
        # Track regime-tool combinations
        regime = decision_data['market_regime']
        self.regime_tool_usage[regime][primary_tool].append(tool_success)
        
        # Track tool combinations
        if decision_data.get('secondary_tool'):
            combo = f"{primary_tool}_{decision_data['secondary_tool']}"
            self.tool_combinations[combo].append(tool_success)
        
        # Update recent performance
        self.recent_performance.append(reward)
        
        # Decay exploration
        if self.learning_active:
            self.exploration_rate = max(self.min_exploration, self.exploration_rate * self.exploration_decay)
        
        # Start learning if enough experience
        if len(self.experience_buffer) >= 100 and not self.learning_active:
            self.learning_active = True
            log.info("Black Box AI learning activated - sufficient experience collected")
    
    def get_subsystem_usage_report(self) -> str:
        """Generate comprehensive subsystem usage report"""
        
        report = f"""
=== BLACK BOX AI - SUBSYSTEM ORCHESTRATION REPORT ===

Learning Status:
- Experience Buffer: {len(self.experience_buffer)} samples
- Learning Active: {self.learning_active}
- Exploration Rate: {self.exploration_rate:.3f}
- Training Steps: {self.step_count}

Tool Performance Analysis:
"""
        
        for tool in ['dna', 'micro', 'temporal', 'immune']:
            performance_history = list(self.tool_performance[tool])
            if performance_history:
                success_rate = np.mean(performance_history)
                recent_performance = np.mean(performance_history[-20:]) if len(performance_history) >= 20 else success_rate
                total_usage = len(performance_history)
                
                report += f"  {tool.upper()}: {success_rate:.1%} overall ({total_usage} uses), {recent_performance:.1%} recent\n"
            else:
                report += f"  {tool.upper()}: No usage yet\n"
        
        # Regime-tool analysis
        report += f"\nRegime-Tool Performance:\n"
        for regime in ['trending', 'volatile', 'sideways', 'reversal']:
            if regime in self.regime_tool_usage:
                report += f"  {regime.upper()}:\n"
                for tool, outcomes in self.regime_tool_usage[regime].items():
                    if outcomes:
                        success_rate = np.mean(outcomes)
                        report += f"    {tool}: {success_rate:.1%} ({len(outcomes)} uses)\n"
        
        # Tool combinations
        if any(self.tool_combinations.values()):
            report += f"\nTool Combinations:\n"
            for combo, outcomes in self.tool_combinations.items():
                if outcomes:
                    success_rate = np.mean(outcomes)
                    report += f"  {combo.replace('_', ' + ').upper()}: {success_rate:.1%} ({len(outcomes)} uses)\n"
        
        # Recent performance
        if self.recent_performance:
            recent_avg = np.mean(list(self.recent_performance)[-20:])
            report += f"\nRecent AI Performance: {recent_avg:.4f} avg reward\n"
        
        report += f"\nAI is learning optimal subsystem orchestration patterns!"
        
        return report
    
    def save_model(self, filepath: str):
        """Save the black box AI models"""
        checkpoint = {
            'tool_selector_state_dict': self.tool_selector.state_dict(),
            'decision_integrator_state_dict': self.decision_integrator.state_dict(),
            'tool_optimizer_state_dict': self.tool_optimizer.state_dict(),
            'decision_optimizer_state_dict': self.decision_optimizer.state_dict(),
            'step_count': self.step_count,
            'exploration_rate': self.exploration_rate,
            'learning_active': self.learning_active,
            'tool_performance': {k: list(v) for k, v in self.tool_performance.items()},
            'regime_tool_usage': dict(self.regime_tool_usage),
            'tool_combinations': dict(self.tool_combinations)
        }
        
        torch.save(checkpoint, filepath)
        log.info(f"Black Box AI model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained black box AI model"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.tool_selector.load_state_dict(checkpoint['tool_selector_state_dict'])
        self.decision_integrator.load_state_dict(checkpoint['decision_integrator_state_dict'])
        self.tool_optimizer.load_state_dict(checkpoint['tool_optimizer_state_dict'])
        self.decision_optimizer.load_state_dict(checkpoint['decision_optimizer_state_dict'])
        
        self.step_count = checkpoint['step_count']
        self.exploration_rate = checkpoint['exploration_rate']
        self.learning_active = checkpoint['learning_active']
        
        # Restore performance tracking
        for tool, history in checkpoint['tool_performance'].items():
            self.tool_performance[tool] = deque(history, maxlen=200)
        
        self.regime_tool_usage = defaultdict(lambda: defaultdict(list), checkpoint['regime_tool_usage'])
        self.tool_combinations = defaultdict(list, checkpoint['tool_combinations'])
        
        log.info(f"Black Box AI model loaded from {filepath}")
        log.info(f"Resumed with {self.step_count} training steps, exploration: {self.exploration_rate:.3f}")