# advanced_position_management.py - AI learns position scaling and dynamic exits

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import deque

@dataclass
class PositionState:
    """Track detailed position state for AI learning"""
    entry_price: float
    current_size: float
    max_size: float
    entry_time: datetime
    current_pnl: float
    max_favorable_excursion: float  # Best profit seen
    max_adverse_excursion: float    # Worst loss seen
    scales_added: int
    partial_exits: int
    tool_used: str
    entry_confidence: float
    current_market_regime: str

class AdvancedPositionManagementAI(nn.Module):
    """
    AI that learns advanced position management:
    1. When to scale into positions (add size)
    2. When to take partial profits
    3. When to exit completely
    4. How to trail stops dynamically
    """
    
    def __init__(self, market_obs_size: int = 15, position_features_size: int = 20):
        super().__init__()
        
        # Position state encoder
        self.position_encoder = nn.Sequential(
            nn.Linear(position_features_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        
        # Market condition encoder
        self.market_encoder = nn.LSTM(market_obs_size, 64, batch_first=True)
        
        # Decision heads for different actions
        self.scale_head = nn.Sequential(
            nn.Linear(96, 64),  # 32 + 64 combined features
            nn.ReLU(),
            nn.Linear(64, 3)    # no_scale, scale_25%, scale_50%
        )
        
        self.exit_head = nn.Sequential(
            nn.Linear(96, 64),
            nn.ReLU(),
            nn.Linear(64, 4)    # hold, exit_25%, exit_50%, exit_100%
        )
        
        self.trail_stop_head = nn.Sequential(
            nn.Linear(96, 64),
            nn.ReLU(),
            nn.Linear(64, 1)    # trail stop distance as % of current profit
        )
        
        # Value estimation for position management
        self.value_head = nn.Sequential(
            nn.Linear(96, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # Regime-specific learning
        self.regime_weights = nn.Parameter(torch.randn(4, 3))  # 4 regimes, 3 actions
        
    def forward(self, market_obs, position_features):
        """
        Args:
            market_obs: (batch, seq_len, market_obs_size) - Recent market data
            position_features: (batch, position_features_size) - Current position state
        """
        # Encode position state
        position_repr = self.position_encoder(position_features)
        
        # Encode market conditions
        market_encoded, _ = self.market_encoder(market_obs)
        market_repr = market_encoded[:, -1]  # Last timestep
        
        # Combine representations
        combined = torch.cat([position_repr, market_repr], dim=-1)
        
        # Generate decisions
        scale_logits = self.scale_head(combined)
        exit_logits = self.exit_head(combined)
        trail_distance = torch.sigmoid(self.trail_stop_head(combined)) * 0.5  # 0-50% of profit
        position_value = self.value_head(combined)
        
        return {
            'scale_logits': scale_logits,
            'exit_logits': exit_logits,
            'trail_distance': trail_distance,
            'position_value': position_value,
            'combined_features': combined
        }

class PositionManagementLearner:
    """Learns advanced position management strategies"""
    
    def __init__(self, base_agent):
        self.base_agent = base_agent
        self.device = base_agent.device
        
        # Advanced position management network
        self.position_ai = AdvancedPositionManagementAI().to(self.device)
        self.position_optimizer = torch.optim.Adam(self.position_ai.parameters(), lr=1e-4)
        
        # Position tracking
        self.current_position: Optional[PositionState] = None
        self.position_history = deque(maxlen=1000)
        
        # Learning data
        self.scaling_outcomes = deque(maxlen=500)
        self.exit_outcomes = deque(maxlen=500)
        self.trail_outcomes = deque(maxlen=500)
        
        # Performance tracking
        self.scaling_success_rate = 0.0
        self.optimal_exit_timing = {}
        self.best_trail_strategies = {}
        
    def create_position_features(self, position: PositionState, current_price: float) -> np.ndarray:
        """Extract features for position management AI"""
        
        if not position:
            return np.zeros(20, dtype=np.float32)
        
        # Calculate current metrics
        current_pnl_pct = (current_price - position.entry_price) / position.entry_price
        if position.current_size < 0:  # Short position
            current_pnl_pct = -current_pnl_pct
        
        time_in_position = (datetime.now() - position.entry_time).total_seconds() / 3600  # Hours
        size_ratio = position.current_size / position.max_size
        
        # Risk metrics
        drawdown_from_peak = (position.max_favorable_excursion - current_pnl_pct) if position.max_favorable_excursion > 0 else 0
        pain_ratio = abs(position.max_adverse_excursion) / max(abs(current_pnl_pct), 0.001)
        
        features = [
            current_pnl_pct,                    # Current P&L %
            position.max_favorable_excursion,   # Best profit seen
            position.max_adverse_excursion,     # Worst loss seen
            drawdown_from_peak,                 # Drawdown from peak
            time_in_position / 24.0,            # Time in position (days)
            size_ratio,                         # Current size vs max
            float(position.scales_added) / 5.0, # Scales added (normalized)
            float(position.partial_exits) / 5.0, # Partial exits taken
            position.entry_confidence,          # Original entry confidence
            pain_ratio,                         # Pain vs gain ratio
            
            # Tool effectiveness indicators
            1.0 if position.tool_used == 'dna' else 0.0,
            1.0 if position.tool_used == 'micro' else 0.0,
            1.0 if position.tool_used == 'temporal' else 0.0,
            1.0 if position.tool_used == 'immune' else 0.0,
            
            # Market regime indicators
            1.0 if position.current_market_regime == 'trending' else 0.0,
            1.0 if position.current_market_regime == 'volatile' else 0.0,
            1.0 if position.current_market_regime == 'sideways' else 0.0,
            1.0 if position.current_market_regime == 'reversal' else 0.0,
            
            # Momentum indicators
            1.0 if current_pnl_pct > 0.01 else 0.0,  # Strong positive
            1.0 if current_pnl_pct < -0.01 else 0.0, # Strong negative
        ]
        
        return np.array(features[:20], dtype=np.float32)
    
    def should_scale_position(self, market_obs: np.ndarray, current_price: float) -> Dict:
        """AI decides whether to scale into position"""
        
        if not self.current_position:
            return {'action': 'no_scale', 'confidence': 0.0}
        
        with torch.no_grad():
            # Prepare inputs
            position_features = self.create_position_features(self.current_position, current_price)
            
            market_tensor = torch.tensor(market_obs, dtype=torch.float32, device=self.device)
            market_tensor = market_tensor.unsqueeze(0).unsqueeze(0)
            
            position_tensor = torch.tensor(position_features, dtype=torch.float32, device=self.device)
            position_tensor = position_tensor.unsqueeze(0)
            
            # Get AI decision
            outputs = self.position_ai(market_tensor, position_tensor)
            scale_probs = F.softmax(outputs['scale_logits'], dim=-1).cpu().numpy()[0]
            
            actions = ['no_scale', 'scale_25%', 'scale_50%']
            action = actions[np.argmax(scale_probs)]
            confidence = float(np.max(scale_probs))
            
            return {
                'action': action,
                'confidence': confidence,
                'scale_amount': 0.25 if 'scale_25%' in action else 0.5 if 'scale_50%' in action else 0.0,
                'reasoning': self._explain_scaling_decision(action, confidence, position_features)
            }
    
    def should_exit_position(self, market_obs: np.ndarray, current_price: float) -> Dict:
        """AI decides whether to exit position (partially or completely)"""
        
        if not self.current_position:
            return {'action': 'hold', 'confidence': 0.0}
        
        with torch.no_grad():
            position_features = self.create_position_features(self.current_position, current_price)
            
            market_tensor = torch.tensor(market_obs, dtype=torch.float32, device=self.device)
            market_tensor = market_tensor.unsqueeze(0).unsqueeze(0)
            
            position_tensor = torch.tensor(position_features, dtype=torch.float32, device=self.device)
            position_tensor = position_tensor.unsqueeze(0)
            
            outputs = self.position_ai(market_tensor, position_tensor)
            exit_probs = F.softmax(outputs['exit_logits'], dim=-1).cpu().numpy()[0]
            
            actions = ['hold', 'exit_25%', 'exit_50%', 'exit_100%']
            action = actions[np.argmax(exit_probs)]
            confidence = float(np.max(exit_probs))
            
            return {
                'action': action,
                'confidence': confidence,
                'exit_amount': 0.25 if 'exit_25%' in action else 0.5 if 'exit_50%' in action else 1.0 if 'exit_100%' in action else 0.0,
                'reasoning': self._explain_exit_decision(action, confidence, position_features)
            }
    
    def get_trail_stop_distance(self, market_obs: np.ndarray, current_price: float) -> float:
        """AI determines optimal trailing stop distance"""
        
        if not self.current_position or self.current_position.current_pnl <= 0:
            return 0.0
        
        with torch.no_grad():
            position_features = self.create_position_features(self.current_position, current_price)
            
            market_tensor = torch.tensor(market_obs, dtype=torch.float32, device=self.device)
            market_tensor = market_tensor.unsqueeze(0).unsqueeze(0)
            
            position_tensor = torch.tensor(position_features, dtype=torch.float32, device=self.device)
            position_tensor = position_tensor.unsqueeze(0)
            
            outputs = self.position_ai(market_tensor, position_tensor)
            trail_distance = float(outputs['trail_distance'].cpu().numpy()[0])
            
            return trail_distance
    
    def start_position(self, entry_price: float, initial_size: float, tool_used: str, 
                      entry_confidence: float, market_regime: str):
        """Start tracking a new position"""
        
        self.current_position = PositionState(
            entry_price=entry_price,
            current_size=initial_size,
            max_size=initial_size * 3.0,  # Allow up to 3x scaling
            entry_time=datetime.now(),
            current_pnl=0.0,
            max_favorable_excursion=0.0,
            max_adverse_excursion=0.0,
            scales_added=0,
            partial_exits=0,
            tool_used=tool_used,
            entry_confidence=entry_confidence,
            current_market_regime=market_regime
        )
        
        print(f"🎯 POSITION STARTED: {tool_used} tool, size {initial_size}, confidence {entry_confidence:.3f}")
    
    def update_position(self, current_price: float):
        """Update position metrics"""
        
        if not self.current_position:
            return
        
        # Calculate current P&L
        pnl_points = current_price - self.current_position.entry_price
        if self.current_position.current_size < 0:  # Short
            pnl_points = -pnl_points
        
        pnl_pct = pnl_points / self.current_position.entry_price
        self.current_position.current_pnl = pnl_pct
        
        # Update excursion metrics
        self.current_position.max_favorable_excursion = max(
            self.current_position.max_favorable_excursion, pnl_pct
        )
        self.current_position.max_adverse_excursion = min(
            self.current_position.max_adverse_excursion, pnl_pct
        )
    
    def add_to_position(self, additional_size: float, current_price: float):
        """Scale into position"""
        
        if not self.current_position:
            return False
        
        if abs(self.current_position.current_size + additional_size) > self.current_position.max_size:
            return False  # Would exceed max size
        
        # Calculate new average entry price
        current_value = self.current_position.current_size * self.current_position.entry_price
        additional_value = additional_size * current_price
        new_total_size = self.current_position.current_size + additional_size
        
        self.current_position.entry_price = (current_value + additional_value) / new_total_size
        self.current_position.current_size = new_total_size
        self.current_position.scales_added += 1
        
        print(f"📈 SCALED POSITION: +{additional_size:.2f}, new size: {new_total_size:.2f}, new avg: ${self.current_position.entry_price:.2f}")
        return True
    
    def partial_exit(self, exit_size: float, exit_price: float) -> float:
        """Partially exit position"""
        
        if not self.current_position:
            return 0.0
        
        # Calculate P&L on exited portion
        exit_pnl = (exit_price - self.current_position.entry_price) * exit_size
        if self.current_position.current_size < 0:  # Short
            exit_pnl = -exit_pnl
        
        self.current_position.current_size -= exit_size
        self.current_position.partial_exits += 1
        
        print(f"💰 PARTIAL EXIT: -{exit_size:.2f}, remaining: {self.current_position.current_size:.2f}, P&L: ${exit_pnl:.2f}")
        return exit_pnl
    
    def close_position(self, exit_price: float) -> float:
        """Close entire position"""
        
        if not self.current_position:
            return 0.0
        
        # Calculate total P&L
        total_pnl = (exit_price - self.current_position.entry_price) * self.current_position.current_size
        if self.current_position.current_size < 0:  # Short
            total_pnl = -total_pnl
        
        # Store position for learning
        self.position_history.append(self.current_position)
        
        print(f"🏁 POSITION CLOSED: P&L ${total_pnl:.2f}, held {(datetime.now() - self.current_position.entry_time).total_seconds()/3600:.1f}h")
        
        self.current_position = None
        return total_pnl
    
    def _explain_scaling_decision(self, action: str, confidence: float, features: np.ndarray) -> str:
        """Explain why AI decided to scale or not"""
        
        current_pnl = features[0]
        time_in_position = features[4] * 24  # Convert back to hours
        
        if action == 'no_scale':
            if current_pnl < 0:
                return f"No scaling: Position underwater ({current_pnl:.2%})"
            else:
                return f"No scaling: Waiting for better entry (confidence: {confidence:.2f})"
        else:
            return f"Scaling {action}: Position profitable ({current_pnl:.2%}), confidence: {confidence:.2f}"
    
    def _explain_exit_decision(self, action: str, confidence: float, features: np.ndarray) -> str:
        """Explain why AI decided to exit or hold"""
        
        current_pnl = features[0]
        max_favorable = features[1]
        drawdown_from_peak = features[3]
        
        if action == 'hold':
            return f"Holding: Still profitable ({current_pnl:.2%}), low exit confidence"
        elif 'exit_100%' in action:
            if drawdown_from_peak > 0.02:
                return f"Full exit: Significant drawdown from peak ({drawdown_from_peak:.2%})"
            else:
                return f"Full exit: Taking profits at {current_pnl:.2%}"
        else:
            return f"Partial exit ({action}): Locking in profits, keeping exposure"

# Integration with existing trade manager
class EnhancedTradeManagerWithPositionAI:
    """Enhanced trade manager with advanced position management"""
    
    def __init__(self, base_trade_manager):
        self.base_manager = base_trade_manager
        self.position_learner = PositionManagementLearner(base_trade_manager.agent)
        
    def on_new_bar(self, msg: Dict):
        """Enhanced bar processing with position management"""
        
        # Get current price
        price = msg.get("price_1m", [4000.0])[-1] if msg.get("price_1m") else 4000.0
        
        # Update position metrics if in position
        if self.position_learner.current_position:
            self.position_learner.update_position(price)
            
            # Get market observation for position decisions
            market_obs = self.base_manager.env.get_obs()
            
            # Check for scaling opportunities
            scale_decision = self.position_learner.should_scale_position(market_obs, price)
            if scale_decision['action'] != 'no_scale' and scale_decision['confidence'] > 0.6:
                scale_size = scale_decision['scale_amount'] * abs(self.position_learner.current_position.current_size)
                if self.position_learner.add_to_position(scale_size, price):
                    print(f"🔄 AI SCALING: {scale_decision['reasoning']}")
            
            # Check for exit opportunities
            exit_decision = self.position_learner.should_exit_position(market_obs, price)
            if exit_decision['action'] != 'hold' and exit_decision['confidence'] > 0.7:
                
                if exit_decision['action'] == 'exit_100%':
                    pnl = self.position_learner.close_position(price)
                    print(f"🚪 AI FULL EXIT: {exit_decision['reasoning']}")
                    # Notify NinjaTrader to close position
                    self._send_exit_signal()
                    
                elif 'exit_' in exit_decision['action']:
                    exit_size = exit_decision['exit_amount'] * abs(self.position_learner.current_position.current_size)
                    pnl = self.position_learner.partial_exit(exit_size, price)
                    print(f"🚪 AI PARTIAL EXIT: {exit_decision['reasoning']}")
                    # Send partial exit signal to NinjaTrader
        
        # Continue with normal entry logic
        self.base_manager.on_new_bar(msg)
        
        # If new position started, register it
        if (self.base_manager.current_position['in_position'] and 
            not self.position_learner.current_position):
            
            self.position_learner.start_position(
                entry_price=price,
                initial_size=1.0,  # Base size
                tool_used=self.base_manager.last_decision.get('primary_tool', 'unknown'),
                entry_confidence=self.base_manager.last_decision.get('confidence', 0.0),
                market_regime=self.base_manager.last_decision.get('market_regime', 'unknown')
            )
    
    def _send_exit_signal(self):
        """Send exit signal to NinjaTrader"""
        self.base_manager.tcp_bridge.send_signal(0, 0.9, "AI_dynamic_exit")

# LEARNING SCHEDULE: When these features activate

def get_position_management_learning_schedule():
    """Define when advanced features become available"""
    
    return {
        "basic_entries": 0,           # Available immediately
        "position_scaling": 50,       # After 50 successful trades
        "partial_exits": 25,          # After 25 trades
        "trailing_stops": 100,        # After 100 trades
        "advanced_combinations": 200, # After 200 trades
    }