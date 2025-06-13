# black_box_subsystem_ai.py
# Clean black box AI that learns to use your existing subsystems as tools

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from collections import deque
import random
import time
import threading
from datetime import datetime

log = logging.getLogger(__name__)

class SubsystemToolNetwork(nn.Module):
    """
    Black box AI that learns to use your existing subsystems as intelligent tools
    """
    
    def __init__(self, market_obs_size: int = 15, subsystem_features_size: int = 20):
        super().__init__()
        
        # Market observation encoder
        self.market_encoder = nn.LSTM(market_obs_size, 64, batch_first=True)
        
        # Subsystem insight processor - learns to interpret your subsystems
        self.subsystem_processor = nn.Sequential(
            nn.Linear(subsystem_features_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        
        # Tool attention - AI learns which subsystems to trust WHEN
        self.tool_attention = nn.MultiheadAttention(32, 4, batch_first=True)
        
        # Context network - combines market + subsystem insights
        self.context_network = nn.Sequential(
            nn.Linear(64 + 32, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # Decision heads - AI learns complete trading strategy
        self.action_head = nn.Linear(64, 3)           # buy/sell/hold
        self.position_size_head = nn.Linear(64, 1)    # position sizing
        self.use_stop_head = nn.Linear(64, 1)         # risk management decision
        self.stop_distance_head = nn.Linear(64, 1)    # stop placement
        self.use_target_head = nn.Linear(64, 1)       # profit taking decision  
        self.target_distance_head = nn.Linear(64, 1)  # target placement
        self.exit_signal_head = nn.Linear(64, 1)      # exit timing
        self.confidence_head = nn.Linear(64, 1)       # overall confidence
        
        # Tool trust learning - dynamic weights for each subsystem
        self.tool_trust_head = nn.Linear(64, 4)       # DNA, Micro, Temporal, Immune
    
    def forward(self, market_obs, subsystem_features):
        """
        Process market data + subsystem insights to make trading decisions
        """
        batch_size = market_obs.shape[0]
        
        # Encode market observations
        market_encoded, _ = self.market_encoder(market_obs)
        market_features = market_encoded[:, -1]  # Last timestep
        
        # Process subsystem insights
        subsystem_processed = self.subsystem_processor(subsystem_features)
        
        # Learn attention over subsystem tools
        attended_tools, attention_weights = self.tool_attention(
            subsystem_processed.unsqueeze(1),
            subsystem_processed.unsqueeze(1),
            subsystem_processed.unsqueeze(1)
        )
        attended_tools = attended_tools.squeeze(1)
        
        # Combine market context with attended subsystem insights
        combined_features = torch.cat([market_features, attended_tools], dim=-1)
        decision_context = self.context_network(combined_features)
        
        # Generate all trading decisions
        outputs = {
            'action_logits': self.action_head(decision_context),
            'position_size': torch.sigmoid(self.position_size_head(decision_context)),
            'use_stop': torch.sigmoid(self.use_stop_head(decision_context)),
            'stop_distance': torch.sigmoid(self.stop_distance_head(decision_context)) * 0.05,  # 0-5%
            'use_target': torch.sigmoid(self.use_target_head(decision_context)),
            'target_distance': torch.sigmoid(self.target_distance_head(decision_context)) * 0.10,  # 0-10%
            'exit_signal': torch.sigmoid(self.exit_signal_head(decision_context)),
            'confidence': torch.sigmoid(self.confidence_head(decision_context)),
            'tool_trust': torch.softmax(self.tool_trust_head(decision_context), dim=-1),
            'attention_weights': attention_weights.squeeze()
        }
        
        return outputs

class SubsystemInterface:
    """
    Interface between black box AI and your existing subsystems
    """
    
    def __init__(self, intelligence_engine):
        self.intel = intelligence_engine
        
        # Track subsystem performance for AI learning
        self.performance_history = {
            'dna': deque(maxlen=50),
            'micro': deque(maxlen=50),
            'temporal': deque(maxlen=50),
            'immune': deque(maxlen=50)
        }
        
        # Tool usage statistics
        self.tool_usage_stats = {
            'dna': {'used': 0, 'successful': 0},
            'micro': {'used': 0, 'successful': 0},
            'temporal': {'used': 0, 'successful': 0},
            'immune': {'used': 0, 'successful': 0}
        }
    
    def extract_subsystem_features(self, prices: List[float], volumes: List[float], 
                                 timestamp: datetime) -> torch.Tensor:
        """
        Extract comprehensive features from your existing subsystems
        """
        # Run your existing intelligence engine
        intel_result = self.intel.process_market_data(prices, volumes, timestamp)
        
        subsystem_signals = intel_result.get('subsystem_signals', {})
        subsystem_scores = intel_result.get('subsystem_scores', {})
        
        features = []
        
        # DNA System features (from your DNASequencingSystem)
        dna_signal = subsystem_signals.get('dna', 0.0)
        dna_confidence = subsystem_scores.get('dna', 0.0)
        dna_patterns_found = intel_result.get('similar_patterns_count', 0)
        current_dna = intel_result.get('current_dna', '')
        dna_novelty = self._calculate_dna_novelty(current_dna)
        
        features.extend([
            dna_signal,
            dna_confidence,
            min(dna_patterns_found / 10.0, 1.0),  # Normalize pattern count
            min(len(current_dna) / 50.0, 1.0),    # Normalize sequence length
            dna_novelty
        ])
        
        # Micro Pattern features (from your MicroPatternNetwork)
        micro_signal = subsystem_signals.get('micro', 0.0)
        micro_confidence = subsystem_scores.get('micro', 0.0)
        micro_pattern_id = intel_result.get('micro_pattern_id', '')
        micro_strength = self._get_micro_pattern_strength(micro_pattern_id)
        
        features.extend([
            micro_signal,
            micro_confidence,
            1.0 if micro_pattern_id else 0.0,
            len(self.intel.micro_system.patterns) / 100.0,  # Pattern library size
            micro_strength
        ])
        
        # Temporal features (from your TemporalPatternArchaeologist)
        temporal_signal = subsystem_signals.get('temporal', 0.0)
        temporal_confidence = subsystem_scores.get('temporal', 0.0)
        temporal_strength = self.intel.temporal_system.get_temporal_strength(timestamp, "general")
        
        features.extend([
            temporal_signal,
            temporal_confidence,
            timestamp.hour / 24.0,        # Time of day
            timestamp.weekday() / 7.0,    # Day of week
            temporal_strength
        ])
        
        # Immune System features (from your MarketImmuneSystem)
        immune_signal = subsystem_signals.get('immune', 0.0)
        immune_confidence = subsystem_scores.get('immune', 0.0)
        is_dangerous = 1.0 if intel_result.get('is_dangerous_pattern', False) else 0.0
        is_beneficial = 1.0 if intel_result.get('is_beneficial_pattern', False) else 0.0
        immune_strength = self.intel.immune_system.get_immune_strength()
        
        features.extend([
            immune_signal,
            immune_confidence,
            is_dangerous,
            is_beneficial,
            immune_strength
        ])
        
        # Pad to fixed size
        while len(features) < 20:
            features.append(0.0)
            
        return torch.tensor(features[:20], dtype=torch.float32)
    
    def _calculate_dna_novelty(self, dna_sequence: str) -> float:
        """Calculate how novel this DNA sequence is"""
        if not dna_sequence:
            return 0.0
        
        existing_patterns = self.intel.dna_system.dna_patterns
        if not existing_patterns:
            return 1.0
        
        max_similarity = 0.0
        for existing_seq in existing_patterns.keys():
            similarity = self.intel.dna_system.calculate_sequence_similarity(
                dna_sequence, existing_seq
            )
            max_similarity = max(max_similarity, similarity)
        
        return 1.0 - max_similarity
    
    def _get_micro_pattern_strength(self, pattern_id: str) -> float:
        """Get strength of micro pattern"""
        if not pattern_id or pattern_id not in self.intel.micro_system.patterns:
            return 0.0
        
        pattern = self.intel.micro_system.patterns[pattern_id]
        return pattern.success_rate if pattern.sample_size >= 3 else 0.0
    
    def record_tool_performance(self, tool_name: str, prediction: float, actual_outcome: float):
        """Track performance of each subsystem tool"""
        accuracy = 1.0 if (prediction > 0) == (actual_outcome > 0) else 0.0
        self.performance_history[tool_name].append(accuracy)
        
        # Update usage stats
        self.tool_usage_stats[tool_name]['used'] += 1
        if accuracy > 0.5:
            self.tool_usage_stats[tool_name]['successful'] += 1
    
    def get_tool_performance_summary(self) -> Dict[str, float]:
        """Get recent performance summary for each tool"""
        summary = {}
        for tool, history in self.performance_history.items():
            if history:
                summary[tool] = np.mean(history)
            else:
                summary[tool] = 0.5  # Neutral if no history
        return summary

class BlackBoxSubsystemAI:
    """
    Main black box AI that learns to use your existing subsystems as tools
    """
    
    def __init__(self, intelligence_engine, obs_size: int = 15):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Black box networks
        self.policy = SubsystemToolNetwork(obs_size, 20).to(self.device)
        self.target = SubsystemToolNetwork(obs_size, 20).to(self.device)
        self.target.load_state_dict(self.policy.state_dict())
        
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=1e-4)
        
        # Interface to your existing subsystems
        self.subsystem_interface = SubsystemInterface(intelligence_engine)
        
        # Experience replay for learning
        self.replay_buffer = deque(maxlen=10000)
        
        # Learning parameters
        self.gamma = 0.99
        self.target_sync_freq = 1000
        self.step_count = 0
        
        # Start background learning
        self._start_background_learning()
        
        log.info("Black Box Subsystem AI initialized")
        log.info("AI will learn to strategically use DNA, Micro, Temporal, and Immune systems")
    
    def make_complete_decision(self, market_obs: np.ndarray, prices: List[float], 
                             volumes: List[float], current_price: float, 
                             timestamp: datetime, in_position: bool = False) -> Dict:
        """
        AI makes complete trading decision using your subsystems as tools
        """
        with torch.no_grad():
            # Prepare inputs
            market_tensor = torch.tensor(market_obs, dtype=torch.float32, device=self.device)
            market_tensor = market_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, obs_size)
            
            # Extract subsystem features
            subsystem_features = self.subsystem_interface.extract_subsystem_features(
                prices, volumes, timestamp
            ).unsqueeze(0).to(self.device)
            
            # AI processes everything
            outputs = self.policy(market_tensor, subsystem_features)
            
            # Extract decisions
            action_probs = torch.softmax(outputs['action_logits'], dim=-1).cpu().numpy()[0]
            action = np.random.choice(3, p=action_probs)
            
            position_size = float(outputs['position_size'].cpu().numpy()[0])
            use_stop_prob = float(outputs['use_stop'].cpu().numpy()[0])
            stop_distance = float(outputs['stop_distance'].cpu().numpy()[0])
            use_target_prob = float(outputs['use_target'].cpu().numpy()[0])
            target_distance = float(outputs['target_distance'].cpu().numpy()[0])
            exit_signal = float(outputs['exit_signal'].cpu().numpy()[0])
            confidence = float(outputs['confidence'].cpu().numpy()[0])
            
            # Tool trust weights learned by AI
            tool_trust = outputs['tool_trust'].cpu().numpy()[0]
            attention_weights = outputs['attention_weights'].cpu().numpy()
            
            # AI decides on risk management
            use_stop = np.random.random() < use_stop_prob
            use_target = np.random.random() < use_target_prob
            
            # Calculate actual prices
            stop_price = None
            target_price = None
            
            if use_stop and action != 0:
                if action == 1:  # Long
                    stop_price = current_price * (1 - stop_distance)
                else:  # Short
                    stop_price = current_price * (1 + stop_distance)
            
            if use_target and action != 0:
                if action == 1:  # Long
                    target_price = current_price * (1 + target_distance)
                else:  # Short
                    target_price = current_price * (1 - target_distance)
            
            # Exit decision for current position
            should_exit = in_position and (exit_signal > 0.7)
            
            # Determine primary tool AI is trusting
            tool_names = ['dna', 'micro', 'temporal', 'immune']
            primary_tool = tool_names[np.argmax(tool_trust)]
            
            return {
                'action': action,
                'position_size': position_size,
                'use_stop': use_stop,
                'stop_price': stop_price,
                'stop_distance_pct': stop_distance * 100,
                'use_target': use_target,
                'target_price': target_price,
                'target_distance_pct': target_distance * 100,
                'should_exit': should_exit,
                'confidence': confidence,
                
                # Tool usage explanation
                'tool_trust': {
                    'dna': float(tool_trust[0]),
                    'micro': float(tool_trust[1]),
                    'temporal': float(tool_trust[2]),
                    'immune': float(tool_trust[3])
                },
                'primary_tool': primary_tool,
                'attention_weights': attention_weights.tolist(),
                'reasoning': f"AI_using_{primary_tool}_tool_conf_{confidence:.3f}",
                
                # Raw data for experience storage
                'raw_market_obs': market_obs.copy(),
                'raw_subsystem_features': subsystem_features.cpu().numpy()[0]
            }
    
    def store_experience(self, decision_data: Dict, reward: float, next_market_obs: np.ndarray, 
                        next_subsystem_features: np.ndarray, done: bool):
        """
        Store experience for learning, including tool usage feedback
        """
        experience = {
            'market_obs': decision_data['raw_market_obs'],
            'subsystem_features': decision_data['raw_subsystem_features'],
            'action': decision_data['action'],
            'tool_trust': decision_data['tool_trust'],
            'reward': reward,
            'next_market_obs': next_market_obs,
            'next_subsystem_features': next_subsystem_features,
            'done': done
        }
        
        self.replay_buffer.append(experience)
        
        # Record tool performance for learning
        primary_tool = decision_data['primary_tool']
        tool_signal = decision_data['tool_trust'][primary_tool]
        self.subsystem_interface.record_tool_performance(primary_tool, tool_signal, reward)
    
    def _start_background_learning(self):
        """Start background learning thread"""
        def learning_loop():
            while True:
                if len(self.replay_buffer) < 100:
                    time.sleep(1)
                    continue
                
                self._train_step()
                time.sleep(0.1)
        
        thread = threading.Thread(target=learning_loop, daemon=True)
        thread.start()
        log.info("Background learning thread started")
    
    def _train_step(self):
        """Train the black box network"""
        if len(self.replay_buffer) < 32:
            return
        
        # Sample experiences
        batch = random.sample(self.replay_buffer, 32)
        
        # Prepare tensors
        market_obs = torch.stack([
            torch.tensor(exp['market_obs'], dtype=torch.float32)
            for exp in batch
        ]).unsqueeze(1).to(self.device)
        
        subsystem_features = torch.stack([
            torch.tensor(exp['subsystem_features'], dtype=torch.float32)
            for exp in batch
        ]).to(self.device)
        
        actions = torch.tensor([exp['action'] for exp in batch], dtype=torch.long, device=self.device)
        rewards = torch.tensor([exp['reward'] for exp in batch], dtype=torch.float32, device=self.device)
        
        next_market_obs = torch.stack([
            torch.tensor(exp['next_market_obs'], dtype=torch.float32)
            for exp in batch
        ]).unsqueeze(1).to(self.device)
        
        next_subsystem_features = torch.stack([
            torch.tensor(exp['next_subsystem_features'], dtype=torch.float32)
            for exp in batch
        ]).to(self.device)
        
        dones = torch.tensor([exp['done'] for exp in batch], dtype=torch.float32, device=self.device)
        
        # Current Q values
        current_outputs = self.policy(market_obs, subsystem_features)
        current_q = current_outputs['action_logits'].gather(1, actions.unsqueeze(1)).squeeze()
        
        # Target Q values
        with torch.no_grad():
            next_outputs = self.target(next_market_obs, next_subsystem_features)
            next_q = next_outputs['action_logits'].max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q
        
        # Train
        loss = torch.nn.functional.mse_loss(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.step_count += 1
        
        # Sync target network
        if self.step_count % self.target_sync_freq == 0:
            self.target.load_state_dict(self.policy.state_dict())
            log.info(f"Target network synced at step {self.step_count}")
    
    def get_subsystem_usage_report(self) -> str:
        """Generate report on how AI is using your subsystems"""
        
        tool_performance = self.subsystem_interface.get_tool_performance_summary()
        usage_stats = self.subsystem_interface.tool_usage_stats
        
        report = f"""
=== BLACK BOX SUBSYSTEM USAGE REPORT ===

AI Learning Progress:
- Training Steps: {self.step_count}
- Experience Buffer: {len(self.replay_buffer)} samples

Tool Performance (Recent):
- DNA System: {tool_performance['dna']:.2%} accuracy ({usage_stats['dna']['successful']}/{usage_stats['dna']['used']} successful)
- Micro Patterns: {tool_performance['micro']:.2%} accuracy ({usage_stats['micro']['successful']}/{usage_stats['micro']['used']} successful)
- Temporal Analysis: {tool_performance['temporal']:.2%} accuracy ({usage_stats['temporal']['successful']}/{usage_stats['temporal']['used']} successful)
- Immune System: {tool_performance['immune']:.2%} accuracy ({usage_stats['immune']['successful']}/{usage_stats['immune']['used']} successful)

Your Existing Subsystems Status:
- DNA Patterns: {len(self.subsystem_interface.intel.dna_system.dna_patterns)}
- Micro Patterns: {len(self.subsystem_interface.intel.micro_system.patterns)}
- Temporal Patterns: {len(self.subsystem_interface.intel.temporal_system.temporal_patterns)}
- Immune Strength: {self.subsystem_interface.intel.immune_system.get_immune_strength():.2%}

AI is learning to dynamically choose the best tools for each market situation!
        """
        
        return report
    
    def save_model(self, filepath: str):
        """Save the trained model"""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'target_state_dict': self.target.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'step_count': self.step_count
        }, filepath)
        log.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.target.load_state_dict(checkpoint['target_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.step_count = checkpoint['step_count']
        log.info(f"Model loaded from {filepath}")