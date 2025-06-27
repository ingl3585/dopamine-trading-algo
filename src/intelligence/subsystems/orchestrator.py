import numpy as np
import logging
from collections import defaultdict, deque
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import random

from .dna_subsystem import DNASubsystem
from .temporal_subsystem import FFTTemporalSubsystem
from .immune_subsystem import EvolvingImmuneSystem

logger = logging.getLogger(__name__)

class EnhancedIntelligenceOrchestrator:
    def __init__(self):
        self.dna_subsystem = DNASubsystem()
        self.temporal_subsystem = FFTTemporalSubsystem()
        self.immune_subsystem = EvolvingImmuneSystem()
        
        # Swarm Intelligence Components
        self.subsystem_votes = deque(maxlen=200)
        self.consensus_history = deque(maxlen=100)
        self.performance_attribution = defaultdict(lambda: deque(maxlen=50))
        self.debate_records = deque(maxlen=500)
        self.disagreement_weights = {'high': 1.5, 'medium': 1.0, 'low': 0.7}
        
        # Tool Evolution & Lifecycle Management
        self.tool_lifecycle = {
            'dna': {'birth_time': datetime.now(), 'performance_history': deque(maxlen=100), 'generation': 1, 'fitness': 0.5},
            'temporal': {'birth_time': datetime.now(), 'performance_history': deque(maxlen=100), 'generation': 1, 'fitness': 0.5},
            'immune': {'birth_time': datetime.now(), 'performance_history': deque(maxlen=100), 'generation': 1, 'fitness': 0.5},
            'microstructure': {'birth_time': datetime.now(), 'performance_history': deque(maxlen=100), 'generation': 1, 'fitness': 0.5}
        }
        
        # Tool Breeding & Hybrid Creation
        self.hybrid_tools = {}
        self.breeding_pool = deque(maxlen=10)
        self.tool_breeding_frequency = 500
        self.breeding_success_rate = deque(maxlen=20)
        self.next_hybrid_id = 1
        
        # Consensus Building
        self.consensus_threshold = 0.6
        self.vote_weights = {'dna': 0.25, 'temporal': 0.25, 'immune': 0.25, 'microstructure': 0.25}
        self.dynamic_weight_adjustment = True
        
        self.decision_count = 0
        self.bootstrap_complete = False

    def process_market_data(self, prices: List[float], volumes: List[float],
                           market_features: Dict, timestamps: Optional[List[float]] = None) -> Dict[str, float]:
        
        try:
            self.decision_count += 1
            
            # Collect individual subsystem votes
            subsystem_signals = self._collect_subsystem_votes(prices, volumes, market_features, timestamps)
            
            # Swarm intelligence consensus building
            consensus_result = self._build_swarm_consensus(subsystem_signals, market_features)
            
            # Tool evolution check
            if self.decision_count % self.tool_breeding_frequency == 0:
                self._evolve_tools()
            
            # Performance attribution tracking
            self._track_performance_attribution(subsystem_signals, consensus_result)
            
            return consensus_result
            
        except Exception as e:
            logger.error(f"Error in orchestrator market data processing: {e}")
            return {'overall_signal': 0.0, 'confidence': 0.1, 'consensus_strength': 0.0}
    
    def _collect_subsystem_votes(self, prices: List[float], volumes: List[float], 
                                market_features: Dict, timestamps: Optional[List[float]]) -> Dict[str, Dict]:
        
        volatility = market_features.get('volatility', 0)
        momentum = market_features.get('price_momentum', 0)
        
        votes = {}
        
        try:
            # DNA Subsystem Vote
            dna_sequence = self.dna_subsystem.encode_market_state(prices[-20:], volumes[-20:], volatility, momentum)
            dna_signal = self.dna_subsystem.analyze_sequence(dna_sequence)
            
            if np.isnan(dna_signal) or np.isinf(dna_signal):
                dna_signal = 0.05
                
            votes['dna'] = {
                'signal': dna_signal,
                'confidence': min(1.0, abs(dna_signal) * 2.0),
                'reasoning': f'DNA sequence: {dna_sequence[:20]}...' if dna_sequence else 'No sequence',
                'strength': abs(dna_signal),
                'context': {'volatility': volatility, 'momentum': momentum}
            }
            
        except Exception as e:
            logger.warning(f"DNA subsystem error: {e}")
            votes['dna'] = {'signal': 0.05, 'confidence': 0.1, 'reasoning': 'Error fallback', 'strength': 0.1, 'context': {}}

        try:
            # Temporal Subsystem Vote
            temporal_signal = self.temporal_subsystem.analyze_cycles(prices, timestamps)
            
            if np.isnan(temporal_signal) or np.isinf(temporal_signal):
                temporal_signal = 0.02
                
            votes['temporal'] = {
                'signal': temporal_signal,
                'confidence': min(1.0, abs(temporal_signal) * 3.0),
                'reasoning': f'Cycles detected: {len(self.temporal_subsystem.dominant_cycles)}',
                'strength': abs(temporal_signal),
                'context': {'price_length': len(prices), 'cycle_count': len(self.temporal_subsystem.dominant_cycles)}
            }
            
        except Exception as e:
            logger.warning(f"Temporal subsystem error: {e}")
            votes['temporal'] = {'signal': 0.02, 'confidence': 0.1, 'reasoning': 'Error fallback', 'strength': 0.1, 'context': {}}

        try:
            # Immune Subsystem Vote
            immune_signal = self.immune_subsystem.detect_threats(market_features)
            
            if np.isnan(immune_signal) or np.isinf(immune_signal):
                immune_signal = -0.01
                
            votes['immune'] = {
                'signal': immune_signal,
                'confidence': min(1.0, abs(immune_signal) * 2.5),
                'reasoning': f'Antibodies: {len(self.immune_subsystem.antibodies)}, Threats detected',
                'strength': abs(immune_signal),
                'context': {'antibody_count': len(self.immune_subsystem.antibodies), 'threat_level': immune_signal}
            }
            
        except Exception as e:
            logger.warning(f"Immune subsystem error: {e}")
            votes['immune'] = {'signal': -0.01, 'confidence': 0.1, 'reasoning': 'Error fallback', 'strength': 0.1, 'context': {}}

        # Add microstructure vote placeholder for future integration
        votes['microstructure'] = {
            'signal': 0.0,
            'confidence': 0.1,
            'reasoning': 'Microstructure analysis placeholder',
            'strength': 0.0,
            'context': {}
        }
        
        return votes

    def _build_swarm_consensus(self, votes: Dict[str, Dict], market_features: Dict) -> Dict[str, float]:
        
        try:
            # Extract signals and confidences
            signals = {tool: vote['signal'] for tool, vote in votes.items()}
            confidences = {tool: vote['confidence'] for tool, vote in votes.items()}
            strengths = {tool: vote['strength'] for tool, vote in votes.items()}
            
            # Dynamic weight adjustment based on recent performance
            if self.dynamic_weight_adjustment:
                self._adjust_vote_weights(signals)
            
            # Calculate disagreement level
            disagreement_level = self._calculate_disagreement(signals)
            
            # Build consensus with disagreement weighting
            consensus_signal = self._weighted_consensus(signals, disagreement_level)
            
            # Calculate overall confidence
            avg_confidence = np.mean(list(confidences.values()))
            consensus_confidence = avg_confidence * (1.0 - disagreement_level * 0.5)
            
            # Calculate consensus strength
            consensus_strength = self._calculate_consensus_strength(signals, strengths)
            
            # Record debate for learning
            debate_record = {
                'timestamp': datetime.now(),
                'votes': votes,
                'disagreement': disagreement_level,
                'consensus_signal': consensus_signal,
                'market_context': market_features
            }
            self.debate_records.append(debate_record)
            
            # Store consensus history
            consensus_data = {
                'signal': consensus_signal,
                'confidence': consensus_confidence,
                'strength': consensus_strength,
                'disagreement': disagreement_level
            }
            self.consensus_history.append(consensus_data)
            
            logger.debug(f"Swarm consensus: signal={consensus_signal:.4f}, confidence={consensus_confidence:.3f}, "
                        f"disagreement={disagreement_level:.3f}, strength={consensus_strength:.3f}")
            
            return {
                'overall_signal': consensus_signal,
                'confidence': consensus_confidence,
                'consensus_strength': consensus_strength,
                'disagreement_level': disagreement_level,
                'dna_signal': signals['dna'],
                'temporal_signal': signals['temporal'],
                'immune_signal': signals['immune'],
                'microstructure_signal': signals['microstructure']
            }
            
        except Exception as e:
            logger.error(f"Error building swarm consensus: {e}")
            return {'overall_signal': 0.0, 'confidence': 0.1, 'consensus_strength': 0.0}

    def _calculate_disagreement(self, signals: Dict[str, float]) -> float:
        
        signal_values = list(signals.values())
        if len(signal_values) < 2:
            return 0.0
        
        # Calculate coefficient of variation (std/mean) as disagreement measure
        mean_signal = np.mean(signal_values)
        std_signal = np.std(signal_values)
        
        if abs(mean_signal) < 1e-8:
            return std_signal  # High disagreement if mean near zero but variance exists
        
        disagreement = min(1.0, std_signal / abs(mean_signal))
        
        # Classify disagreement level
        if disagreement > 0.7:
            disagreement_category = 'high'
        elif disagreement > 0.4:
            disagreement_category = 'medium'
        else:
            disagreement_category = 'low'
        
        return disagreement * self.disagreement_weights[disagreement_category]

    def _weighted_consensus(self, signals: Dict[str, float], disagreement: float) -> float:
        
        # Apply dynamic weights
        weighted_sum = 0.0
        total_weight = 0.0
        
        for tool, signal in signals.items():
            weight = self.vote_weights.get(tool, 0.25)
            
            # Boost weight for high-performing tools
            recent_performance = np.mean(list(self.tool_lifecycle[tool]['performance_history']))
            if recent_performance > 0.1:
                weight *= 1.2
            elif recent_performance < -0.1:
                weight *= 0.8
            
            weighted_sum += signal * weight
            total_weight += weight
        
        if total_weight == 0:
            return 0.0
        
        consensus = weighted_sum / total_weight
        
        # Apply disagreement penalty/bonus
        if disagreement > 0.5:
            consensus *= 0.8  # Reduce signal strength during high disagreement
        elif disagreement < 0.2:
            consensus *= 1.1  # Boost signal strength during strong consensus
        
        return float(np.clip(consensus, -1.0, 1.0))

    def _calculate_consensus_strength(self, signals: Dict[str, float], strengths: Dict[str, float]) -> float:
        
        # Combine signal alignment with individual strengths
        signal_values = list(signals.values())
        strength_values = list(strengths.values())
        
        # Directional alignment
        positive_signals = sum(1 for s in signal_values if s > 0.05)
        negative_signals = sum(1 for s in signal_values if s < -0.05)
        neutral_signals = len(signal_values) - positive_signals - negative_signals
        
        alignment = max(positive_signals, negative_signals, neutral_signals) / len(signal_values)
        
        # Average strength
        avg_strength = np.mean(strength_values)
        
        # Combined consensus strength
        consensus_strength = alignment * avg_strength
        
        return float(np.clip(consensus_strength, 0.0, 1.0))

    def _adjust_vote_weights(self, current_signals: Dict[str, float]):
        
        # Adjust weights based on recent performance attribution
        total_adjustment = 0.0
        
        for tool in current_signals.keys():
            if tool in self.performance_attribution:
                recent_outcomes = list(self.performance_attribution[tool])[-10:]
                if len(recent_outcomes) > 5:
                    avg_outcome = np.mean(recent_outcomes)
                    
                    # Positive outcomes increase weight, negative decrease
                    adjustment = avg_outcome * 0.1  # Max 10% adjustment
                    new_weight = self.vote_weights[tool] + adjustment
                    self.vote_weights[tool] = np.clip(new_weight, 0.05, 0.7)  # Keep reasonable bounds
                    total_adjustment += adjustment
        
        # Normalize weights to sum to 1.0
        weight_sum = sum(self.vote_weights.values())
        if weight_sum > 0:
            for tool in self.vote_weights:
                self.vote_weights[tool] /= weight_sum

    def _track_performance_attribution(self, signals: Dict[str, Dict], consensus_result: Dict[str, float]):
        
        # Store current vote for future outcome learning
        vote_record = {
            'timestamp': datetime.now(),
            'signals': {tool: vote['signal'] for tool, vote in signals.items()},
            'consensus': consensus_result['overall_signal'],
            'confidence': consensus_result['confidence']
        }
        self.subsystem_votes.append(vote_record)

    def _evolve_tools(self):
        
        try:
            logger.info("Starting tool evolution cycle...")
            
            # Evaluate current tool fitness
            self._evaluate_tool_fitness()
            
            # Attempt tool breeding
            self._breed_successful_tools()
            
            # Lifecycle management
            self._manage_tool_lifecycle()
            
            logger.info(f"Tool evolution complete. Active tools: {len(self.tool_lifecycle)}, "
                       f"Hybrid tools: {len(self.hybrid_tools)}")
            
        except Exception as e:
            logger.error(f"Error in tool evolution: {e}")

    def _evaluate_tool_fitness(self):
        
        for tool_name, tool_data in self.tool_lifecycle.items():
            performance_history = list(tool_data['performance_history'])
            
            if len(performance_history) > 10:
                # Calculate fitness based on recent performance
                recent_performance = performance_history[-20:]
                avg_performance = np.mean(recent_performance)
                consistency = 1.0 - np.std(recent_performance)
                
                # Fitness combines performance and consistency
                fitness = (avg_performance + 1.0) * 0.5 + consistency * 0.3
                tool_data['fitness'] = np.clip(fitness, 0.0, 1.0)
                
                logger.debug(f"Tool {tool_name} fitness: {fitness:.3f} (perf: {avg_performance:.3f}, consistency: {consistency:.3f})")

    def _breed_successful_tools(self):
        
        # Find top performing tools for breeding
        tool_fitness = [(name, data['fitness']) for name, data in self.tool_lifecycle.items()]
        tool_fitness.sort(key=lambda x: x[1], reverse=True)
        
        if len(tool_fitness) >= 2:
            # Breed top 2 tools
            parent1, fitness1 = tool_fitness[0]
            parent2, fitness2 = tool_fitness[1]
            
            if fitness1 > 0.6 and fitness2 > 0.6:  # Only breed high-fitness tools
                hybrid_name = f"hybrid_{self.next_hybrid_id}"
                self.next_hybrid_id += 1
                
                # Create hybrid tool by combining characteristics
                hybrid_data = {
                    'birth_time': datetime.now(),
                    'performance_history': deque(maxlen=100),
                    'generation': max(self.tool_lifecycle[parent1]['generation'], 
                                    self.tool_lifecycle[parent2]['generation']) + 1,
                    'fitness': (fitness1 + fitness2) / 2,
                    'parents': [parent1, parent2],
                    'hybrid_weight': 0.5  # Equal blend initially
                }
                
                self.hybrid_tools[hybrid_name] = hybrid_data
                self.breeding_pool.append((parent1, parent2, hybrid_name))
                
                # Track breeding success
                self.breeding_success_rate.append(1.0)
                
                logger.info(f"Bred new hybrid tool: {hybrid_name} from {parent1} + {parent2}")
            else:
                self.breeding_success_rate.append(0.0)

    def _manage_tool_lifecycle(self):
        
        # Age-based lifecycle management
        current_time = datetime.now()
        tools_to_remove = []
        
        for tool_name, tool_data in list(self.hybrid_tools.items()):
            age_days = (current_time - tool_data['birth_time']).days
            
            # Remove low-performing old hybrids
            if age_days > 7 and tool_data['fitness'] < 0.3:
                tools_to_remove.append(tool_name)
            elif age_days > 30:  # Maximum age
                tools_to_remove.append(tool_name)
        
        for tool_name in tools_to_remove:
            del self.hybrid_tools[tool_name]
            logger.info(f"Removed aged/low-performing hybrid tool: {tool_name}")

    def learn_from_outcome(self, outcome: float, context: Optional[Dict] = None):
        
        try:
            # Attribute outcome to recent votes
            if len(self.subsystem_votes) > 0:
                recent_vote = self.subsystem_votes[-1]
                
                # Update performance attribution for each subsystem
                for tool, signal in recent_vote['signals'].items():
                    if tool in self.performance_attribution:
                        # Weight outcome by signal contribution
                        contribution = abs(signal) if abs(signal) > 0.01 else 0.01
                        attributed_outcome = outcome * contribution
                        
                        self.performance_attribution[tool].append(attributed_outcome)
                        
                        # Update tool lifecycle performance
                        if tool in self.tool_lifecycle:
                            self.tool_lifecycle[tool]['performance_history'].append(attributed_outcome)
            
            # Learn in individual subsystems
            if context:
                # DNA learning
                if 'dna_sequence' in context and context['dna_sequence']:
                    self.dna_subsystem.learn_from_outcome(context['dna_sequence'], outcome)
                
                # Temporal learning
                if 'cycles_info' in context and context['cycles_info']:
                    self.temporal_subsystem.learn_from_outcome(context['cycles_info'], outcome)
                
                # Immune learning
                if 'market_state' in context:
                    self.immune_subsystem.learn_threat(context['market_state'], outcome)
            
            logger.debug(f"Orchestrator learned from outcome: {outcome:.4f}")
            
        except Exception as e:
            logger.error(f"Error in orchestrator learning: {e}")

    def get_comprehensive_stats(self) -> Dict:
        
        try:
            # DNA subsystem stats
            dna_stats = {
                'total_sequences': len(self.dna_subsystem.sequences),
                'elite_sequences': len(self.dna_subsystem.elite_sequences),
                'generation_count': self.dna_subsystem.generation_count,
                'learning_events': self.dna_subsystem.total_learning_events
            }
            
            # Temporal subsystem stats
            temporal_cycles = len(self.temporal_subsystem.dominant_cycles)
            
            # Immune subsystem stats
            immune_stats = {
                'total_antibodies': len(self.immune_subsystem.antibodies),
                'memory_cells': len(self.immune_subsystem.t_cell_memory),
                'antibody_generations': self.immune_subsystem.antibody_generations,
                'learning_events': self.immune_subsystem.total_learning_events
            }
            
            # Consensus and performance stats
            consensus_stats = {
                'total_decisions': self.decision_count,
                'consensus_history_length': len(self.consensus_history),
                'debate_records_length': len(self.debate_records),
                'hybrid_tools_count': len(self.hybrid_tools),
                'current_vote_weights': dict(self.vote_weights)
            }
            
            # Tool performance attribution
            attribution_summary = {}
            for tool, outcomes in self.performance_attribution.items():
                if len(outcomes) > 0:
                    attribution_summary[tool] = {
                        'avg_outcome': float(np.mean(list(outcomes))),
                        'outcome_count': len(outcomes),
                        'recent_avg': float(np.mean(list(outcomes)[-10:])) if len(outcomes) >= 10 else 0.0
                    }
            
            return {
                'dna_evolution': dna_stats,
                'temporal_cycles': temporal_cycles,
                'immune_system': immune_stats,
                'consensus_building': consensus_stats,
                'performance_attribution': attribution_summary,
                'tool_fitness': {name: data['fitness'] for name, data in self.tool_lifecycle.items()}
            }
            
        except Exception as e:
            logger.error(f"Error getting comprehensive stats: {e}")
            return {}

    def close_progress_bars(self, microstructure_engine=None):
        
        # Close any progress tracking for bootstrap completion
        try:
            if hasattr(self.dna_subsystem, '_progress_tracker'):
                delattr(self.dna_subsystem, '_progress_tracker')
            if hasattr(self.temporal_subsystem, '_progress_tracker'):
                delattr(self.temporal_subsystem, '_progress_tracker')
            if hasattr(self.immune_subsystem, '_progress_tracker'):
                delattr(self.immune_subsystem, '_progress_tracker')
                
            logger.debug("Progress trackers closed for live trading")
            
        except Exception as e:
            logger.warning(f"Error closing progress bars: {e}")