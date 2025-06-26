import numpy as np
import logging
from collections import defaultdict, deque
from datetime import datetime
from typing import Dict, List, Optional

from .dna_subsystem import DNASubsystem
from .temporal_subsystem import FFTTemporalSubsystem
from .immune_subsystem import EvolvingImmuneSystem

logger = logging.getLogger(__name__)

class EnhancedIntelligenceOrchestrator:
    def __init__(self):
        self.dna_subsystem = DNASubsystem()
        self.temporal_subsystem = FFTTemporalSubsystem()
        self.immune_subsystem = EvolvingImmuneSystem()
        
        self.subsystem_votes = deque(maxlen=200)
        self.consensus_history = deque(maxlen=100)
        self.performance_attribution = defaultdict(lambda: deque(maxlen=50))
        
        self.tool_lifecycle = {
            'dna': {'birth_time': datetime.now(), 'performance_history': deque(maxlen=100)},
            'temporal': {'birth_time': datetime.now(), 'performance_history': deque(maxlen=100)},
            'immune': {'birth_time': datetime.now(), 'performance_history': deque(maxlen=100)},
            'microstructure': {'birth_time': datetime.now(), 'performance_history': deque(maxlen=100)}
        }
        
        self.hybrid_tools = {}
        self.tool_breeding_frequency = 500
        self.decision_count = 0
        self.bootstrap_complete = False

    def process_market_data(self, prices: List[float], volumes: List[float],
                           market_features: Dict, timestamps: Optional[List[float]] = None) -> Dict[str, float]:
        
        try:
            self.decision_count += 1
            
            volatility = market_features.get('volatility', 0)
            momentum = market_features.get('price_momentum', 0)
            
            try:
                dna_sequence = self.dna_subsystem.encode_market_state(prices[-20:], volumes[-20:], volatility, momentum)
                dna_signal = self.dna_subsystem.analyze_sequence(dna_sequence)
                
                if np.isnan(dna_signal) or np.isinf(dna_signal):
                    logger.warning("Invalid DNA signal detected, using baseline")
                    dna_signal = 0.05
                elif dna_signal == 0.0 and len(self.dna_subsystem.sequences) > 0:
                    dna_signal = 0.05
                    
                logger.debug(f"Final DNA signal: {dna_signal:.4f}")
                    
            except Exception as e:
                logger.error(f"DNA analysis failed: {e}")
                dna_sequence = ""
                dna_signal = 0.05
            
            try:
                temporal_signal = self.temporal_subsystem.analyze_cycles(prices, timestamps)
                
                if np.isnan(temporal_signal) or np.isinf(temporal_signal):
                    logger.warning("Invalid temporal signal detected, using baseline")
                    temporal_signal = 0.02
                elif temporal_signal == 0.0:
                    temporal_signal = 0.02
                    
                logger.debug(f"Final temporal signal: {temporal_signal:.4f}")
                    
            except Exception as e:
                logger.error(f"Temporal analysis failed: {e}")
                temporal_signal = 0.02
            
            try:
                immune_signal = self.immune_subsystem.detect_threats(market_features)
                
                if np.isnan(immune_signal) or np.isinf(immune_signal):
                    logger.warning("Invalid immune signal detected, using baseline")
                    immune_signal = -0.01
                elif immune_signal == 0.0 and len(self.immune_subsystem.antibodies) == 0:
                    immune_signal = -0.01
                    
                logger.debug(f"Final immune signal: {immune_signal:.4f}")
                    
            except Exception as e:
                logger.error(f"Immune analysis failed: {e}")
                immune_signal = -0.01
            
            votes = {
                'dna': dna_signal,
                'temporal': temporal_signal,
                'immune': immune_signal
            }
            
            for tool, signal in votes.items():
                if not (np.isnan(signal) or np.isinf(signal)):
                    self.tool_lifecycle[tool]['performance_history'].append(signal)
            
            try:
                consensus_strength = self._calculate_enhanced_consensus(votes)
                if np.isnan(consensus_strength) or np.isinf(consensus_strength):
                    consensus_strength = 0.5
            except Exception as e:
                logger.warning(f"Consensus calculation failed: {e}")
                consensus_strength = 0.5
            
            try:
                active_weights = self._calculate_dynamic_weights(market_features, votes)
            except Exception as e:
                logger.warning(f"Weight calculation failed: {e}")
                active_weights = {'dna': 0.4, 'temporal': 0.4, 'immune': 0.2}
            
            try:
                overall_signal = sum(votes[tool] * active_weights.get(tool, 0.0) for tool in votes.keys())
                
                logger.debug(f"Raw overall signal: {overall_signal:.4f} from votes: {votes}")
                
                if np.isnan(overall_signal) or np.isinf(overall_signal):
                    logger.warning("Invalid overall signal detected, using baseline")
                    overall_signal = 0.02
                
                if overall_signal == 0.0:
                    baseline_signal = (votes['dna'] * 0.4 + votes['temporal'] * 0.4 + abs(votes['immune']) * 0.2)
                    if baseline_signal > 0:
                        overall_signal = baseline_signal
                        logger.debug(f"Applied baseline overall signal: {overall_signal:.4f}")
                
                if consensus_strength > 0.8:
                    overall_signal *= 1.4
                elif consensus_strength < 0.3:
                    overall_signal *= 0.6
                
                logger.debug(f"Final overall signal: {overall_signal:.4f}")
                    
            except Exception as e:
                logger.error(f"Overall signal calculation failed: {e}")
                overall_signal = 0.02
            
            if self.decision_count % self.tool_breeding_frequency == 0:
                try:
                    self._attempt_tool_breeding(votes, market_features)
                except Exception as e:
                    logger.warning(f"Tool breeding failed: {e}")
            
            try:
                self.subsystem_votes.append({
                    'votes': votes.copy(),
                    'consensus': consensus_strength,
                    'weights': active_weights.copy(),
                    'timestamp': datetime.now()
                })
                self.consensus_history.append(consensus_strength)
            except Exception as e:
                logger.warning(f"Failed to store voting data: {e}")
            
            return {
                'dna_signal': dna_signal,
                'temporal_signal': temporal_signal,
                'immune_signal': immune_signal,
                'overall_signal': overall_signal,
                'consensus_strength': consensus_strength,
                'active_weights': active_weights,
                'current_patterns': {
                    'dna_sequence': dna_sequence if 'dna_sequence' in locals() else "",
                    'dominant_cycles': len(self.temporal_subsystem.dominant_cycles),
                    'active_antibodies': len(self.immune_subsystem.antibodies),
                    'hybrid_tools': len(self.hybrid_tools)
                }
            }
            
        except Exception as e:
            logger.error(f"Critical error in orchestrator processing: {e}")
            return {
                'dna_signal': 0.0,
                'temporal_signal': 0.0,
                'immune_signal': 0.0,
                'overall_signal': 0.0,
                'consensus_strength': 0.5,
                'active_weights': {'dna': 0.4, 'temporal': 0.4, 'immune': 0.2},
                'current_patterns': {
                    'dna_sequence': "",
                    'dominant_cycles': 0,
                    'active_antibodies': 0,
                    'hybrid_tools': 0
                }
            }

    def _calculate_enhanced_consensus(self, votes: Dict[str, float]) -> float:
        signals = list(votes.values())
        
        if not signals:
            return 0.0
        
        positive_signals = sum(1 for s in signals if s > 0.1)
        negative_signals = sum(1 for s in signals if s < -0.1)
        neutral_signals = sum(1 for s in signals if abs(s) <= 0.1)
        
        total_signals = len(signals)
        directional_consensus = max(positive_signals, negative_signals, neutral_signals) / total_signals
        
        signal_magnitudes = [abs(s) for s in signals]
        magnitude_std = np.std(signal_magnitudes) if len(signal_magnitudes) > 1 else 0
        magnitude_consensus = 1.0 / (1.0 + magnitude_std)
        
        return (directional_consensus * 0.7 + magnitude_consensus * 0.3)

    def _calculate_dynamic_weights(self, market_features: Dict, votes: Dict[str, float]) -> Dict[str, float]:
        weights = {'dna': 0.4, 'temporal': 0.4, 'immune': 0.2}
        
        volatility = market_features.get('volatility', 0.02)
        if volatility > 0.05:
            weights['immune'] += 0.2
            weights['dna'] -= 0.1
            weights['temporal'] -= 0.1
        elif volatility < 0.01:
            weights['temporal'] += 0.2
            weights['immune'] -= 0.1
            weights['dna'] -= 0.1
        
        for tool in weights.keys():
            recent_performance = list(self.tool_lifecycle[tool]['performance_history'])[-10:]
            if len(recent_performance) >= 5:
                avg_performance = np.mean([abs(p) for p in recent_performance])
                if avg_performance > 0.3:
                    weights[tool] *= 1.2
                elif avg_performance < 0.1:
                    weights[tool] *= 0.8
        
        total_weight = sum(weights.values())
        return {tool: weight / total_weight for tool, weight in weights.items()}

    def _attempt_tool_breeding(self, votes: Dict[str, float], market_features: Dict):
        tool_pairs = [('dna', 'temporal'), ('dna', 'immune'), ('temporal', 'immune')]
        
        for tool1, tool2 in tool_pairs:
            signal1, signal2 = votes[tool1], votes[tool2]
            
            if abs(signal1) > 0.3 and abs(signal2) > 0.3 and np.sign(signal1) == np.sign(signal2):
                hybrid_name = f"{tool1}_{tool2}_hybrid"
                
                if hybrid_name not in self.hybrid_tools:
                    self.hybrid_tools[hybrid_name] = {
                        'parents': (tool1, tool2),
                        'birth_time': datetime.now(),
                        'performance_history': deque(maxlen=50),
                        'activation_conditions': self._create_activation_conditions(market_features),
                        'combination_strategy': 'weighted_average'
                    }
                    
                    logger.info(f"Created hybrid tool: {hybrid_name}")

    def _create_activation_conditions(self, market_features: Dict) -> Dict:
        return {
            'min_volatility': market_features.get('volatility', 0.02) * 0.8,
            'max_volatility': market_features.get('volatility', 0.02) * 1.2,
            'momentum_threshold': abs(market_features.get('price_momentum', 0)) * 0.5
        }

    def learn_from_outcome(self, outcome: float, context: Dict):
        if not isinstance(context, dict):
            logger.error(f"Context is not a dictionary: type={type(context)}, content={str(context)[:200]}")
            return
        
        dna_sequence = context.get('dna_sequence', '')
        if not isinstance(dna_sequence, str):
            logger.error(f"DNA sequence is not a string: type={type(dna_sequence)}, content={str(dna_sequence)[:100]}")
            dna_sequence = str(dna_sequence) if dna_sequence else ''
        
        cycles_info = context.get('cycles_info', [])
        if not isinstance(cycles_info, list):
            logger.error(f"Cycles info is not a list: type={type(cycles_info)}, content={str(cycles_info)[:100]}")
            cycles_info = []
        
        market_state = context.get('market_state', {})
        if not isinstance(market_state, dict):
            logger.error(f"Market state is not a dict: type={type(market_state)}, content={str(market_state)[:100]}")
            market_state = {}
        
        microstructure_signal = context.get('microstructure_signal', 0.0)
        if not isinstance(microstructure_signal, (int, float)):
            logger.error(f"Microstructure signal is not numeric: type={type(microstructure_signal)}, content={str(microstructure_signal)[:100]}")
            microstructure_signal = 0.0
        
        if dna_sequence:
            try:
                self.dna_subsystem.learn_from_outcome(dna_sequence, outcome)
            except Exception as e:
                logger.error(f"Error in DNA subsystem learning: {e}")
        
        recent_cycles = []
        try:
            if len(self.temporal_subsystem.dominant_cycles) > 0:
                recent_cycles = list(self.temporal_subsystem.dominant_cycles)[-1] if self.temporal_subsystem.dominant_cycles else []
        except Exception as e:
            logger.error(f"Error extracting recent cycles: {e}")
            recent_cycles = []
        
        if recent_cycles or cycles_info:
            try:
                cycles_to_learn = cycles_info if cycles_info else recent_cycles
                self.temporal_subsystem.learn_from_outcome(cycles_to_learn, outcome)
            except Exception as e:
                logger.error(f"Error in temporal subsystem learning: {e}")
        
        if market_state:
            try:
                self.immune_subsystem.learn_threat(market_state, outcome)
            except Exception as e:
                logger.error(f"Error in immune subsystem learning: {e}")
        
        if microstructure_signal != 0.0:
            self.tool_lifecycle['microstructure']['performance_history'].append(microstructure_signal)
        
        try:
            if self.subsystem_votes:
                recent_vote = self.subsystem_votes[-1]
                
                if not isinstance(recent_vote, dict):
                    logger.error(f"Recent vote is not a dict: type={type(recent_vote)}, content={str(recent_vote)[:100]}")
                else:
                    votes = recent_vote.get('votes', {})
                    weights = recent_vote.get('weights', {})
                    
                    if not isinstance(votes, dict):
                        logger.error(f"Votes is not a dict: type={type(votes)}, content={str(votes)[:100]}")
                        votes = {}
                    
                    if not isinstance(weights, dict):
                        logger.error(f"Weights is not a dict: type={type(weights)}, content={str(weights)[:100]}")
                        weights = {}
                    
                    for tool, signal in votes.items():
                        try:
                            if tool in weights:
                                attribution_score = outcome * signal * weights[tool]
                                self.performance_attribution[tool].append(attribution_score)
                        except Exception as e:
                            logger.error(f"Error in performance attribution for tool {tool}: {e}")
                
                if microstructure_signal != 0.0:
                    try:
                        micro_attribution = outcome * microstructure_signal * 0.2
                        self.performance_attribution['microstructure'].append(micro_attribution)
                    except Exception as e:
                        logger.error(f"Error in microstructure attribution: {e}")
        except Exception as e:
            logger.error(f"Error in performance attribution section: {e}")
        
        for hybrid_name, hybrid_data in self.hybrid_tools.items():
            hybrid_data['performance_history'].append(outcome * 0.5)
        
        if len(self.subsystem_votes) % 100 == 0:
            try:
                self._evolve_tools()
            except Exception as e:
                logger.warning(f"Tool evolution failed: {e}")

    def _evolve_tools(self):
        self.immune_subsystem.evolve_antibodies()
        
        current_time = datetime.now()
        tools_to_remove = []
        
        for hybrid_name, hybrid_data in self.hybrid_tools.items():
            if len(hybrid_data['performance_history']) >= 20:
                avg_performance = np.mean(list(hybrid_data['performance_history']))
                if avg_performance < -0.2:
                    tools_to_remove.append(hybrid_name)
        
        for tool_name in tools_to_remove:
            del self.hybrid_tools[tool_name]
            logger.info(f"Removed underperforming hybrid tool: {tool_name}")

    def get_comprehensive_stats(self) -> Dict:
        dna_stats = self.dna_subsystem.get_evolution_stats()
        immune_stats = self.immune_subsystem.get_immune_stats()
        
        attribution_stats = {}
        for tool, scores in self.performance_attribution.items():
            if scores:
                attribution_stats[tool] = {
                    'avg_attribution': np.mean(list(scores)),
                    'attribution_volatility': np.std(list(scores)),
                    'positive_attributions': sum(1 for s in scores if s > 0),
                    'total_attributions': len(scores)
                }
        
        return {
            'dna_evolution': dna_stats,
            'immune_system': immune_stats,
            'temporal_cycles': len(self.temporal_subsystem.cycle_memory),
            'recent_consensus': np.mean(list(self.consensus_history)) if self.consensus_history else 0.0,
            'performance_attribution': attribution_stats,
            'hybrid_tools': {
                'count': len(self.hybrid_tools),
                'tools': list(self.hybrid_tools.keys())
            },
            'decision_count': self.decision_count
        }