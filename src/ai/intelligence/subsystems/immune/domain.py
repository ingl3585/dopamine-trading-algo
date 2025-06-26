"""
Evolving Immune System Domain - Adaptive threat detection per prompt.txt
"""

import numpy as np
import logging
from collections import deque
from datetime import datetime
from typing import Dict

from .antibodies import AntibodyEvolution
from .threats import ThreatAnalyzer

logger = logging.getLogger(__name__)

class EvolvingImmuneSystem:
    """
    Evolving Immune System per prompt.txt:
    - Adaptive antibodies that evolve to recognize new threat patterns
    - Immune memory T-cells for quick recognition of returning threats
    - Autoimmune prevention to avoid rejecting profitable but unusual patterns
    - Threat evolution tracking that adapts to changing market dangers
    """
    
    def __init__(self):
        self.antibodies = {}
        self.t_cell_memory = deque(maxlen=200)
        self.threat_evolution_tracker = {}
        self.autoimmune_prevention = set()
        self.antibody_generations = 0
        self.threat_severity_threshold = -0.3
        self.memory_consolidation_threshold = 3
        self.adaptive_response_rate = 0.1
        self.total_learning_events = 0
        self.learning_batch_size = 20
        
        # Domain services
        self.antibody_evolution = AntibodyEvolution()
        self.threat_analyzer = ThreatAnalyzer()

    def detect_threats(self, market_state: Dict) -> float:
        """Detect threats using evolved antibodies and T-cell memory"""
        try:
            threat_level = 0.0
            
            pattern_signature = self.threat_analyzer.create_pattern_signature(market_state)
            
            if not pattern_signature:
                logger.debug("Empty pattern signature for immune analysis")
                return 0.0
            
            logger.debug(f"Immune analysis: pattern='{pattern_signature[:50]}...', antibodies={len(self.antibodies)}")
            
            # Check against evolved antibodies
            antibody_matches = 0
            for antibody_pattern, data in self.antibodies.items():
                try:
                    similarity = self.threat_analyzer.pattern_similarity(pattern_signature, antibody_pattern)
                    
                    if similarity > 0.7:
                        antibody_matches += 1
                        # Enhanced threat calculation with memory strength
                        memory_boost = 1.0 + (data.get('memory_count', 0) * 0.1)
                        generation_factor = 1.0 + (data.get('generation', 0) * 0.05)
                        threat_contribution = data.get('strength', 0.0) * similarity * memory_boost * generation_factor
                        
                        if not (np.isnan(threat_contribution) or np.isinf(threat_contribution)):
                            threat_level += threat_contribution
                            logger.debug(f"Antibody match: similarity={similarity:.3f}, contribution={threat_contribution:.4f}")
                except Exception as e:
                    logger.warning(f"Error processing antibody {antibody_pattern}: {e}")
                    continue
            
            # Enhanced T-cell memory response
            tcell_matches = 0
            for past_threat in self.t_cell_memory:
                try:
                    similarity = self.threat_analyzer.pattern_similarity(pattern_signature, past_threat['pattern'])
                    if similarity > 0.8:
                        tcell_matches += 1
                        severity = past_threat.get('severity', 0.0)
                        severity_factor = min(2.0, abs(severity) / 0.1) if severity != 0 else 1.0
                        threat_contribution = severity * similarity * severity_factor
                        
                        if not (np.isnan(threat_contribution) or np.isinf(threat_contribution)):
                            threat_level += threat_contribution
                            logger.debug(f"T-cell match: similarity={similarity:.3f}, contribution={threat_contribution:.4f}")
                except Exception as e:
                    logger.warning(f"Error processing T-cell memory: {e}")
                    continue
            
            # Autoimmune prevention
            if pattern_signature in self.autoimmune_prevention:
                threat_level *= 0.2
                logger.debug("Autoimmune prevention activated")
            
            # Adaptive threat evolution detection
            try:
                evolved_threat_level = self.threat_analyzer.detect_evolved_threats(
                    pattern_signature, market_state, self.threat_evolution_tracker
                )
                if not (np.isnan(evolved_threat_level) or np.isinf(evolved_threat_level)):
                    threat_level += evolved_threat_level
            except Exception as e:
                logger.warning(f"Error in evolved threat detection: {e}")
            
            # If no threats detected but we have antibodies, provide small baseline
            if threat_level == 0.0 and len(self.antibodies) > 0:
                threat_level = -0.01
                logger.debug("No threats detected, using baseline negative signal")
            
            if np.isnan(threat_level) or np.isinf(threat_level):
                logger.warning("Invalid final threat level")
                return 0.0
            
            final_signal = -min(1.0, max(-1.0, threat_level))
            logger.debug(f"Immune result: {final_signal:.4f} (antibody_matches={antibody_matches}, tcell_matches={tcell_matches})")
            
            return final_signal
            
        except Exception as e:
            logger.error(f"Error in immune threat detection: {e}")
            return 0.0

    def learn_threat(self, market_state: Dict, threat_level: float, is_bootstrap: bool = False):
        """Learn from threat outcomes to evolve immune system"""
        if not isinstance(market_state, dict):
            logger.error(f"Market state is not a dict: type={type(market_state)}")
            return

        pattern = self.threat_analyzer.create_pattern_signature(market_state)
        if not pattern:
            return

        self.total_learning_events += 1
        learning_threshold = -0.15 if is_bootstrap else self.threat_severity_threshold

        antibody_created = False
        if threat_level < learning_threshold:  # Significant threat detected
            if pattern in self.antibodies:
                data = self.antibodies[pattern]
                strength_update = self.adaptive_response_rate * (1.0 + data['memory_count'] * 0.1)
                data['strength'] = min(1.0, data['strength'] + strength_update)
                data['memory_count'] += 1
                if data['memory_count'] >= self.memory_consolidation_threshold:
                    data['specificity'] = min(1.0, data['specificity'] + 0.1)
            else:
                self.antibodies[pattern] = {
                    'strength': 0.5,
                    'specificity': 0.7,
                    'memory_count': 1,
                    'generation': self.antibody_generations
                }
                antibody_created = True
            
            # Enhanced T-cell memory
            self.t_cell_memory.append({
                'pattern': pattern,
                'severity': threat_level,
                'timestamp': datetime.now(),
                'market_context': market_state.copy()
            })
            
            # Track threat evolution
            self.threat_analyzer.track_threat_evolution(
                pattern, market_state, threat_level, self.threat_evolution_tracker
            )

        elif threat_level > 0.3:  # False positive (good outcome from supposed threat)
            # Enhanced autoimmune prevention
            self.autoimmune_prevention.add(pattern)
            
            # Weaken antibody if it exists
            if pattern in self.antibodies:
                data = self.antibodies[pattern]
                data['strength'] = max(0.1, data['strength'] - 0.3)
                data['specificity'] = max(0.3, data['specificity'] - 0.1)

    def evolve_antibodies(self):
        """Evolve antibodies using genetic algorithms"""
        if len(self.antibodies) < 10:
            return
        
        self.antibody_generations += 1
        evolved_antibodies = self.antibody_evolution.evolve_antibody_population(
            self.antibodies, self.antibody_generations
        )
        
        # Remove weak antibodies
        final_antibodies = {}
        for pattern, data in evolved_antibodies.items():
            if data['strength'] > 0.2 or data['memory_count'] > 2:
                final_antibodies[pattern] = data
        
        self.antibodies = final_antibodies

    def get_immune_stats(self) -> Dict:
        """Get comprehensive immune system statistics"""
        if not self.antibodies:
            return {'total_antibodies': 0}
        
        strengths = [data['strength'] for data in self.antibodies.values()]
        specificities = [data['specificity'] for data in self.antibodies.values()]
        memory_counts = [data['memory_count'] for data in self.antibodies.values()]
        generations = [data['generation'] for data in self.antibodies.values()]
        
        return {
            'total_antibodies': len(self.antibodies),
            'avg_strength': np.mean(strengths),
            'avg_specificity': np.mean(specificities),
            'total_memory_events': sum(memory_counts),
            'max_generation': max(generations) if generations else 0,
            't_cell_memory_size': len(self.t_cell_memory),
            'autoimmune_prevention_patterns': len(self.autoimmune_prevention),
            'threat_evolution_tracking': len(self.threat_evolution_tracker),
            'antibody_generations': self.antibody_generations
        }