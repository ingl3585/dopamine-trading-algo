import numpy as np
import logging
from collections import deque
from datetime import datetime
from typing import Dict

logger = logging.getLogger(__name__)

class EvolvingImmuneSystem:
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

    def detect_threats(self, market_state: Dict) -> float:
        try:
            threat_level = 0.0
            
            pattern_signature = self._create_pattern_signature(market_state)
            
            if not pattern_signature:
                logger.debug("Empty pattern signature for immune analysis")
                return 0.0
            
            logger.debug(f"Immune analysis: pattern='{pattern_signature[:50]}...', antibodies={len(self.antibodies)}")
            
            antibody_matches = 0
            for antibody_pattern, data in self.antibodies.items():
                try:
                    similarity = self._pattern_similarity(pattern_signature, antibody_pattern)
                    
                    if similarity > 0.7:
                        antibody_matches += 1
                        memory_boost = 1.0 + (data.get('memory_count', 0) * 0.1)
                        generation_factor = 1.0 + (data.get('generation', 0) * 0.05)
                        threat_contribution = data.get('strength', 0.0) * similarity * memory_boost * generation_factor
                        
                        if not (np.isnan(threat_contribution) or np.isinf(threat_contribution)):
                            threat_level += threat_contribution
                            logger.debug(f"Antibody match: similarity={similarity:.3f}, contribution={threat_contribution:.4f}")
                except Exception as e:
                    logger.warning(f"Error processing antibody {antibody_pattern}: {e}")
                    continue
            
            tcell_matches = 0
            for past_threat in self.t_cell_memory:
                try:
                    similarity = self._pattern_similarity(pattern_signature, past_threat['pattern'])
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
            
            if pattern_signature in self.autoimmune_prevention:
                threat_level *= 0.2
                logger.debug("Autoimmune prevention activated")
            
            try:
                evolved_threat_level = self._detect_evolved_threats(pattern_signature, market_state)
                if not (np.isnan(evolved_threat_level) or np.isinf(evolved_threat_level)):
                    threat_level += evolved_threat_level
            except Exception as e:
                logger.warning(f"Error in evolved threat detection: {e}")
            
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

    def _create_pattern_signature(self, market_state: Dict) -> str:
        signature_parts = []
        
        if 'volatility' in market_state:
            vol_bucket = min(99, int(market_state['volatility'] * 1000) // 10)
            signature_parts.append(f"vol_{vol_bucket}")
        
        if 'volume_momentum' in market_state:
            vol_mom = market_state['volume_momentum']
            vol_mom_bucket = min(99, max(0, int((vol_mom + 1.0) * 50)))
            signature_parts.append(f"vmom_{vol_mom_bucket}")
        
        if 'price_momentum' in market_state:
            price_mom = market_state['price_momentum']
            price_mom_bucket = min(99, max(0, int((price_mom + 1.0) * 50)))
            signature_parts.append(f"pmom_{price_mom_bucket}")
        
        if 'time_of_day' in market_state:
            time_val = market_state['time_of_day']
            hour = int(time_val * 24)
            
            if 9 <= hour <= 16:
                session = "regular"
            elif 4 <= hour <= 9:
                session = "premarket"
            elif 16 <= hour <= 20:
                session = "afterhours"
            else:
                session = "overnight"
            
            signature_parts.append(f"session_{session}")
            signature_parts.append(f"hour_{hour}")
        
        if 'regime' in market_state:
            regime = market_state['regime']
            signature_parts.append(f"regime_{regime}")
        
        return "_".join(signature_parts)

    def _pattern_similarity(self, pattern1: str, pattern2: str) -> float:
        if not pattern1 or not pattern2:
            return 0.0
        
        parts1 = pattern1.split("_")
        parts2 = pattern2.split("_")
        
        component_weights = {
            'vol': 0.3,
            'vmom': 0.2,
            'pmom': 0.2,
            'session': 0.15,
            'hour': 0.1,
            'regime': 0.05
        }
        
        total_similarity = 0.0
        total_weight = 0.0
        
        comp1 = {part.split('_')[0]: part for part in parts1 if '_' in part}
        comp2 = {part.split('_')[0]: part for part in parts2 if '_' in part}
        
        for component, weight in component_weights.items():
            if component in comp1 and component in comp2:
                if comp1[component] == comp2[component]:
                    total_similarity += weight
                elif component in ['vol', 'vmom', 'pmom']:
                    try:
                        val1 = int(comp1[component].split('_')[1])
                        val2 = int(comp2[component].split('_')[1])
                        numerical_sim = 1.0 - abs(val1 - val2) / 100.0
                        total_similarity += weight * max(0, numerical_sim)
                    except:
                        pass
                total_weight += weight
        
        return total_similarity / max(total_weight, 1e-8)

    def _detect_evolved_threats(self, pattern_signature: str, market_state: Dict) -> float:
        evolved_threat_level = 0.0
        
        current_volatility = market_state.get('volatility', 0.0)
        current_momentum = market_state.get('price_momentum', 0.0)
        
        if current_volatility > 0.1:
            evolved_threat_level -= 0.2
        
        if abs(current_momentum) > 0.05:
            evolved_threat_level -= 0.1
        
        if pattern_signature in self.threat_evolution_tracker:
            evolution_data = self.threat_evolution_tracker[pattern_signature]
            severity_history = list(evolution_data['severity_history'])
            
            if len(severity_history) >= 3:
                recent_trend = np.mean(severity_history[-3:]) - np.mean(severity_history[:-3])
                if recent_trend < -0.1:
                    evolved_threat_level -= 0.3
        
        return evolved_threat_level

    def learn_threat(self, market_state: Dict, threat_level: float, is_bootstrap: bool = False):
        if not isinstance(market_state, dict):
            logger.error(f"Immune market_state is not a dict: type={type(market_state)}, content={str(market_state)[:200]}")
            return

        pattern = self._create_pattern_signature(market_state)
        if not pattern:
            return

        self.total_learning_events += 1

        learning_threshold = -0.15 if is_bootstrap else self.threat_severity_threshold

        antibody_created = False
        if threat_level < learning_threshold:
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
            
            self.t_cell_memory.append({
                'pattern': pattern,
                'severity': threat_level,
                'timestamp': datetime.now(),
                'market_context': market_state.copy()
            })
            
            self._track_threat_evolution(pattern, market_state, threat_level)

        elif threat_level > 0.3:
            self.autoimmune_prevention.add(pattern)
            
            if pattern in self.antibodies:
                data = self.antibodies[pattern]
                data['strength'] = max(0.1, data['strength'] - 0.3)
                data['specificity'] = max(0.3, data['specificity'] - 0.1)

    def _track_threat_evolution(self, pattern: str, market_state: Dict, threat_level: float):
        if pattern not in self.threat_evolution_tracker:
            self.threat_evolution_tracker[pattern] = {
                'first_seen': datetime.now(),
                'severity_history': deque(maxlen=20),
                'volatility_trend': market_state.get('volatility', 0),
                'mutation_indicators': set()
            }
        
        evolution_data = self.threat_evolution_tracker[pattern]
        evolution_data['severity_history'].append(threat_level)
        
        current_vol = market_state.get('volatility', 0)
        evolution_data['volatility_trend'] = (evolution_data['volatility_trend'] * 0.9 + current_vol * 0.1)
        
        pattern_parts = pattern.split('_')
        for part in pattern_parts:
            if 'extreme' in part or 'spike' in part or 'crash' in part:
                evolution_data['mutation_indicators'].add(part)

    def evolve_antibodies(self):
        if len(self.antibodies) < 10:
            return
        
        self.antibody_generations += 1
        evolved_antibodies = {}
        
        antibody_scores = []
        for pattern, data in self.antibodies.items():
            effectiveness = data['strength'] * data['specificity'] * (1 + data['memory_count'] * 0.1)
            antibody_scores.append((pattern, effectiveness, data))
        
        antibody_scores.sort(key=lambda x: x[1], reverse=True)
        
        top_antibodies = antibody_scores[:len(antibody_scores)//2]
        for pattern, score, data in top_antibodies:
            evolved_antibodies[pattern] = data.copy()
            evolved_antibodies[pattern]['generation'] = self.antibody_generations
        
        for i in range(len(top_antibodies)//2):
            parent1_pattern, _, parent1_data = top_antibodies[i]
            parent2_pattern, _, parent2_data = top_antibodies[(i+1) % len(top_antibodies)]
            
            offspring_pattern = self._crossover_patterns(parent1_pattern, parent2_pattern)
            
            if offspring_pattern and offspring_pattern not in evolved_antibodies:
                offspring_data = {
                    'strength': (parent1_data['strength'] + parent2_data['strength']) / 2,
                    'specificity': (parent1_data['specificity'] + parent2_data['specificity']) / 2,
                    'memory_count': 0,
                    'generation': self.antibody_generations
                }
                
                if np.random.random() < 0.1:
                    offspring_data['strength'] += np.random.normal(0, 0.1)
                    offspring_data['specificity'] += np.random.normal(0, 0.05)
                    offspring_data['strength'] = np.clip(offspring_data['strength'], 0.1, 1.0)
                    offspring_data['specificity'] = np.clip(offspring_data['specificity'], 0.1, 1.0)
                
                evolved_antibodies[offspring_pattern] = offspring_data
        
        final_antibodies = {}
        for pattern, data in evolved_antibodies.items():
            if data['strength'] > 0.2 or data['memory_count'] > 2:
                final_antibodies[pattern] = data
        
        self.antibodies = final_antibodies

    def _crossover_patterns(self, pattern1: str, pattern2: str) -> str:
        parts1 = pattern1.split('_')
        parts2 = pattern2.split('_')
        
        comp1 = {}
        comp2 = {}
        
        for part in parts1:
            if '_' in part:
                key = part.split('_')[0]
                comp1[key] = part
        
        for part in parts2:
            if '_' in part:
                key = part.split('_')[0]
                comp2[key] = part
        
        offspring_parts = []
        all_components = set(comp1.keys()) | set(comp2.keys())
        
        for component in all_components:
            if component in comp1 and component in comp2:
                chosen_part = comp1[component] if np.random.random() < 0.5 else comp2[component]
                offspring_parts.append(chosen_part)
            elif component in comp1:
                offspring_parts.append(comp1[component])
            elif component in comp2:
                offspring_parts.append(comp2[component])
        
        return "_".join(offspring_parts)

    def get_immune_stats(self) -> Dict:
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