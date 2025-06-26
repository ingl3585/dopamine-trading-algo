"""
Antibody Evolution Engine - Genetic algorithms for antibody evolution
"""

import numpy as np
import random
import logging
from typing import Dict

logger = logging.getLogger(__name__)

class AntibodyEvolution:
    """
    Advanced antibody evolution with genetic algorithms
    """
    
    def evolve_antibody_population(self, antibodies: Dict, generation_count: int) -> Dict:
        """Evolve antibody population using genetic algorithms"""
        evolved_antibodies = {}
        
        # Sort antibodies by effectiveness
        antibody_scores = []
        for pattern, data in antibodies.items():
            effectiveness = data['strength'] * data['specificity'] * (1 + data['memory_count'] * 0.1)
            antibody_scores.append((pattern, effectiveness, data))
        
        antibody_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Keep top performers
        top_antibodies = antibody_scores[:len(antibody_scores)//2]
        for pattern, score, data in top_antibodies:
            evolved_antibodies[pattern] = data.copy()
            evolved_antibodies[pattern]['generation'] = generation_count
        
        # Generate new antibodies through mutation and crossover
        for i in range(len(top_antibodies)//2):
            parent1_pattern, _, parent1_data = top_antibodies[i]
            parent2_pattern, _, parent2_data = top_antibodies[(i+1) % len(top_antibodies)]
            
            # Crossover: combine pattern components
            offspring_pattern = self._crossover_patterns(parent1_pattern, parent2_pattern)
            
            if offspring_pattern and offspring_pattern not in evolved_antibodies:
                # Inherit traits from parents
                offspring_data = {
                    'strength': (parent1_data['strength'] + parent2_data['strength']) / 2,
                    'specificity': (parent1_data['specificity'] + parent2_data['specificity']) / 2,
                    'memory_count': 0,
                    'generation': generation_count
                }
                
                # Mutation
                if np.random.random() < 0.1:
                    offspring_data['strength'] += np.random.normal(0, 0.1)
                    offspring_data['specificity'] += np.random.normal(0, 0.05)
                    offspring_data['strength'] = np.clip(offspring_data['strength'], 0.1, 1.0)
                    offspring_data['specificity'] = np.clip(offspring_data['specificity'], 0.1, 1.0)
                
                evolved_antibodies[offspring_pattern] = offspring_data
        
        return evolved_antibodies

    def _crossover_patterns(self, pattern1: str, pattern2: str) -> str:
        """Create offspring pattern through crossover"""
        parts1 = pattern1.split('_')
        parts2 = pattern2.split('_')
        
        # Create component dictionaries
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
        
        # Combine components randomly
        offspring_parts = []
        all_components = set(comp1.keys()) | set(comp2.keys())
        
        for component in all_components:
            if component in comp1 and component in comp2:
                # Choose randomly from parents
                chosen_part = comp1[component] if np.random.random() < 0.5 else comp2[component]
                offspring_parts.append(chosen_part)
            elif component in comp1:
                offspring_parts.append(comp1[component])
            elif component in comp2:
                offspring_parts.append(comp2[component])
        
        return "_".join(offspring_parts)