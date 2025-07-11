"""
Configuration Manager for AI Trading Personality System

Handles loading, validation, and management of personality configuration
"""

import json
import logging
import os
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path

from .trading_personality import PersonalityConfig
from .personality_integration import PersonalityIntegrationConfig
from src.core.configuration_manager import ConfigurationManager

logger = logging.getLogger(__name__)

@dataclass
class PersonalitySystemConfig:
    """Complete configuration for the personality system"""
    
    # Core personality settings
    personality: PersonalityConfig
    integration: PersonalityIntegrationConfig
    
    # Component configurations
    llm_config: Dict[str, Any]
    emotional_config: Dict[str, Any]
    memory_config: Dict[str, Any]
    commentary_config: Dict[str, Any]
    
    # Development settings
    development_config: Dict[str, Any]
    
    # Available personality presets
    personality_presets: Dict[str, Dict[str, Any]]

class PersonalityConfigManager(ConfigurationManager):
    """
    Manages configuration for the AI Trading Personality system
    Extends ConfigurationManager to avoid duplicate configuration loading logic
    """
    
    def __init__(self, config_file: str = "config/personality_config.json"):
        super().__init__(config_file)
        self.config: Optional[PersonalitySystemConfig] = None
        self.default_config_loaded = False
        
        # Load personality-specific configuration
        self._load_personality_config()
    
    def _load_personality_config(self) -> bool:
        """
        Load personality-specific configuration using base class functionality
        
        Returns:
            bool: Success status
        """
        
        try:
            # The base class loads configuration in __init__, so we can use self.settings
            if self.settings:
                # Parse the loaded data into personality configuration
                self.config = self._parse_config_data(self.settings)
                logger.info(f"Loaded personality configuration from {self.config_file}")
                return True
            else:
                logger.warning(f"Config file {self.config_file} not found, using defaults")
                self.config = self._create_default_config()
                self.default_config_loaded = True
                return True
                
        except Exception as e:
            logger.error(f"Error loading personality configuration: {e}")
            logger.info("Using default configuration")
            self.config = self._create_default_config()
            self.default_config_loaded = True
            return False
    
    
    def get_personality_config(self, personality_name: str = None) -> PersonalityConfig:
        """
        Get personality configuration for specified personality
        
        Args:
            personality_name: Name of personality preset to use
            
        Returns:
            PersonalityConfig: Configured personality settings
        """
        
        if not self.config:
            return PersonalityConfig()
        
        # Use specified personality or default
        if personality_name and personality_name in self.config.personality_presets:
            preset = self.config.personality_presets[personality_name]
            
            return PersonalityConfig(
                personality_name=preset.get('name', personality_name),
                base_confidence=preset.get('base_confidence', 0.6),
                emotional_sensitivity=self.config.emotional_config.get('fear_sensitivity', 0.8),
                memory_weight=preset.get('memory_weight', 0.3),
                consistency_preference=preset.get('consistency_preference', 0.8),
                max_commentary_length=self.config.commentary_config.get('max_length', 200),
                min_commentary_interval=self.config.commentary_config.get('min_interval', 30.0),
                llm_model=self.config.llm_config.get('model_name', 'gpt-4'),
                llm_temperature=self.config.llm_config.get('temperature', 0.7),
                llm_max_tokens=self.config.llm_config.get('max_tokens', 300)
            )
        else:
            return self.config.personality
    
    def get_integration_config(self) -> PersonalityIntegrationConfig:
        """Get integration configuration"""
        return self.config.integration if self.config else PersonalityIntegrationConfig()
    
    def get_llm_config(self) -> Dict[str, Any]:
        """Get LLM configuration"""
        return self.config.llm_config if self.config else {}
    
    
    def get_emotional_config(self) -> Dict[str, Any]:
        """Get emotional engine configuration"""
        return self.config.emotional_config if self.config else {}
    
    def get_memory_config(self) -> Dict[str, Any]:
        """Get memory system configuration"""
        return self.config.memory_config if self.config else {}
    
    def get_commentary_config(self) -> Dict[str, Any]:
        """Get commentary system configuration"""
        return self.config.commentary_config if self.config else {}
    
    def get_available_personalities(self) -> Dict[str, str]:
        """
        Get list of available personality presets
        
        Returns:
            Dict[str, str]: Mapping of personality names to descriptions
        """
        
        if not self.config or not self.config.personality_presets:
            return {'alex': 'Default analytical personality'}
        
        personalities = {}
        for name, preset in self.config.personality_presets.items():
            description = f"{preset.get('name', name)} - {', '.join(preset.get('traits', []))}"
            personalities[name] = description
        
        return personalities
    
    
    
    
    def validate_config(self) -> tuple[bool, list[str]]:
        """
        Validate current configuration
        
        Returns:
            tuple[bool, list[str]]: (is_valid, error_messages)
        """
        
        if not self.config:
            return False, ["No configuration loaded"]
        
        errors = []
        
        # Validate personality config
        if self.config.personality.base_confidence < 0 or self.config.personality.base_confidence > 1:
            errors.append("base_confidence must be between 0 and 1")
        
        if self.config.personality.emotional_sensitivity < 0 or self.config.personality.emotional_sensitivity > 2:
            errors.append("emotional_sensitivity must be between 0 and 2")
        
        # Validate LLM config
        if not self.config.llm_config.get('model_name'):
            errors.append("LLM model_name is required")
        
        if self.config.llm_config.get('temperature', 0.7) < 0 or self.config.llm_config.get('temperature', 0.7) > 2:
            errors.append("LLM temperature must be between 0 and 2")
        
        
        # Validate memory config
        memory_file = self.config.memory_config.get('memory_file')
        if memory_file:
            memory_dir = os.path.dirname(memory_file)
            if memory_dir and not os.path.exists(memory_dir):
                try:
                    os.makedirs(memory_dir, exist_ok=True)
                except Exception:
                    errors.append(f"Cannot create memory directory: {memory_dir}")
        
        return len(errors) == 0, errors
    
    def _parse_config_data(self, config_data: Dict[str, Any]) -> PersonalitySystemConfig:
        """Parse configuration data from file"""
        
        # Extract personality configuration
        personality_data = config_data.get('personality', {})
        personality_config = PersonalityConfig(
            personality_name=personality_data.get('personality_name', 'Alex'),
            base_confidence=personality_data.get('base_confidence', 0.6),
            emotional_sensitivity=personality_data.get('emotional_sensitivity', 0.8),
            memory_weight=personality_data.get('memory_weight', 0.3),
            consistency_preference=personality_data.get('consistency_preference', 0.8),
            max_commentary_length=config_data.get('commentary', {}).get('max_length', 200),
            min_commentary_interval=config_data.get('commentary', {}).get('min_interval', 30.0),
            llm_model=config_data.get('llm', {}).get('model_name', 'gpt-4'),
            llm_temperature=config_data.get('llm', {}).get('temperature', 0.7),
            llm_max_tokens=config_data.get('llm', {}).get('max_tokens', 300),
            llm_base_url=config_data.get('llm', {}).get('base_url', 'http://localhost:11434'),
            llm_api_key=config_data.get('llm', {}).get('api_key', ''),
            mock_llm=config_data.get('development', {}).get('mock_llm', False)
        )
        
        # Extract integration configuration
        integration_data = config_data.get('integration', {})
        integration_config = PersonalityIntegrationConfig(
            enabled=personality_data.get('enabled', True),
            personality_name=personality_data.get('personality_name', 'Alex'),
            auto_commentary=personality_data.get('auto_commentary', True),
            commentary_interval=personality_data.get('commentary_interval', 120.0),
            log_commentary=personality_data.get('log_commentary', True),
            save_commentary_history=personality_data.get('save_commentary_history', True),
            llm_model=config_data.get('llm', {}).get('model_name', 'gpt-4'),
            llm_api_key=config_data.get('llm', {}).get('api_key', '')
        )
        
        return PersonalitySystemConfig(
            personality=personality_config,
            integration=integration_config,
            llm_config=config_data.get('llm', {}),
            emotional_config=config_data.get('emotional_engine', {}),
            memory_config=config_data.get('memory', {}),
            commentary_config=config_data.get('commentary', {}),
            development_config=config_data.get('development', {}),
            personality_presets=config_data.get('personalities', {})
        )
    
    def _create_default_config(self) -> PersonalitySystemConfig:
        """Create default configuration"""
        
        default_personality = PersonalityConfig()
        default_integration = PersonalityIntegrationConfig()
        
        return PersonalitySystemConfig(
            personality=default_personality,
            integration=default_integration,
            llm_config={
                'model_name': 'gpt-4',
                'temperature': 0.7,
                'max_tokens': 300,
                'api_key': '',
                'base_url': ''
            },
            emotional_config={
                'base_confidence': 0.6,
                'fear_sensitivity': 0.8,
                'excitement_threshold': 0.7,
                'emotional_dampening': 0.7
            },
            memory_config={
                'memory_file': 'data/personality_memory.json',
                'short_term_maxlen': 100,
                'long_term_maxlen': 1000,
                'session_maxlen': 50
            },
            commentary_config={
                'max_length': 200,
                'min_interval': 30.0,
                'default_style': 'analytical',
                'default_tone': 'professional'
            },
            development_config={
                'mock_llm': True,
                'debug_logging': True,
                'test_mode': False
            },
            personality_presets={
                'alex': {
                    'name': 'Alex',
                    'traits': ['analytical', 'honest', 'adaptive', 'risk-aware'],
                    'base_confidence': 0.6,
                    'risk_tolerance': 0.5,
                    'emotional_stability': 0.7
                }
            }
        )
    
    def _config_to_dict(self, config: PersonalitySystemConfig) -> Dict[str, Any]:
        """Convert configuration to dictionary format"""
        
        return {
            'personality': {
                'enabled': config.integration.enabled,
                'personality_name': config.personality.personality_name,
                'base_confidence': config.personality.base_confidence,
                'emotional_sensitivity': config.personality.emotional_sensitivity,
                'memory_weight': config.personality.memory_weight,
                'consistency_preference': config.personality.consistency_preference,
                'auto_commentary': config.integration.auto_commentary,
                'commentary_interval': config.integration.commentary_interval,
                'log_commentary': config.integration.log_commentary,
                'save_commentary_history': config.integration.save_commentary_history
            },
            'llm': config.llm_config,
            'emotional_engine': config.emotional_config,
            'memory': config.memory_config,
            'commentary': config.commentary_config,
            'development': config.development_config,
            'personalities': config.personality_presets
        }
    
    def get_config_summary(self) -> Dict[str, Any]:
        """Get summary of current configuration"""
        
        if not self.config:
            return {'status': 'No configuration loaded'}
        
        is_valid, errors = self.validate_config()
        
        return {
            'status': 'valid' if is_valid else 'invalid',
            'errors': errors,
            'personality_name': self.config.personality.personality_name,
            'llm_model': self.config.llm_config.get('model_name', 'unknown'),
            'available_personalities': list(self.config.personality_presets.keys()),
            'development_mode': self.config.development_config.get('test_mode', False),
            'config_file': self.config_file,
            'default_config_used': self.default_config_loaded
        }