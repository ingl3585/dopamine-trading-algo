# src/core/dependency_manager.py

"""
Centralized Dependency Management System

This module provides a clean, tiered approach to dependency management with graceful fallbacks
for optional dependencies. It follows the Single Responsibility Principle by centralizing
all dependency checking and import management.

Design Principles:
- Separation of Concerns: Dependencies are categorized by tier and functionality
- Open/Closed Principle: Easy to add new dependencies without modifying existing code
- Dependency Inversion: High-level modules don't depend on low-level import details
"""

import logging
import sys
import warnings
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
from dataclasses import dataclass

logger = logging.getLogger(__name__)

class DependencyTier(Enum):
    """Dependency classification for tiered management"""
    ESSENTIAL = "essential"      # System cannot function without these
    ENHANCED = "enhanced"        # System works but with reduced functionality
    OPTIONAL = "optional"        # System works fine, just missing convenience features

@dataclass
class DependencySpec:
    """Specification for a dependency with fallback information"""
    name: str
    tier: DependencyTier
    fallback_available: bool
    fallback_message: str
    import_names: List[str]  # Specific imports needed from this package

class DependencyManager:
    """
    Centralized dependency manager following clean architecture principles.
    
    Responsibilities:
    - Check and validate dependencies on system startup
    - Provide graceful fallbacks for missing optional dependencies
    - Maintain clear separation between essential and optional functionality
    - Offer consistent error messaging and user guidance
    """
    
    def __init__(self):
        self.available_dependencies: Dict[str, bool] = {}
        self.fallback_implementations: Dict[str, Any] = {}
        self.dependency_specs = self._initialize_dependency_specs()
        self._setup_fallback_implementations()
    
    def _initialize_dependency_specs(self) -> Dict[str, DependencySpec]:
        """Initialize dependency specifications following the Open/Closed principle"""
        return {
            # Essential Dependencies (Tier 1)
            'numpy': DependencySpec(
                name='numpy',
                tier=DependencyTier.ESSENTIAL,
                fallback_available=False,
                fallback_message='NumPy is essential for mathematical operations',
                import_names=['numpy']
            ),
            'torch': DependencySpec(
                name='torch',
                tier=DependencyTier.ESSENTIAL,
                fallback_available=False,
                fallback_message='PyTorch is essential for neural network functionality',
                import_names=['torch']
            ),
            
            # Enhanced Dependencies (Tier 2)
            'scipy': DependencySpec(
                name='scipy',
                tier=DependencyTier.ENHANCED,
                fallback_available=True,
                fallback_message='SciPy enhances FFT and statistical analysis. NumPy alternatives will be used.',
                import_names=['scipy.fft', 'scipy.stats', 'scipy.optimize']
            ),
            
            # Optional Dependencies (Tier 3)
            'matplotlib': DependencySpec(
                name='matplotlib',
                tier=DependencyTier.OPTIONAL,
                fallback_available=True,
                fallback_message='Matplotlib enables advanced plotting. Basic visualization disabled.',
                import_names=['matplotlib.pyplot']
            )
        }
    
    def check_all_dependencies(self) -> Dict[str, Any]:
        """
        Comprehensive dependency check with detailed reporting.
        
        Returns:
            Dict containing check results and system capabilities
        """
        results = {
            'all_essential_available': True,
            'missing_essential': [],
            'missing_enhanced': [],
            'missing_optional': [],
            'system_capabilities': {},
            'warnings': []
        }
        
        logger.info("Starting comprehensive dependency check...")
        
        for dep_name, spec in self.dependency_specs.items():
            is_available = self._check_single_dependency(dep_name, spec)
            self.available_dependencies[dep_name] = is_available
            
            if not is_available:
                if spec.tier == DependencyTier.ESSENTIAL:
                    results['all_essential_available'] = False
                    results['missing_essential'].append(dep_name)
                    logger.critical(f"ESSENTIAL dependency missing: {dep_name}")
                elif spec.tier == DependencyTier.ENHANCED:
                    results['missing_enhanced'].append(dep_name)
                    results['warnings'].append(f"Enhanced functionality disabled: {spec.fallback_message}")
                    logger.warning(f"ENHANCED dependency missing: {dep_name} - {spec.fallback_message}")
                else:  # OPTIONAL
                    results['missing_optional'].append(dep_name)
                    logger.info(f"Optional dependency missing: {dep_name} - {spec.fallback_message}")
        
        # Set system capabilities based on available dependencies
        results['system_capabilities'] = self._determine_system_capabilities()
        
        self._log_dependency_summary(results)
        return results
    
    def _check_single_dependency(self, dep_name: str, spec: DependencySpec) -> bool:
        """Check availability of a single dependency with detailed import testing"""
        try:
            # Test each required import from the dependency
            for import_name in spec.import_names:
                __import__(import_name)
            
            logger.debug(f"✓ {dep_name} successfully imported")
            return True
            
        except ImportError as e:
            logger.debug(f"✗ {dep_name} import failed: {e}")
            return False
        except Exception as e:
            logger.warning(f"Unexpected error checking {dep_name}: {e}")
            return False
    
    def _determine_system_capabilities(self) -> Dict[str, bool]:
        """Determine what system capabilities are available based on dependencies"""
        capabilities = {
            'neural_networks': self.available_dependencies.get('torch', False),
            'advanced_math': self.available_dependencies.get('numpy', False),
            'enhanced_fft': self.available_dependencies.get('scipy', False),
            'statistical_analysis': self.available_dependencies.get('scipy', False),
            'optimization': self.available_dependencies.get('scipy', False),
            'visualization': self.available_dependencies.get('matplotlib', False)
        }
        
        # Composite capabilities
        capabilities['full_temporal_analysis'] = (
            capabilities['advanced_math'] and capabilities['enhanced_fft']
        )
        capabilities['advanced_risk_management'] = (
            capabilities['advanced_math'] and capabilities['statistical_analysis']
        )
        
        return capabilities
    
    def _setup_fallback_implementations(self):
        """Setup fallback implementations for missing dependencies"""
        # NumPy-based FFT fallback for SciPy
        self.fallback_implementations['scipy_fft'] = self._create_numpy_fft_fallback()
        
        # Basic statistics fallback for SciPy stats
        self.fallback_implementations['scipy_stats'] = self._create_basic_stats_fallback()
        
        # Simple optimization fallback for SciPy optimize
        self.fallback_implementations['scipy_optimize'] = self._create_basic_optimize_fallback()
    
    def _create_numpy_fft_fallback(self) -> object:
        """Create NumPy-based FFT fallback for SciPy FFT functionality"""
        import numpy as np
        
        class NumpyFFTFallback:
            """NumPy-based fallback for scipy.fft functionality"""
            
            @staticmethod
            def fft(data):
                """Basic FFT using NumPy"""
                return np.fft.fft(data)
            
            @staticmethod
            def fftfreq(n, d=1.0):
                """Frequency array using NumPy"""
                return np.fft.fftfreq(n, d)
            
            @staticmethod
            def rfft(data):
                """Real FFT using NumPy"""
                return np.fft.rfft(data)
            
            @staticmethod
            def irfft(data):
                """Inverse real FFT using NumPy"""
                return np.fft.irfft(data)
        
        return NumpyFFTFallback()
    
    def _create_basic_stats_fallback(self) -> object:
        """Create basic statistics fallback for SciPy stats functionality"""
        import numpy as np
        
        class BasicStatsFallback:
            """Basic statistics fallback for scipy.stats functionality"""
            
            @staticmethod
            def percentileofscore(data, score):
                """Calculate percentile score using NumPy"""
                return (np.sum(np.array(data) <= score) / len(data)) * 100
            
            @staticmethod
            def genpareto():
                """Placeholder for Generalized Pareto Distribution"""
                class GenParetoFallback:
                    @staticmethod
                    def fit(data):
                        # Simple fallback using basic statistics
                        mean_val = np.mean(data)
                        std_val = np.std(data)
                        return 0.1, mean_val, std_val  # shape, loc, scale
                
                return GenParetoFallback()
            
            @staticmethod
            def corrcoef(*args, **kwargs):
                """Correlation coefficient using NumPy"""
                return np.corrcoef(*args, **kwargs)
        
        return BasicStatsFallback()
    
    def _create_basic_optimize_fallback(self) -> object:
        """Create basic optimization fallback for SciPy optimize functionality"""
        import numpy as np
        
        class BasicOptimizeFallback:
            """Basic optimization fallback for scipy.optimize functionality"""
            
            @staticmethod
            def minimize_scalar(func, bounds=None, method='bounded'):
                """Simple grid search optimization"""
                if bounds is None:
                    bounds = (-10, 10)
                
                # Simple grid search
                x_values = np.linspace(bounds[0], bounds[1], 100)
                y_values = [func(x) for x in x_values]
                min_idx = np.argmin(y_values)
                
                class OptimizeResult:
                    def __init__(self, x, fun):
                        self.x = x
                        self.fun = fun
                        self.success = True
                
                return OptimizeResult(x_values[min_idx], y_values[min_idx])
        
        return BasicOptimizeFallback()
    
    def get_import(self, dependency_name: str, fallback_key: Optional[str] = None):
        """
        Get import with automatic fallback handling.
        
        Args:
            dependency_name: Name of the dependency to import
            fallback_key: Key for fallback implementation if needed
            
        Returns:
            The imported module or fallback implementation
            
        Raises:
            ImportError: If essential dependency is missing and no fallback available
        """
        if self.available_dependencies.get(dependency_name, False):
            return __import__(dependency_name)
        
        # Check if fallback is available
        spec = self.dependency_specs.get(dependency_name)
        if spec and spec.fallback_available and fallback_key:
            fallback = self.fallback_implementations.get(fallback_key)
            if fallback:
                logger.info(f"Using fallback implementation for {dependency_name}")
                return fallback
        
        # Handle missing essential dependencies
        if spec and spec.tier == DependencyTier.ESSENTIAL:
            raise ImportError(
                f"Essential dependency '{dependency_name}' is not available. "
                f"Please install it using: pip install {dependency_name}"
            )
        
        # For enhanced/optional dependencies, return None to signal unavailability
        logger.warning(f"Dependency '{dependency_name}' not available, returning None")
        return None
    
    def is_available(self, dependency_name: str) -> bool:
        """Check if a dependency is available"""
        return self.available_dependencies.get(dependency_name, False)
    
    def get_fallback(self, fallback_key: str):
        """Get a specific fallback implementation"""
        return self.fallback_implementations.get(fallback_key)
    
    def _log_dependency_summary(self, results: Dict[str, Any]):
        """Log comprehensive dependency check summary"""
        logger.info("=== Dependency Check Summary ===")
        
        if results['all_essential_available']:
            logger.info("✓ All essential dependencies available")
        else:
            logger.critical("✗ Missing essential dependencies: " + 
                          ", ".join(results['missing_essential']))
        
        if results['missing_enhanced']:
            logger.warning("⚠ Missing enhanced dependencies: " + 
                          ", ".join(results['missing_enhanced']))
        
        if results['missing_optional']:
            logger.info("ℹ Missing optional dependencies: " + 
                       ", ".join(results['missing_optional']))
        
        capabilities = results['system_capabilities']
        enabled_features = [k for k, v in capabilities.items() if v]
        disabled_features = [k for k, v in capabilities.items() if not v]
        
        if enabled_features:
            logger.info("✓ Enabled features: " + ", ".join(enabled_features))
        if disabled_features:
            logger.info("✗ Disabled features: " + ", ".join(disabled_features))
        
        logger.info("=== End Dependency Summary ===")
    
    def validate_runtime_requirements(self, required_capabilities: List[str]) -> bool:
        """
        Validate that runtime requirements are met for specific functionality.
        
        Args:
            required_capabilities: List of capability names that must be available
            
        Returns:
            True if all required capabilities are available
        """
        capabilities = self._determine_system_capabilities()
        missing_capabilities = [
            cap for cap in required_capabilities 
            if not capabilities.get(cap, False)
        ]
        
        if missing_capabilities:
            logger.error(f"Missing required capabilities: {missing_capabilities}")
            return False
        
        return True

# Global instance for system-wide dependency management
dependency_manager = DependencyManager()

def check_dependencies() -> Dict[str, Any]:
    """Convenience function for dependency checking"""
    return dependency_manager.check_all_dependencies()

def get_safe_import(dependency_name: str, fallback_key: Optional[str] = None):
    """Convenience function for safe imports with fallbacks"""
    return dependency_manager.get_import(dependency_name, fallback_key)

def is_dependency_available(dependency_name: str) -> bool:
    """Convenience function to check dependency availability"""
    return dependency_manager.is_available(dependency_name)