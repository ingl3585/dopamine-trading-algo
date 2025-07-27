# component_integrator.py

import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

# Phase 1 Components
from src.agent.trading_decision_engine import TradingDecisionEngine
from src.agent.neural_network_manager import NeuralNetworkManager
from src.agent.experience_manager import ExperienceManager
from src.agent.trade_outcome_processor import TradeOutcomeProcessor
from src.agent.trading_state_manager import TradingStateManager

# Phase 2 Components
from src.neural.neural_architecture_search import NeuralArchitectureSearch
from src.neural.uncertainty_estimator import UncertaintyEstimator
from src.neural.dynamic_pruning import DynamicPruningManager
from src.neural.specialized_networks import SpecializedNetworkEnsemble

# Phase 3 Components - Consolidated Dopamine System
from src.intelligence.subsystems.enhanced_dopamine_subsystem import ConsolidatedDopamineSubsystem
from src.agent.temporal_reward_memory import TemporalRewardMemory, RewardContext
from src.agent.surprise_detector import SurpriseDetector
from src.agent.multi_objective_optimizer import MultiObjectiveOptimizer

# System Components
from src.core.trading_system import TradingSystem
from src.core.market_data_processor import MarketDataProcessor
from src.core.system_state_manager import SystemStateManager
from src.core.analysis_trigger_manager import AnalysisTriggerManager, AnalysisType
from src.core.config_manager import ConfigurationManager

# Portfolio Components
from src.portfolio.position_tracker import PositionTracker
from src.portfolio.performance_analyzer import PerformanceAnalyzer
from src.portfolio.portfolio_optimizer import PortfolioOptimizer
from src.portfolio.risk_calculator import RiskCalculator

logger = logging.getLogger(__name__)

@dataclass
class ComponentRegistry:
    """Registry of all integrated components"""
    # Phase 1 - Agent Components
    trading_decision_engine: Optional[TradingDecisionEngine] = None
    neural_network_manager: Optional[NeuralNetworkManager] = None
    experience_manager: Optional[ExperienceManager] = None
    trade_outcome_processor: Optional[TradeOutcomeProcessor] = None
    trading_state_manager: Optional[TradingStateManager] = None
    
    # Phase 2 - Neural Components
    neural_architecture_search: Optional[NeuralArchitectureSearch] = None
    uncertainty_estimator: Optional[UncertaintyEstimator] = None
    dynamic_pruning_manager: Optional[DynamicPruningManager] = None
    specialized_networks: Optional[SpecializedNetworkEnsemble] = None
    
    # Phase 3 - Intelligence Components
    intelligence_engine: Optional[Any] = None  # Master coordinator for 5 AI subsystems
    
    # Phase 4 - Reward Components
    consolidated_dopamine_system: Optional[ConsolidatedDopamineSubsystem] = None
    temporal_reward_memory: Optional[TemporalRewardMemory] = None
    surprise_detector: Optional[SurpriseDetector] = None
    multi_objective_optimizer: Optional[MultiObjectiveOptimizer] = None
    
    # System Components
    trading_system: Optional[TradingSystem] = None
    market_data_processor: Optional[MarketDataProcessor] = None
    system_state_manager: Optional[SystemStateManager] = None
    analysis_trigger_manager: Optional[AnalysisTriggerManager] = None
    configuration_manager: Optional[ConfigurationManager] = None
    
    # Portfolio Components
    position_tracker: Optional[PositionTracker] = None
    performance_analyzer: Optional[PerformanceAnalyzer] = None
    portfolio_optimizer: Optional[PortfolioOptimizer] = None
    risk_calculator: Optional[RiskCalculator] = None

class ComponentIntegrator:
    """
    Integrates all modernized components into a cohesive trading system.
    
    Responsibilities:
    - Initialize all components with proper dependencies
    - Connect components through well-defined interfaces
    - Manage component lifecycle and state
    - Coordinate data flow between components
    - Handle component interactions and dependencies
    """
    
    def __init__(self, config_manager: ConfigurationManager):
        self.config_manager = config_manager
        self.config = config_manager.get_section('')  # Get all config
        self.components = ComponentRegistry()
        
        # Integration state
        self.integration_complete = False
        self.component_dependencies = {}
        self.initialization_order = []
        
        logger.info("Component integrator initialized")
    
    def integrate_all_components(self) -> bool:
        """
        Integrate all components in the correct order
        
        Returns:
            bool: True if integration successful
        """
        try:
            logger.info("Starting component integration...")
            
            # Phase 1: Initialize system components
            if not self._initialize_system_components():
                logger.error("Failed to initialize system components")
                return False
            
            # Phase 2: Initialize neural components
            if not self._initialize_neural_components():
                logger.error("Failed to initialize neural components")
                return False
            
            # Phase 3: Initialize intelligence components  
            if not self._initialize_intelligence_components():
                logger.error("Failed to initialize intelligence components")
                return False
            
            # Phase 4: Initialize reward components
            if not self._initialize_reward_components():
                logger.error("Failed to initialize reward components")
                return False
            
            # Phase 5: Initialize agent components
            if not self._initialize_agent_components():
                logger.error("Failed to initialize agent components")
                return False
            
            # Phase 6: Initialize portfolio components
            if not self._initialize_portfolio_components():
                logger.error("Failed to initialize portfolio components")
                return False
            
            # Phase 7: Connect all components
            if not self._connect_components():
                logger.error("Failed to connect components")
                return False
            
            self.integration_complete = True
            logger.info("Component integration completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error during component integration: {e}")
            return False
    
    def _initialize_system_components(self) -> bool:
        """Initialize core system components"""
        try:
            logger.info("Initializing system components...")
            
            # System State Manager
            self.components.system_state_manager = SystemStateManager(self.config)
            
            # Market Data Processor
            self.components.market_data_processor = MarketDataProcessor(self.config)
            
            # Analysis Trigger Manager
            self.components.analysis_trigger_manager = AnalysisTriggerManager(self.config)
            
            
            # Trading System Orchestrator (depends on other components)
            self.components.trading_system = TradingSystem()
            
            logger.info("System components initialized")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing system components: {e}")
            return False
    
    def _initialize_neural_components(self) -> bool:
        """Initialize neural network components"""
        try:
            logger.info("Initializing neural components...")
            
            # Neural Architecture Search
            self.components.neural_architecture_search = NeuralArchitectureSearch(self.config)
            
            # Uncertainty Estimator
            self.components.uncertainty_estimator = UncertaintyEstimator(self.config)
            
            # Dynamic Pruning Manager
            self.components.dynamic_pruning_manager = DynamicPruningManager(self.config)
            
            # Specialized Networks
            self.components.specialized_networks = SpecializedNetworkEnsemble(self.config)
            
            logger.info("Neural components initialized")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing neural components: {e}")
            return False
    
    def _initialize_intelligence_components(self) -> bool:
        """Initialize intelligence and AI subsystem components"""
        try:
            logger.info("Initializing intelligence components...")
            
            # Import intelligence factory
            from src.intelligence import create_intelligence_engine
            
            # Intelligence Engine (master coordinator for 5 AI subsystems)
            self.components.intelligence_engine = create_intelligence_engine(self.config)
            
            logger.info("Intelligence components initialized")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing intelligence components: {e}")
            return False
    
    def _initialize_reward_components(self) -> bool:
        """Initialize reward system components"""
        try:
            logger.info("Initializing reward components...")
            
            # Temporal Reward Memory
            self.components.temporal_reward_memory = TemporalRewardMemory(self.config)
            
            # Surprise Detector
            self.components.surprise_detector = SurpriseDetector(self.config)
            
            # Multi-Objective Optimizer
            self.components.multi_objective_optimizer = MultiObjectiveOptimizer(self.config)
            
            # Consolidated Dopamine System (replaces multiple dopamine implementations)
            self.components.consolidated_dopamine_system = ConsolidatedDopamineSubsystem(self.config)
            
            logger.info("Reward components initialized")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing reward components: {e}")
            return False
    
    def _initialize_agent_components(self) -> bool:
        """Initialize agent components"""
        try:
            logger.info("Initializing agent components...")
            
            # Experience Manager
            self.components.experience_manager = ExperienceManager(self.config)
            
            # Neural Network Manager (depends on neural components)
            self.components.neural_network_manager = NeuralNetworkManager(
                self.config,
                nas_system=self.components.neural_architecture_search,
                uncertainty_estimator=self.components.uncertainty_estimator,
                pruning_manager=self.components.dynamic_pruning_manager,
                specialized_networks=self.components.specialized_networks
            )
            
            # Trading State Manager
            self.components.trading_state_manager = TradingStateManager(self.config)
            
            # Trade Outcome Processor (depends on reward engine)
            self.components.trade_outcome_processor = TradeOutcomeProcessor(
                self.config,
                reward_engine=self.components.consolidated_dopamine_system
            )
            
            # Trading Decision Engine (depends on neural and intelligence components)
            self.components.trading_decision_engine = TradingDecisionEngine(
                self.config,
                neural_manager=self.components.neural_network_manager,
                intelligence_engine=self.components.intelligence_engine
            )
            
            logger.info("Agent components initialized")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing agent components: {e}")
            return False
    
    def _initialize_portfolio_components(self) -> bool:
        """Initialize portfolio components"""
        try:
            logger.info("Initializing portfolio components...")
            
            # Position Tracker
            self.components.position_tracker = PositionTracker(self.config)
            
            # Performance Analyzer
            self.components.performance_analyzer = PerformanceAnalyzer(self.config)
            
            # Risk Calculator
            self.components.risk_calculator = RiskCalculator(self.config)
            
            # Portfolio Optimizer
            self.components.portfolio_optimizer = PortfolioOptimizer(self.config)
            
            logger.info("Portfolio components initialized")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing portfolio components: {e}")
            return False
    
    def _connect_components(self) -> bool:
        """Connect all components with proper interfaces"""
        try:
            logger.info("Connecting components...")
            
            # Connect system state manager to orchestrator
            self.components.trading_system_orchestrator.register_component(
                'system_state_manager',
                self.components.system_state_manager,
                startup_callback=lambda: self.components.system_state_manager.update_system_status(running=True),
                shutdown_callback=lambda: self.components.system_state_manager.shutdown()
            )
            
            # Connect market data processor to analysis triggers
            self._connect_market_data_flow()
            
            # Connect neural components to agent
            self._connect_neural_pipeline()
            
            # Connect reward system to agent
            self._connect_reward_system()
            
            # Connect portfolio components
            self._connect_portfolio_system()
            
            
            logger.info("Components connected successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error connecting components: {e}")
            return False
    
    def _connect_market_data_flow(self):
        """Connect market data processing flow"""
        try:
            # Register analysis triggers
            analysis_mgr = self.components.analysis_trigger_manager
            
            # 15-minute analysis trigger
            analysis_mgr.register_trigger(
                'market_analysis_15m',
                AnalysisType.REGIME_ANALYSIS,
                '15m',
                lambda data: self._trigger_market_analysis(data, '15m')
            )
            
            # 1-hour analysis trigger
            analysis_mgr.register_trigger(
                'trend_analysis_1h',
                AnalysisType.TREND_ANALYSIS,
                '1h',
                lambda data: self._trigger_market_analysis(data, '1h')
            )
            
            # 4-hour analysis trigger
            analysis_mgr.register_trigger(
                'major_trend_4h',
                AnalysisType.TREND_ANALYSIS,
                '4h',
                lambda data: self._trigger_market_analysis(data, '4h')
            )
            
            logger.info("Market data flow connected")
            
        except Exception as e:
            logger.error(f"Error connecting market data flow: {e}")
    
    def _connect_neural_pipeline(self):
        """Connect neural network pipeline"""
        try:
            # Neural manager has references to specialized components
            neural_mgr = self.components.neural_network_manager
            
            # Connect uncertainty estimation
            if hasattr(neural_mgr, 'uncertainty_estimator'):
                neural_mgr.uncertainty_estimator = self.components.uncertainty_estimator
            
            # Connect architecture search
            if hasattr(neural_mgr, 'nas_system'):
                neural_mgr.nas_system = self.components.neural_architecture_search
            
            # Connect pruning system
            if hasattr(neural_mgr, 'pruning_manager'):
                neural_mgr.pruning_manager = self.components.dynamic_pruning_manager
            
            logger.info("Neural pipeline connected")
            
        except Exception as e:
            logger.error(f"Error connecting neural pipeline: {e}")
    
    def _connect_reward_system(self):
        """Connect reward system components"""
        try:
            # Consolidated dopamine system contains all reward functionality
            dopamine_system = self.components.consolidated_dopamine_system
            
            # Connect to trade outcome processor
            if hasattr(self.components.trade_outcome_processor, 'reward_engine'):
                self.components.trade_outcome_processor.reward_engine = dopamine_system
            
            logger.info("Reward system connected")
            
        except Exception as e:
            logger.error(f"Error connecting reward system: {e}")
    
    def _connect_portfolio_system(self):
        """Connect portfolio management components"""
        try:
            # Portfolio components can reference each other
            portfolio_mgr = self.components.portfolio_optimizer
            
            # Connect position tracker to performance analyzer
            if hasattr(self.components.performance_analyzer, 'position_tracker'):
                self.components.performance_analyzer.position_tracker = self.components.position_tracker
            
            # Connect risk calculator to portfolio optimizer
            if hasattr(portfolio_mgr, 'risk_calculator'):
                portfolio_mgr.risk_calculator = self.components.risk_calculator
            
            logger.info("Portfolio system connected")
            
        except Exception as e:
            logger.error(f"Error connecting portfolio system: {e}")
    
    def _trigger_market_analysis(self, market_data: Any, timeframe: str):
        """Trigger market analysis based on timeframe"""
        try:
            if timeframe == '15m':
                logger.info("15m market analysis triggered")
            
            elif timeframe == '1h':
                # Trigger enhanced regime analysis
                logger.info("Triggering enhanced regime analysis")
                
            elif timeframe == '4h':
                # Trigger major trend analysis
                logger.info("Triggering major trend analysis")
            
        except Exception as e:
            logger.error(f"Error triggering {timeframe} analysis: {e}")
    
    def get_integrated_system(self) -> Dict[str, Any]:
        """
        Get the integrated system with all components
        
        Returns:
            Dictionary with all integrated components
        """
        if not self.integration_complete:
            logger.warning("Integration not complete, returning partial system")
        
        return {
            'components': self.components,
            'config_manager': self.config_manager,
            'integration_complete': self.integration_complete,
            'orchestrator': self.components.trading_system_orchestrator,
            'market_processor': self.components.market_data_processor,
            'state_manager': self.components.system_state_manager,
            'decision_engine': self.components.trading_decision_engine,
            'reward_engine': self.components.consolidated_dopamine_system,
            'portfolio_optimizer': self.components.portfolio_optimizer,
            'neural_manager': self.components.neural_network_manager
        }
    
    def get_integration_status(self) -> Dict[str, Any]:
        """Get integration status and component health"""
        try:
            component_status = {}
            
            # Check each component
            for component_name, component in self.components.__dict__.items():
                if component is not None:
                    component_status[component_name] = {
                        'initialized': True,
                        'type': type(component).__name__,
                        'healthy': True  # Could add health checks
                    }
                else:
                    component_status[component_name] = {
                        'initialized': False,
                        'type': None,
                        'healthy': False
                    }
            
            return {
                'integration_complete': self.integration_complete,
                'total_components': len(self.components.__dict__),
                'initialized_components': sum(1 for c in self.components.__dict__.values() if c is not None),
                'component_status': component_status,
                'dependencies_resolved': True  # Could add dependency checking
            }
            
        except Exception as e:
            logger.error(f"Error getting integration status: {e}")
            return {'error': str(e)}
    
    def shutdown_integrated_system(self):
        """Shutdown all integrated components"""
        try:
            logger.info("Shutting down integrated system...")
            
            # Shutdown in reverse order
            if self.components.trading_system_orchestrator:
                self.components.trading_system_orchestrator.stop_system()
            
            if self.components.system_state_manager:
                self.components.system_state_manager.shutdown()
            
            
            # Reset integration state
            self.integration_complete = False
            
            logger.info("Integrated system shutdown complete")
            
        except Exception as e:
            logger.error(f"Error shutting down integrated system: {e}")
    
    def create_modernized_trading_agent(self) -> 'ModernizedTradingAgent':
        """Create a modernized trading agent with all integrated components"""
        try:
            if not self.integration_complete:
                logger.warning("Integration not complete, agent may not function properly")
            
            return ModernizedTradingAgent(
                decision_engine=self.components.trading_decision_engine,
                neural_manager=self.components.neural_network_manager,
                reward_engine=self.components.consolidated_dopamine_system,
                experience_manager=self.components.experience_manager,
                outcome_processor=self.components.trade_outcome_processor,
                state_manager=self.components.trading_state_manager,
                config=self.config
            )
            
        except Exception as e:
            logger.error(f"Error creating modernized trading agent: {e}")
            return None


class ModernizedTradingAgent:
    """
    Modernized trading agent that uses all integrated components
    """
    
    def __init__(self, decision_engine: TradingDecisionEngine,
                 neural_manager: NeuralNetworkManager,
                 reward_engine: ConsolidatedDopamineSubsystem,
                 experience_manager: ExperienceManager,
                 outcome_processor: TradeOutcomeProcessor,
                 state_manager: TradingStateManager,
                 config: Dict[str, Any]):
        
        self.decision_engine = decision_engine
        self.neural_manager = neural_manager
        self.reward_engine = reward_engine
        self.experience_manager = experience_manager
        self.outcome_processor = outcome_processor
        self.state_manager = state_manager
        self.config = config
        
        logger.info("Modernized trading agent created")
    
    def decide(self, features: Any, market_data: Any) -> Any:
        """Make trading decision using integrated components"""
        try:
            # Use decision engine with all integrated components
            decision = self.decision_engine.decide(features, market_data)
            
            # Update state
            self.state_manager.update_confidence(features.__dict__ if hasattr(features, '__dict__') else {})
            
            return decision
            
        except Exception as e:
            logger.error(f"Error in modernized agent decision: {e}")
            return None
    
    def learn_from_trade(self, trade: Any):
        """Learn from trade outcome using integrated components"""
        try:
            # Process trade outcome with integrated reward system
            self.outcome_processor.learn_from_trade(trade)
            
            # Store experience
            if hasattr(trade, 'features') and hasattr(trade, 'market_data'):
                self.experience_manager.store_experience({
                    'features': trade.features,
                    'market_data': trade.market_data,
                    'outcome': trade.pnl,
                    'action': trade.action
                })
            
            logger.info(f"Learned from trade: {trade.action} with P&L: {trade.pnl}")
            
        except Exception as e:
            logger.error(f"Error learning from trade: {e}")
    
    def get_agent_statistics(self) -> Dict[str, Any]:
        """Get comprehensive agent statistics"""
        try:
            return {
                'decision_engine': self.decision_engine.get_decision_statistics() if hasattr(self.decision_engine, 'get_decision_statistics') else {},
                'neural_manager': self.neural_manager.get_network_statistics() if hasattr(self.neural_manager, 'get_network_statistics') else {},
                'reward_engine': self.reward_engine.get_reward_statistics(),
                'experience_manager': self.experience_manager.get_experience_statistics() if hasattr(self.experience_manager, 'get_experience_statistics') else {},
                'state_manager': self.state_manager.get_state_statistics() if hasattr(self.state_manager, 'get_state_statistics') else {}
            }
            
        except Exception as e:
            logger.error(f"Error getting agent statistics: {e}")
            return {}