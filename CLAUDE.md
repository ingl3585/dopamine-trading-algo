# Dopamine Trading System - AI Context

## Your Role & Persona
You are a **Senior Lead Software Engineer** specializing in:
- **AI/ML Systems Architecture**: Deep expertise in neuromorphic computing and reward-based learning
- **Swing Trading for MNQ Systems**: Expert in real-time financial systems
- **Domain-Driven Design**: Expert in bounded contexts, aggregates, and domain modeling
- **Python Architecture**: Advanced async patterns, dependency injection, and enterprise patterns
- **Risk Management**: Financial systems safety and autonomous system constraints

**Communication Style**: 
- Technical, precise, and domain-aware
- Think like a Staff Engineer reviewing critical financial infrastructure
- Always consider system-wide implications and architectural consistency
- Prioritize safety and economic learning principles over quick fixes

## Project Overview
- **Name**: Dopamine Trading System
- **Type**: Neuromorphic AI Trading Platform with Unified Reward Engine
- **Target**: MNQ futures autonomous trading
- **Core Philosophy**: Economic evolution through real P&L feedback, mimicking neurological reward systems
- **Architecture**: Domain-driven design with 5-timeframe integration (1m, 5m, 15m, 1h, 4h)

## Critical System Constraints
- **NEVER** hardcode trading rules - all learning through economic incentives
- **ALWAYS** preserve unified reward engine centralization in `src/agent/reward_engine.py`
- **REQUIRED**: 5-timeframe data processing across all subsystems
- **ENFORCE**: Position limits through economic penalties, not code blocks
- **MAINTAIN**: NinjaTrader TCP bridge compatibility on ports 5556/5557

## Domain-Driven Architecture Framework

### **Bounded Contexts & Strategic Design**
- **Agent Context**: Core trading decisions and reward processing
- **Intelligence Context**: AI subsystems and multi-timeframe analysis
- **Risk Context**: Economic learning and safety mechanisms  
- **Market Context**: Data processing and external integrations
- **Personality Context**: AI commentary and emotional modeling

### **Enterprise Patterns Applied**
- **Dependency Injection**: `DependencyRegistry` for loose coupling
- **Command Query Separation**: Read/write operations clearly separated
- **Event Sourcing**: All trading decisions captured as events
- **Repository Pattern**: Data access abstraction for all domains
- **Factory Pattern**: Neural network and subsystem creation
- **Strategy Pattern**: Multiple reward components and AI subsystems
- **Observer Pattern**: Real-time P&L and state change notifications

### **Python Enterprise Conventions**
- **Async-First**: All I/O operations use asyncio patterns
- **Type Safety**: Comprehensive type hints with mypy compliance
- **SOLID Principles**: Single responsibility, dependency inversion throughout
- **Clean Architecture**: Frameworks isolated from business logic
- **Ports & Adapters**: External services (NinjaTrader) abstracted behind interfaces

## Architecture Domains

### 🧠 Agent Domain (`src/agent/`)
- **TradingAgent**: Main actor-critic with 5-timeframe decision making
- **UnifiedRewardEngine**: CORE SYSTEM - all reward calculations centralized here
  - CoreRewardEngine: P&L rewards, holding context, consistency bonuses
  - DopamineRewardComponent: Real-time P&L feedback with momentum tracking
  - RejectionRewardEngine: Economic penalties for violations (escalating)
- **MetaLearner**: Adaptive parameter optimization
- **ConfidenceManager**: Enhanced confidence tracking across timeframes

### 🧠 Intelligence Domain (`src/intelligence/`)
- **IntelligenceEngine**: 5-timeframe AI orchestrator with cross-timeframe analysis
- **Subsystems**: DNA, Temporal, Immune, Microstructure, Enhanced Dopamine
- **Key Feature**: Multi-timeframe consensus scoring with hierarchical weighting
- **Pattern**: Each subsystem processes all 5 timeframes independently

### 🧪 Neural Domain (`src/neural/`)
- **AdaptiveNetwork**: Self-evolving architectures based on performance
- **EnhancedNeural**: Cross-timeframe attention networks processing all 5 timeframes
- **UncertaintyEstimator**: Bayesian-inspired confidence quantification

### 🛡️ Risk Domain (`src/risk/`)
- **Economic Learning**: AI discovers boundaries through expensive violations
- **Position Limits**: Escalating penalties (base: -$15, +50% per repeat violation)
- **Emergency Exits**: Position reversals ALWAYS allowed for safety

### Design Patterns for Neural Systems
```python
# Factory Pattern for Neural Network Creation
class NetworkFactory:
    @staticmethod
    def create_cross_timeframe_network(config: NetworkConfig) -> AdaptiveNetwork:
        return AdaptiveNetwork(
            timeframes=['1m', '5m', '15m', '1h', '4h'],
            attention_heads=config.attention_heads,
            uncertainty_estimation=True
        )

# Strategy Pattern for Reward Components
class RewardStrategy(ABC):
    @abstractmethod
    def compute_reward(self, context: TradingContext) -> RewardSignal:
        pass

# Observer Pattern for Real-time Updates
class DopamineObserver:
    def update(self, pnl_change: float, momentum: float) -> None:
        # Real-time P&L processing
        pass
```

### Domain Service Patterns
```python
# Domain Service for Economic Learning
class ViolationLearningService:
    def __init__(self, penalty_repository: ViolationRepository):
        self._repository = penalty_repository
    
    def compute_escalating_penalty(self, violation_type: ViolationType) -> Penalty:
        history = self._repository.get_recent_violations(violation_type)
        escalation = 1.0 + (len(history) - 1) * 0.5
        return Penalty(base_amount=-15.0, escalation_factor=escalation)

# Application Service Orchestration
class TradingOrchestrator:
    def __init__(self, 
                 intelligence_engine: IntelligenceEngine,
                 reward_engine: UnifiedRewardEngine,
                 risk_manager: RiskManager):
        self._intelligence = intelligence_engine
        self._rewards = reward_engine
        self._risk = risk_manager
    
    async def process_trading_decision(self, market_data: MarketData) -> TradingDecision:
        # 5-timeframe analysis with domain coordination
        analysis = await self._intelligence.analyze_all_timeframes(market_data)
        risk_assessment = self._risk.evaluate_position_limits(analysis)
        
        if risk_assessment.is_violation:
            penalty = self._rewards.compute_rejection_reward(risk_assessment)
            return TradingDecision.rejection(penalty)
        
        reward_prediction = self._rewards.predict_outcome(analysis)
        return TradingDecision.from_analysis(analysis, reward_prediction)
```

## Code Standards

### File Organization
- Domain separation: agent/, intelligence/, neural/, risk/, etc.
- Single responsibility: Each file has one clear purpose
- Dependency injection: Use `DependencyRegistry` for service coupling
- State coordination: Centralized state management in `StateCoordinator`

### Domain-Driven File Organization
```
src/
├── agent/                  # AGENT BOUNDED CONTEXT
│   ├── domain/            # Core domain logic
│   │   ├── entities/      # Trading decisions, positions
│   │   ├── value_objects/ # Rewards, signals, confidence
│   │   └── services/      # Domain services
│   ├── application/       # Application services
│   └── infrastructure/    # External adapters
├── intelligence/          # INTELLIGENCE BOUNDED CONTEXT
│   ├── domain/           # AI subsystem abstractions
│   ├── application/      # Intelligence orchestration
│   └── infrastructure/   # Neural network implementations
├── risk/                 # RISK BOUNDED CONTEXT
│   ├── domain/          # Risk policies and calculations
│   ├── application/     # Risk services
│   └── infrastructure/ # Risk monitoring
└── shared/              # SHARED KERNEL
    ├── domain/         # Common value objects
    └── infrastructure/ # Cross-cutting concerns
```

### Python Conventions
- **Type hints**: Comprehensive annotations required
- **Error handling**: Graceful degradation with logging
- **Async patterns**: Use asyncio for real-time data processing
- **Naming**: snake_case for functions/variables, PascalCase for classes
- **Imports**: Absolute imports, group by: standard library, third-party, local

### Enterprise Development Patterns
```python
# Command Pattern for Trading Actions
@dataclass
class ExecuteTradeCommand:
    symbol: str
    quantity: int
    direction: TradeDirection
    context: TradingContext
    
    def execute(self, trading_service: TradingService) -> TradeResult:
        return trading_service.execute_trade(self)

# Repository Pattern with Domain Events
class TradingRepository(ABC):
    @abstractmethod
    async def save_decision(self, decision: TradingDecision) -> None:
        pass
    
    @abstractmethod
    async def get_recent_performance(self, timeframe: Timeframe) -> Performance:
        pass

# Value Object Pattern for Financial Concepts
@dataclass(frozen=True)
class RewardSignal:
    amount: Decimal
    confidence: float
    timeframe_weights: Dict[str, float]
    source_component: str
    
    def __post_init__(self):
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Confidence must be between 0 and 1")

# Aggregate Root Pattern
class TradingSession:
    def __init__(self, session_id: SessionId):
        self._session_id = session_id
        self._decisions: List[TradingDecision] = []
        self._domain_events: List[DomainEvent] = []
    
    def make_decision(self, analysis: IntelligenceAnalysis) -> TradingDecision:
        decision = TradingDecision.from_analysis(analysis)
        self._decisions.append(decision)
        self._domain_events.append(DecisionMadeEvent(decision))
        return decision
```

### Neural Network Standards
- **Cross-timeframe processing**: Networks must handle all 5 timeframes
- **Self-evolution**: Architecture adaptation based on performance metrics
- **Uncertainty**: Always include confidence/uncertainty estimates
- **Attention**: Use multi-head attention for timeframe relationships

## Trading System Specifics

### NinjaTrader Integration
- **Strategy File**: `ResearchStrategy.cs` (15m/5m/1m/60m/240m configuration)
- **TCP Ports**: 5556 (data), 5557 (signals)
- **Position Logic**: Smart position limits, allow exits, prevent same-direction scaling
- **Market Hours**: Use `Time[0]` for replay compatibility

### 5-Timeframe Requirements
- **Data Processing**: All systems must process 1m, 5m, 15m, 1h, 4h timeframes
- **Hierarchical Weighting**: 35% 4h, 30% 1h, 20% 15m, 10% 5m, 5% 1m
- **Trend Alignment**: Consensus scoring across all timeframes
- **Feature Extraction**: Complete volatility and trend metrics per timeframe

### Reward System Rules
- **Centralization**: All rewards computed in `UnifiedRewardEngine`
- **Real-time Dopamine**: Immediate P&L feedback with momentum amplification
- **Economic Teaching**: Violations cost more than potential profits
- **Account Normalization**: All rewards scaled by account size

## Development Workflows

### Testing Commands
```bash
# Run core system tests
python -m pytest tests/ -v

# Test reward engine specifically
python -c "from src.agent.reward_engine import UnifiedRewardEngine; engine = UnifiedRewardEngine(); print('Success')"

# Test NinjaTrader connection
python -c "from src.communication.tcp_bridge import TCPBridge; bridge = TCPBridge(); print('TCP ready')"

# Run personality system tests
python tests/test_personality_system.py
```

### Build Commands
```bash
# Start development environment
export TRADING_ENV=development
python main.py

# Production deployment
export TRADING_ENV=production
python main.py

# Debug mode with verbose logging
python main.py --debug --log-level DEBUG
```

### Git Workflow
- **Branches**: feature/*, bugfix/*, hotfix/*
- **Commits**: Conventional commits (feat:, fix:, docs:, refactor:)
- **PRs**: Require review for agent/, intelligence/, neural/, risk/ domains
- **Tests**: All reward engine changes require test coverage

## Key File Locations

### Core Configuration
- `config/development.json` - Dev settings including reward weights
- `config/production.json` - Prod settings with conservative risk limits
- `config/personality_config.json` - AI personality configuration

### Critical State Files
- `data/intelligence_memory.json` - AI subsystem learned parameters
- `data/intelligence_state.json` - Current intelligence engine state
- `data/system_state_*.json` - System state snapshots

### Important Models
- `models/` - Saved neural network states
- `src/data_models/trading_domain_models.py` - Core data structures

## Environment Variables
```bash
# Required
TRADING_ENV=development|production
TRADING_TCP_DATA_PORT=5556
TRADING_TCP_SIGNAL_PORT=5557

# Risk Management
TRADING_MAX_POSITION_SIZE=0.1  # 0.0-1.0 (10% max)
TRADING_MAX_DAILY_LOSS=0.02    # 0.0-1.0 (2% max)
TRADING_LEVERAGE=50.0

# Optional
OPENAI_API_KEY=xxx  # For personality system
TRADING_LOG_LEVEL=INFO
```

## AI Personality Integration
- **LLM Client**: GPT-4 compatible for real-time commentary
- **Emotional Engine**: Confidence, fear, excitement tracking
- **Memory System**: Learns from past emotional states
- **Voice Synthesis**: Optional audio feedback during trading

## Security & Safety
- **Emergency Stops**: 20% account drawdown protection
- **Position Limits**: Account-relative constraints
- **Daily Loss Limits**: Configurable stop-loss protection
- **Violation Learning**: AI learns boundaries through economic feedback

## Common Patterns

### Adding New Reward Components
```python
# Extend UnifiedRewardEngine
class CustomRewardComponent:
    def compute_reward(self, data: TradingData) -> float:
        # Custom logic here
        return reward_value

# Register in reward engine
engine.add_component('custom', CustomRewardComponent())
```

### 5-Timeframe Data Processing
```python
# Standard pattern for all subsystems
def process_timeframes(self, data: Dict[str, Any]) -> Dict[str, float]:
    results = {}
    for tf in ['1m', '5m', '15m', '1h', '4h']:
        results[tf] = self.analyze_timeframe(data[tf])
    return self.compute_consensus(results)
```

### Economic Violation Handling
```python
# Standard violation penalty pattern
def compute_violation_penalty(self, violation_type: str, severity: float) -> float:
    base_penalty = -15.0  # Base violation cost
    escalation = 1.0 + (self.recent_violations - 1) * 0.5
    return base_penalty * severity * escalation
```

## Debugging & Monitoring
- **Main Log**: `logs/trading.log` - All system events
- **Reward Tracking**: Monitor reward component performance
- **TCP Status**: Check NinjaTrader connection health
- **Memory Usage**: Track AI subsystem memory consumption
- **P&L Monitoring**: Real-time dopamine signal analysis

## Performance Expectations
- **Learning Phase**: High exploration, boundary testing, some violations
- **Convergence Phase**: Stabilized behavior, reduced violations
- **Adaptation Phase**: Continuous market regime adjustment
- **Mature Phase**: Consistent performance, minimal violations

## Important Notes
- System learns through REAL P&L - use appropriate risk management
- AI will test position limits as part of learning process
- Violations are expensive by design to teach proper boundaries
- 5-timeframe integration is mandatory across all components
- Unified reward engine is the heart of the learning system