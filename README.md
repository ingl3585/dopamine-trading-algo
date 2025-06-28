# Actor-Critic ML Trading System

A fully autonomous, self-evolving black-box trading algorithm for MNQ futures using reinforcement learning. Built with Domain-Driven Design (DDD) and Clean Architecture patterns. Features advanced AI subsystems including genetic algorithms, FFT-based temporal analysis, evolving immune systems, and market microstructure intelligence.

## 🎯 **Key Features**
- **Fully Autonomous**: No hardcoded trading rules - learns optimal behavior through experience
- **Self-Evolving**: Meta-learning adapts all parameters based on market feedback
- **Position-Aware Learning**: Intelligent position limits with economic penalty learning
- **Context-Dependent Holding**: Rewards intelligent patience vs overtrading
- **Emergency Exit Capability**: Always allows position reversals for risk management
- **Real-Time Adaptation**: Works with NinjaTrader at 1x-10x replay speeds

## 🏗️ Architecture Overview

### Domain-Driven Design Structure

```
src/
├── intelligence/                # INTELLIGENCE DOMAIN - AI & ML Components
│   ├── intelligence_engine.py   # Main AI orchestrator with swarm intelligence
│   └── subsystems/              # Four specialized AI subsystems (consolidated)
│       ├── dna_subsystem.py     # 16-base genetic encoding & evolution
│       ├── temporal_subsystem.py # FFT cycle detection & lunar patterns
│       ├── immune_subsystem.py  # Adaptive threat detection system
│       ├── microstructure_subsystem.py # Market microstructure analysis (moved from market/)
│       └── orchestrator.py      # Subsystem coordination
├── agent/                       # AGENT DOMAIN - Trading Agents & Meta-Learning
│   ├── trading_agent.py         # Main trading agent with actor-critic
│   ├── meta_learner.py          # Meta-learning with context-dependent holding rewards
│   └── real_time_adaptation.py  # Real-time market adaptation engine
├── neural/                      # NEURAL DOMAIN - Neural Networks
│   ├── adaptive_network.py      # Self-evolving neural architectures
│   └── enhanced_neural.py       # Advanced neural components
├── market_analysis/             # MARKET DATA DOMAIN - Data Processing & Analysis
│   ├── data_processor.py        # Market data processing with position synchronization
│   ├── market_microstructure.py # Smart money & order flow analysis
│   └── advanced_market_intelligence.py # Comprehensive market AI
├── market_data/                 # MARKET DATA INTERFACES (flattened from market/)
│   └── processor.py             # Market data interfaces (moved up from market/market_data/)
├── data_models/                 # DATA MODELS (separated from trading/)
│   └── models.py                # Trading domain models
├── repositories/                # DATA REPOSITORIES (separated from trading/)
│   └── repositories.py          # Trading data repositories
├── services/                    # SERVICES (separated from trading/)
│   └── trading_service.py       # Trading service layer
├── risk/                        # RISK DOMAIN - Risk Management with Economic Penalties
│   ├── risk_manager.py          # Enhanced with escalating position limit penalties
│   ├── advanced_risk.py         # Advanced risk algorithms
│   ├── portfolio/               # Portfolio management
│   │   └── manager.py           # Portfolio optimization with comprehensive analytics
│   ├── portfolio.py             # Portfolio tracking and analytics
│   └── risk_learning_engine.py  # Risk learning engine
├── communication/               # COMMUNICATION DOMAIN - External Interfaces
│   └── tcp_bridge.py            # TCP server with net liquidation prioritization
├── monitoring/                  # MONITORING DOMAIN - System Health
│   └── system_monitor.py        # System performance monitoring
├── core/                        # CORE SYSTEM - Integration & Configuration
│   ├── orchestrator.py          # Main system orchestrator (renamed from main.py)
│   ├── trading_system.py        # Main trading system coordinator
│   └── config.py                # Environment-aware configuration
└── shared/                      # SHARED KERNEL - Common Types (consolidated)
    └── types.py                 # Combined intelligence & shared types
```

### 🔄 **Recent Architecture Improvements**

#### **Directory Restructuring**
- **Flattened market_data/**: Moved `processor.py` up from `market/market_data/` to `market_data/`
- **Separated trading domain**: Split `trading/` into `data_models/`, `repositories/`, and `services/`
- **Microstructure consolidation**: Moved to `intelligence/subsystems/microstructure_subsystem.py`
- **Types consolidation**: Combined `intelligence_types.py` and `shared/types.py`

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- NinjaTrader 8 with TCP bridge capability
- Required Python packages (see requirements.txt)

### Installation

1. **Clone and setup**:
   ```bash
   cd Actor_Critic_ML_NT
   pip install -r requirements.txt
   ```

2. **Configure environment**:
   ```bash
   # Development
   export TRADING_ENV=development
   
   # Production
   export TRADING_ENV=production
   ```

3. **Start the system**:
   ```bash
   # Main trading system entry point
   python main.py
   
   # Alternative DDD orchestrator
   python src/core/main.py
   ```

### NinjaTrader Setup

1. **Deploy ResearchStrategy.cs** to NinjaTrader 8 strategies folder
2. **Configure TCP ports**:
   - Data Port: 5556 (market data streaming)
   - Signal Port: 5557 (AI signal reception)
3. **Enable multi-entry support**: Strategy supports up to 10 entries per direction
4. **Set instrument**: Configured for MNQ futures trading
5. **Historical data**: Ensure 10+ days of historical data for bootstrap

#### **NinjaTrader Features**
- **Position Reversal Logic**: Automatically handles long↔short reversals
- **Smart Position Limits**: Blocks same-direction scaling at 10 contracts, allows exits
- **Market Time Synchronization**: Uses `Time[0]` for 10x replay compatibility
- **Enhanced Account Data**: Streams net liquidation, margin usage, position size
- **Trade Completion Tracking**: Detailed P&L and timing data to Python

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `TRADING_ENV` | Environment (development/production) | development |
| `TRADING_TCP_DATA_PORT` | TCP data port | 5556 |
| `TRADING_TCP_SIGNAL_PORT` | TCP signal port | 5557 |
| `TRADING_MAX_POSITION_SIZE` | Max position size (0-1) | 0.1 |
| `TRADING_MAX_DAILY_LOSS` | Max daily loss (0-1) | 0.02 |
| `TRADING_LEVERAGE` | Trading leverage | 50.0 |

### Configuration Files

- `config/development.json` - Development settings
- `config/production.json` - Production settings

Example production config:
```json
{
    "log_level": "INFO",
    "max_position_size": 0.05,
    "max_daily_loss": 0.015,
    "kelly_lookback": 200,
    "close_positions_on_shutdown": true
}
```

## 🧠 AI Architecture

### 🎯 **Autonomous Learning System**

The system implements a sophisticated **economic learning mechanism** that teaches optimal trading behavior through experience:

#### **Position Limit Learning**
- **Economic Penalties**: Position limit violations cost -$15+ (3-5x average profit)
- **Escalating Punishments**: Repeated violations get +50% worse each time
- **Smart Boundary Testing**: AI learns limits naturally through economic feedback
- **Emergency Exits Preserved**: Position reversals always allowed for risk management

#### **Context-Dependent Holding Rewards**
- **Intelligent Patience**: Rewards holding during uncertain conditions (confidence < 0.3)
- **Opportunity Cost**: Small penalties for holding during high-confidence signals (> 0.7)
- **Anti-Overtrading**: Discourages trading with very low confidence (< 0.2)
- **Balanced Learning**: Prevents both overtrading and excessive inaction

#### **Meta-Learning Adaptation**
- **Self-Evolving Parameters**: All trading parameters adapt based on market feedback
- **Multi-Component Rewards**: PnL, hold time, win rate, consistency, preservation, holding context
- **Account-Aware Scaling**: Risk parameters adjust automatically for account size
- **Real-Time Learning**: No need for offline training or manual parameter tuning

### Intelligence Domain (`src/intelligence/`)
Central AI coordination with four specialized subsystems:

#### 1. DNA Subsystem (`dna_subsystem.py`)
- **16-base genetic encoding** of market patterns (A-P bases)
- **Genetic breeding algorithms** with elite population management
- **Mutation and natural selection** for pattern evolution
- **Performance-based sequence survival** and aging mechanisms

#### 2. Temporal Subsystem (`temporal_subsystem.py`)
- **FFT-based cycle detection** across multiple timeframes (64, 128, 256)
- **Seasonal and lunar pattern analysis** with interference modeling
- **Dominant cycle tracking** and prediction algorithms
- **Cycle importance weighting** based on historical performance

#### 3. Immune Subsystem (`immune_subsystem.py`)
- **Evolving antibody system** for market threat detection
- **T-cell memory** with adaptive response learning
- **Autoimmune prevention** to avoid false positives
- **Pattern signature matching** with similarity algorithms

#### 4. Microstructure Engine (`market_analysis/market_microstructure.py`)
- **Smart money vs retail flow** pattern detection
- **Order flow analysis** with regime classification
- **Market depth and liquidity** assessment algorithms
- **Real-time tape reading** and momentum detection

### Agent Domain (`src/agent/`)
Trading agents with advanced learning capabilities:
- **Actor-Critic Trading Agent** with self-evolving architecture
- **Meta-Learning Engine** for rapid adaptation
- **Real-Time Adaptation** with multi-armed bandit algorithms
- **Few-Shot Learning** for new market conditions

## 📊 Risk Management

### 🛡️ **Intelligent Risk Learning**

#### **Economic Position Limits**
- **Learning-Based Limits**: AI discovers optimal position sizes through experience
- **Economic Enforcement**: Violations cost more than potential profits (-$15+ penalties)
- **Escalating Consequences**: Repeated violations within 10 minutes get exponentially worse
- **Position Synchronization**: Real-time position tracking between NinjaTrader and Python

#### **Advanced Position Sizing**
- **Kelly Criterion Optimization**: Dynamic position sizing based on historical performance
- **Multi-Factor Assessment**: Confidence, volatility, account balance, and market conditions
- **Risk Learning Engine**: Learns optimal sizing for different market regimes
- **Account-Aware Scaling**: Risk parameters automatically adjust for account size

#### **Multi-Layer Risk Protection**
- **Real-Time Drawdown Prevention**: Dynamic position adjustments during losses
- **Monte Carlo Simulation**: Risk scenario analysis for position sizing
- **Emergency Exit Logic**: Position reversals always allowed regardless of limits
- **Daily Loss Limits**: Configurable maximum daily loss protection

## 🔄 **Autonomous Trading Pipeline**

### 1. **Bootstrap Phase** (Initialization)
- **Historical Data Loading**: 10+ days of multi-timeframe data from NinjaTrader
- **AI Subsystem Calibration**: DNA, temporal, immune, and microstructure initialization
- **Meta-Learning State Restoration**: Load previous learning progress if available
- **Account Size Adaptation**: Automatically adjust risk parameters for account balance

### 2. **Real-Time Decision Loop** (Live Trading)
- **Multi-Timeframe Data Processing**: 1m, 5m, 15m market data streams
- **Four-Subsystem Intelligence Analysis**: Genetic, temporal, immune, microstructure signals
- **Confidence-Based Decision Making**: Hold vs trade decisions based on signal strength
- **Economic Risk Assessment**: Position limits, Kelly sizing, violation tracking
- **NinjaTrader Execution**: Smart position reversals and emergency exit capability

### 3. **Continuous Learning Loop** (Adaptation)
- **Trade Outcome Analysis**: P&L, hold time, exit reason, market conditions
- **Economic Penalty Learning**: Position limit violations with escalating costs
- **Context-Dependent Holding**: Reward patience vs penalize missed opportunities
- **Meta-Parameter Evolution**: All trading parameters self-adjust based on performance
- **Risk Adaptation**: Kelly criterion and position sizing optimization

### 4. **Learning Features**
- **No Hardcoded Rules**: System learns optimal behavior through economic incentives
- **Position Boundary Discovery**: AI tests limits and learns from expensive violations
- **Frequency Optimization**: Learns optimal trading frequency vs holding patterns
- **Market Regime Adaptation**: Parameters adjust automatically for different conditions

## 🧪 Testing

### Run Integration Tests
```bash
python run_tests.py
```

### Test Coverage
- Domain service integration
- AI subsystem functionality
- Risk management validation
- Configuration system testing
- TCP connection handling

## 📈 Monitoring

### System Status Logging
- Trade execution summary
- AI signal analysis
- Risk metrics tracking
- Portfolio performance

### Performance Analytics
- Win rate and profit factor
- Sharpe ratio calculation
- Maximum drawdown tracking
- Kelly fraction optimization

## 🔒 Safety Features

### Emergency Stops
- Maximum margin usage limits (95%)
- Maximum drawdown protection (20%)
- Daily loss limits (configurable)
- Position size constraints

### Graceful Shutdown
- Position closure options
- Final statistics reporting
- TCP connection cleanup
- Data persistence

## 🛠️ Development

### Domain-Driven Design Principles
- **Bounded Contexts**: Each domain (`intelligence/`, `agent/`, `trading/`, `risk/`) is self-contained
- **Domain Services**: Complex business logic encapsulation in service layers
- **Factory Patterns**: Clean object creation with `create_*` functions
- **Dependency Injection**: Loose coupling between domains via interfaces

### Clean Architecture Benefits
- **Separated Concerns**: Intelligence, trading, risk, and market domains are independent
- **Consolidated Subsystems**: Single-file subsystems for easier maintenance
- **Absolute Imports**: All internal imports use `src.` prefix for clarity
- **Modular Design**: Easy to test, extend, and modify individual components

### Code Quality
- Clean Architecture patterns with proper domain separation
- Comprehensive error handling with graceful degradation
- Extensive logging and monitoring across all domains
- Type hints and comprehensive documentation
- Consolidated file structure eliminates redundancy

## 🔧 **Recent Critical Fixes**

### **Position Synchronization Issues** ✅
- **Problem**: NinjaTrader position data wasn't reaching Python risk manager
- **Solution**: Added `total_position_size` field to MarketData class with proper TCP streaming
- **Impact**: Position limits now work correctly, learning feedback restored

### **Market Time vs Real Time** ✅  
- **Problem**: Cooldowns used real-time timestamps, broke at 10x replay speed
- **Solution**: Changed ResearchStrategy.cs to use `Time[0]` (market time) consistently
- **Impact**: System works correctly at 1x-10x replay speeds

### **Economic Learning Failure** ✅
- **Problem**: Position violations (-0.5 penalty) vs profitable trades (+$3-6)
- **Solution**: Increased violation penalties to -$15+ with escalating multipliers
- **Impact**: AI now learns position limits are economically painful

### **Net Liquidation vs Balance** ✅
- **Problem**: System used account balance instead of net liquidation for position sizing
- **Solution**: TCP bridge now prioritizes net liquidation (includes unrealized P&L)
- **Impact**: More accurate position sizing and risk assessment

### **Overtrading vs Holding Balance** ✅
- **Problem**: No reward for holding, AI always tried to trade
- **Solution**: Context-dependent holding rewards based on confidence levels
- **Impact**: AI learns when to be patient vs when to act

## 📝 Learnable Parameters

The system uses meta-learning for **ALL** trading parameters - nothing is hardcoded:

### **Position Management**
- Maximum position sizes (learned through economic penalties)
- Position sizing factors (Kelly criterion optimization)
- Exposure scaling thresholds (risk concentration learning)

### **Trading Behavior**  
- Trading frequency vs holding patterns (confidence-dependent rewards)
- Confidence thresholds for action vs patience
- Stop loss and profit target preferences

### **Risk Management**
- Daily loss tolerance factors (account-aware scaling)
- Margin utilization limits (dynamic based on performance)
- Consecutive loss tolerance (learned resilience)

### **Market Adaptation**
- Subsystem weighting (DNA, temporal, immune, microstructure)
- Exploration vs exploitation balance
- Architecture evolution (neural network sizing)

**Only operational settings are hardcoded**: TCP ports, file paths, emergency safety limits.

## 🔄 Continuous Evolution

### AI Adaptation
- Genetic algorithm evolution
- Neural network weight updates
- Pattern recognition improvement
- Market regime adaptation

### Risk Optimization
- Kelly criterion refinement
- Volatility model updates
- Correlation analysis enhancement
- Drawdown recovery strategies

## 📞 Support

For system configuration, AI parameter tuning, or integration assistance, refer to the codebase documentation and configuration examples.

---

## 🚨 **Important Notes**

### **Autonomous System Characteristics**
- **Fully Autonomous**: System makes all trading decisions without human intervention
- **Self-Evolving**: Parameters continuously adapt based on market feedback  
- **Black-Box Learning**: AI discovers optimal strategies through experience, not rules
- **Position Limit Learning**: AI will test boundaries and learn from expensive violations
- **Emergency Exits**: System can always reverse positions for risk management

### **Development & Testing**
- **Market Replay Testing**: Designed to work at 1x-10x speeds for backtesting
- **Learning Persistence**: System saves/loads learning progress between sessions
- **Comprehensive Logging**: Detailed debug output for all learning decisions
- **Architecture Validation**: All import paths and dependencies verified

### **File Changes Summary**
See `LEARNING_FIXES.md` for detailed technical documentation of all recent improvements.

---

**⚠️ Risk Disclaimer**: This is a fully autonomous, self-evolving algorithmic trading system that learns through economic incentives. The AI will test position limits and market boundaries as part of its learning process. Use appropriate risk management, only trade with funds you can afford to lose, and monitor the system during its learning phase. Past performance does not guarantee future results. The system's autonomous nature means it will make independent trading decisions based on its learned experience.