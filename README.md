# Actor-Critic ML Trading System

A sophisticated, modular trading system built with Domain-Driven Design (DDD) and Clean Architecture patterns. Features advanced AI subsystems including genetic algorithms, FFT-based temporal analysis, evolving immune systems, and market microstructure intelligence.

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
│       └── orchestrator.py      # Subsystem coordination
├── agent/                       # AGENT DOMAIN - Trading Agents & Meta-Learning
│   ├── trading_agent.py         # Main trading agent with actor-critic
│   ├── meta_learner.py          # Meta-learning and adaptation
│   └── real_time_adaptation.py  # Real-time market adaptation engine
├── neural/                      # NEURAL DOMAIN - Neural Networks
│   ├── adaptive_network.py      # Self-evolving neural architectures
│   └── enhanced_neural.py       # Advanced neural components
├── market_analysis/             # MARKET DOMAIN - Data Processing & Analysis
│   ├── data_processor.py        # Market data processing & features
│   ├── market_microstructure.py # Smart money & order flow analysis
│   └── advanced_market_intelligence.py # Comprehensive market AI
├── trading/                     # TRADING DOMAIN - Execution & Positions
│   ├── domain/                  # Core trading business logic
│   │   ├── services.py          # Trading service layer
│   │   ├── models.py            # Trading domain models
│   │   └── repositories.py      # Trading data repositories
│   └── infrastructure/          # NinjaTrader integration
│       └── ninjatrader.py       # TCP bridge implementation
├── risk/                        # RISK DOMAIN - Risk Management
│   ├── risk_manager.py          # Main risk management coordinator
│   ├── advanced_risk.py         # Advanced risk algorithms
│   ├── portfolio.py             # Portfolio tracking and analytics
│   ├── management/service.py    # Kelly criterion & dynamic risk
│   └── portfolio/manager.py     # Portfolio optimization
├── communication/               # COMMUNICATION DOMAIN - External Interfaces
│   └── tcp_bridge.py            # TCP server for NinjaTrader
├── market/                      # MARKET INTERFACE - External Market Data
│   ├── market_data/processor.py # Market data interfaces
│   └── microstructure/analyzer.py # Market structure analysis
├── monitoring/                  # MONITORING DOMAIN - System Health
│   └── system_monitor.py        # System performance monitoring
├── core/                        # CORE SYSTEM - Integration & Configuration
│   ├── main.py                  # Alternative orchestrator entry point
│   ├── trading_system.py        # Main trading system coordinator
│   └── config.py                # Environment-aware configuration
└── shared/                      # SHARED KERNEL - Common Types
    └── types.py                 # Domain interfaces & data types
```

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

1. Configure NinjaTrader TCP bridge on ports:
   - Data Port: 5556
   - Signal Port: 5557

2. Ensure historical data is available for bootstrap

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

### Kelly Criterion Position Sizing
- Dynamic position sizing based on historical performance
- Risk-adjusted returns optimization
- Drawdown protection mechanisms

### Multi-Factor Risk Assessment
- Position risk analysis
- Confidence-based adjustments
- Volatility and correlation factors
- Daily loss limits and emergency stops

## 🔄 Trading Pipeline

1. **Bootstrap Phase**:
   - Historical data collection from NinjaTrader
   - AI subsystem training and calibration
   - Pattern recognition initialization

2. **Live Trading Loop**:
   - Real-time market data processing
   - AI analysis with all four subsystems
   - Risk assessment and position sizing
   - Trade execution and portfolio management

3. **Continuous Learning**:
   - Trade outcome analysis
   - AI subsystem evolution
   - Risk parameter optimization

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

## 📝 Learnable Parameters

The system uses meta-learning for most trading parameters:
- Risk per trade factors
- Position sizing multipliers
- Confidence thresholds
- Stop/target preferences
- Maximum trade frequencies

Only operational settings are hardcoded (ports, directories, emergency limits).

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

**⚠️ Risk Disclaimer**: This is a sophisticated algorithmic trading system. Use appropriate risk management and only trade with funds you can afford to lose. Past performance does not guarantee future results.