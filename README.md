# Actor-Critic ML Trading System

A sophisticated, modular trading system built with Domain-Driven Design (DDD) and Clean Architecture patterns. Features advanced AI subsystems including genetic algorithms, FFT-based temporal analysis, evolving immune systems, and market microstructure intelligence.

## 🏗️ Architecture Overview

### Domain-Driven Design Structure

```
src/
├── ai/                          # AI DOMAIN - Sophisticated Intelligence
│   └── intelligence/            
│       ├── engine.py           # Main AI orchestrator with swarm intelligence
│       └── subsystems/         # Four specialized AI subsystems
│           ├── dna/            # 16-base genetic encoding & evolution
│           ├── temporal/       # FFT cycle detection & lunar patterns
│           ├── immune/         # Adaptive threat detection system
│           └── microstructure/ # Smart money & order flow analysis
├── trading/                    # TRADING DOMAIN - Execution & Positions
│   ├── domain/services.py      # Core trading business logic
│   └── infrastructure/         # NinjaTrader integration
├── market/                     # MARKET DOMAIN - Data Processing
│   ├── data/processor.py       # Market data processing & features
│   └── microstructure/         # Advanced market analysis
├── risk/                       # RISK DOMAIN - Risk Management
│   ├── management/service.py   # Kelly criterion & dynamic risk
│   └── portfolio/manager.py    # Portfolio optimization & analytics
├── core/                       # CORE SYSTEM - Integration
│   ├── main.py                 # Main orchestrator entry point
│   └── config.py               # Environment-aware configuration
└── shared/                     # SHARED KERNEL - Common types
    └── types.py                # Domain interfaces & data types
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

## 🧠 AI Subsystems

### 1. DNA Subsystem
- **16-base genetic encoding** of market patterns
- **Genetic breeding algorithms** for pattern evolution
- **Mutation and selection** for adaptation

### 2. Temporal Subsystem
- **FFT-based cycle detection** with multiple timeframes
- **Seasonal and lunar pattern analysis**
- **Cycle interference modeling**

### 3. Immune Subsystem
- **Evolving antibody system** for threat detection
- **T-cell memory** for pattern recognition
- **Autoimmune prevention** mechanisms

### 4. Microstructure Subsystem
- **Smart money detection** algorithms
- **Order flow analysis** and regime classification
- **Market depth and liquidity** assessment

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
- **Bounded Contexts**: Each domain is self-contained
- **Domain Services**: Complex business logic encapsulation
- **Factory Patterns**: Clean object creation
- **Dependency Injection**: Loose coupling between domains

### Code Quality
- Clean Architecture patterns
- Comprehensive error handling
- Extensive logging and monitoring
- Type hints and documentation

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