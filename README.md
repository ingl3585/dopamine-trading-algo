```
██████╗  ██████╗ ██████╗  █████╗ ███╗   ███╗██╗███╗   ██╗███████╗
██╔══██╗██╔═══██╗██╔══██╗██╔══██╗████╗ ████║██║████╗  ██║██╔════╝
██║  ██║██║   ██║██████╔╝███████║██╔████╔██║██║██╔██╗ ██║█████╗  
██║  ██║██║   ██║██╔═══╝ ██╔══██║██║╚██╔╝██║██║██║╚██╗██║██╔══╝  
██████╔╝╚██████╔╝██║     ██║  ██║██║ ╚═╝ ██║██║██║ ╚████║███████╗
╚═════╝  ╚═════╝ ╚═╝     ╚═╝  ╚═╝╚═╝     ╚═╝╚═╝╚═╝  ╚═══╝╚══════╝
```

# **Dopamine Trading System**

An advanced AI-powered trading system that uses neuromorphic computing principles and 5 intelligent subsystems to make sophisticated trading decisions. The system integrates with NinjaTrader via TCP bridge for live trading execution.

## 🧠 Core AI Subsystems

The trading system employs 5 specialized AI subsystems that work together as intelligent tools:

- **🧬 DNA Subsystem**: Pattern recognition and sequence analysis for market behavior encoding
- **⏰ Temporal Subsystem**: Cycle detection and time-based pattern analysis  
- **🛡️ Immune Subsystem**: Risk assessment and anomaly detection for market protection
- **📊 Microstructure Subsystem**: Market regime analysis and order flow assessment
- **💫 Dopamine Subsystem**: Reward optimization and learning enhancement using neuromorphic principles

## 🏗️ System Architecture

### Entry Point
- **main.py** → **TradingSystemOrchestrator** (primary system coordinator)

### Core Infrastructure
- **IntelligenceEngine**: Coordinates all 5 AI subsystems
- **TradingService**: Handles trade execution through NinjaTrader repository
- **TCPBridge**: Real-time communication with NinjaTrader (ports 5556/5557)
- **RiskManager**: Position sizing and risk assessment
- **MarketDataProcessor**: Real-time market data processing and validation

### Support Components
- **AnalysisTriggerManager**: Timeframe-based analysis coordination
- **PersonalityIntegrationManager**: AI commentary and personality system
- **ConfigurationManager**: Type-safe configuration management
- **SystemStateManager**: Unified state persistence and recovery

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- NinjaTrader 8 with custom ResearchStrategy.cs addon
- Required Python packages (see requirements.txt)

### Installation
```bash
# Clone the repository
git clone [repository-url]
cd dopamine_trading_algo

# Install dependencies
pip install -r requirements.txt

# Configure system
cp config/development.json config/local.json
# Edit config/local.json with your settings
```

### Running the System
```bash
# Start the trading system
python main.py

# The system will:
# 1. Start TCP server on ports 5556/5557
# 2. Wait for NinjaTrader connection
# 3. Bootstrap AI subsystems with historical data
# 4. Begin live trading analysis and execution
```

## 📊 Trading Decision Process

1. **Market Data Ingestion**: Real-time OHLCV data from NinjaTrader
2. **AI Subsystem Analysis**: Each subsystem analyzes market from its specialized perspective
3. **Signal Aggregation**: Weighted combination of all 5 subsystem signals (0.2 each)
4. **Confidence Assessment**: Dynamic confidence calculation (0.2-0.9 range)
5. **Risk Evaluation**: Position sizing and risk assessment
6. **Trade Execution**: Orders sent to NinjaTrader via TCP bridge
7. **Learning Update**: AI subsystems learn from trade outcomes

## 🔧 Configuration

### Core Settings (config/development.json)
```json
{
  "tcp_data_port": 5556,
  "tcp_signal_port": 5557,
  "trading_interval_seconds": 60,
  "min_trade_interval_seconds": 300,
  "historical_data_timeout": 120,
  "confidence_threshold": 0.4,
  "consensus_threshold": 0.3,
  "risk_threshold": 0.8
}
```

### Personality Configuration (config/personality_config.json)
- AI commentary settings
- Emotional response parameters
- Learning adaptation rates

## 📈 Key Features

### Advanced AI Integration
- **5 Specialized Subsystems**: Each brings unique market analysis perspective
- **Dynamic Confidence**: Real-time confidence assessment (not fixed values)
- **Consensus Analysis**: Agreement measurement between subsystems
- **Adaptive Learning**: Continuous improvement from trading outcomes

### Risk Management
- **Position Sizing**: Kelly Criterion and confidence-based sizing
- **Risk Assessment**: Multi-factor risk evaluation
- **Drawdown Protection**: Automatic position reduction during losses
- **Margin Management**: Real-time margin utilization monitoring

### Real-time Integration
- **TCP Bridge**: Direct NinjaTrader communication
- **Live Data Processing**: Real-time OHLCV data handling
- **Order Execution**: Immediate trade execution with confirmation
- **Account Monitoring**: Real-time P&L and position tracking

## 🛠️ System Status

### Current Version: **Modernized & Optimized**
- ✅ All 5 AI subsystems integrated and contributing to decisions
- ✅ Dynamic confidence calculations (0.2-0.9 range)
- ✅ Real NinjaTrader integration (not simulation)
- ✅ Responsive trading (confidence threshold: 0.4)
- ✅ TCP bridge fully functional

### Ongoing: **Codebase Cleanup Initiative**
- 🔄 Removing redundant code and over-engineering
- 🔄 Simplifying architecture while preserving functionality
- 🔄 Reducing ~3,000 lines of redundant code (40% reduction)
- See **CLAUDE.md** for detailed cleanup progress

## 📋 Logging & Monitoring

### Trading Decisions
```
Price: $22497.50 | Signal: 0.125 | Conf: 0.66 | Consensus: 0.66 | Risk: 0.29 | Action: BUY
```

### Trade Execution
```
Trade 1 executed: buy size=1.0 pnl=0.00
Signal sent: BUY 1.0 @ 22497.50 (Conf: 0.66)
```

### System Health
```
TCP server started, waiting for historical data...
Live trading loop started...
Position sync: NinjaTrader reports 1 contracts
```

## 🔬 Technical Details

### AI Signal Processing
- **Weight Distribution**: Each of 5 subsystems contributes 20% to overall signal
- **Confidence Calculation**: Individual subsystem confidence + volatility-based adjustment
- **Consensus Measurement**: Agreement analysis between all subsystem signals
- **Signal Range**: -1.0 to +1.0 (normalized for consistent interpretation)

### Performance Optimization
- **Memory Efficient**: Circular buffers for historical data
- **CPU Optimized**: Vectorized calculations where possible
- **Network Efficient**: Binary TCP protocol for NinjaTrader communication
- **Startup Optimized**: Parallel component initialization

## 📚 Documentation

- **CLAUDE.md**: Detailed development progress and cleanup tracking
- **ResearchStrategy.cs**: NinjaTrader addon for data/signal communication
- **config/**: Configuration templates and examples
- **logs/**: System logs and trading records

## ⚠️ Important Notes

### Production Use
- **Paper Trading**: Test thoroughly before live trading
- **Risk Management**: Start with small position sizes
- **Monitoring**: Always monitor system performance and P&L
- **Backup**: Maintain configuration and state backups

### Development
- **Testing**: Run system tests after any modifications
- **Configuration**: Use development.json for testing
- **Logging**: Monitor logs/ directory for system health
- **Updates**: Follow cleanup progress in CLAUDE.md

---

*For detailed development progress and system architecture evolution, see **CLAUDE.md***