# Research-Aligned Futures Trading System

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![NinjaTrader 8](https://img.shields.io/badge/NinjaTrader-8.0+-orange.svg)](https://ninjatrader.com/)

A sophisticated algorithmic trading system that combines academic research-backed technical analysis with machine learning for futures market trading. Built on the principle that **simpler systems often outperform complex ones**, this project implements a multi-timeframe approach using proven indicators with logistic regression ML models.

## 🎯 Project Overview

This trading system bridges NinjaTrader 8 (C#) with Python-based machine learning through TCP communication, implementing research-validated strategies that prioritize signal quality over quantity. The system achieves institutional-grade architecture while maintaining simplicity and interpretability.

### Key Features

- **Multi-Timeframe Analysis**: 15-minute trend identification with 5-minute entry signals
- **Research-Backed Indicators**: RSI, Bollinger Bands, EMA, SMA, and Volume analysis
- **Machine Learning Integration**: Logistic regression with normalized feature engineering
- **Quality-Focused Signaling**: 60% minimum confidence threshold with signal quality assessment
- **Professional Risk Management**: 2% risk per trade with 2:1 reward-to-risk ratios
- **Real-Time TCP Bridge**: Low-latency communication between NinjaTrader and Python
- **Modular Architecture**: Clean separation of concerns for maintainability and testing

## 🏗️ System Architecture

```
┌─────────────────┐    TCP Bridge     ┌─────────────────┐
│   NinjaTrader   │◄─────────────────►│  Python System  │
│   (C# Strategy) │   Port 5556/5557  │   (ML Engine)    │
└─────────────────┘                   └─────────────────┘
         │                                       │
         ▼                                       ▼
┌─────────────────┐                   ┌─────────────────┐
│  Market Data    │                   │ Feature Engine  │
│  • 15m Prices   │                   │ • RSI           │
│  • 5m Prices    │                   │ • Bollinger     │
│  • Volume       │                   │ • EMA/SMA       │
│  • Indicators   │                   │ • Vol Ratios    │
└─────────────────┘                   └─────────────────┘
                                               │
                                               ▼
                                    ┌─────────────────┐
                                    │ ML Model        │
                                    │ • Logistic Reg  │
                                    │ • Confidence    │
                                    │ • Quality Score │
                                    └─────────────────┘
                                               │
                                               ▼
                                    ┌─────────────────┐
                                    │ Trading Signals │
                                    │ • Buy/Sell/Hold │
                                    │ • Position Size │
                                    │ • Risk Controls │
                                    └─────────────────┘
```

## 📈 Performance Characteristics

Based on academic research and professional benchmarking:

- **Target Sharpe Ratio**: 0.8-1.2 (intermediate-level performance)
- **Expected Annual Returns**: 15-35% after transaction costs
- **Maximum Drawdown**: <20% with proper risk management
- **Win Rate Target**: 55-60% with 2:1 reward-to-risk ratio
- **Signal Frequency**: 8-12 trades per month for quality-focused approach

## 🚀 Quick Start

### Prerequisites

- **NinjaTrader 8**: Professional license with Strategy Development Suite
- **Python 3.8+**: With pip package manager
- **Windows OS**: Required for NinjaTrader integration
- **Futures Trading Account**: With real-time data feed

### Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/research-trading-system.git
   cd research-trading-system
   ```

2. **Install Python Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install NinjaScript Strategy**
   - Copy `ResearchStrategy.cs` to NinjaTrader's Strategy folder
   - Compile through NinjaTrader's NinjaScript Editor
   - Ensure all references compile successfully

4. **Verify Directory Structure**
   ```
   research-trading-system/
   ├── main.py                    # Entry point
   ├── config.py                  # Configuration settings
   ├── requirements.txt           # Python dependencies
   ├── ResearchStrategy.cs        # NinjaTrader strategy
   ├── core/
   │   └── trading_system.py      # Main system logic
   ├── communication/
   │   └── tcp_bridge.py          # TCP communication layer
   ├── features/
   │   └── feature_extractor.py   # Technical indicator features
   ├── indicators/
   │   └── research_indicators.py # Technical analysis functions
   ├── models/
   │   └── logistic_model.py      # Machine learning model
   └── models/                    # Model persistence directory
   ```

### Configuration

1. **Edit Configuration** (`config.py`)
   ```python
   # Core indicator settings (research-optimized)
   RSI_PERIOD = 14
   BB_PERIOD = 20
   EMA_PERIOD = 20
   SMA_PERIOD = 50
   
   # ML parameters
   CONFIDENCE_THRESHOLD = 0.6
   ML_LOOKBACK = 50
   
   # TCP settings (default ports)
   FEATURE_PORT = 5556
   SIGNAL_PORT = 5557
   ```

2. **NinjaTrader Strategy Parameters**
   - Risk Percent: 0.02 (2% per trade)
   - Stop Loss Ticks: 20
   - Take Profit Ticks: 40 (2:1 ratio)
   - Min Confidence: 0.6
   - Max Position Size: 2

### Running the System

1. **Start Python ML Engine**
   ```bash
   python main.py
   ```
   Output should show:
   ```
   RESEARCH-ALIGNED TRADING SYSTEM
   Features: RSI + Bollinger Bands + EMA + SMA + Volume
   Timeframes: 15min (trend) + 5min (entry)
   ML Model: Logistic Regression
   Waiting for NinjaTrader connection...
   ```

2. **Launch NinjaTrader Strategy**
   - Open NinjaTrader 8
   - Navigate to Control Center → Strategies → New Strategy
   - Select "ResearchStrategy"
   - Configure instrument (e.g., ES 12-25 for E-mini S&P 500)
   - Set data series: Primary 1-minute, Secondary 15-minute, Tertiary 5-minute
   - Enable strategy on live account or simulation

3. **Verify Connection**
   Python console should display:
   ```
   TCP Bridge initialized on localhost:5556 (features) and localhost:5557 (signals)
   Feature connection established from ('127.0.0.1', xxxxx)
   Signal connection established from ('127.0.0.1', xxxxx)
   NinjaTrader connected successfully
   ```

## 🔧 System Components

### Core Components

#### 1. Trading System (`core/trading_system.py`)
Central orchestrator managing data flow, model training, and signal generation.

#### 2. TCP Bridge (`communication/tcp_bridge.py`)
High-performance communication layer handling real-time data exchange between NinjaTrader and Python.

#### 3. Feature Extractor (`features/feature_extractor.py`)
Converts raw market data into normalized features for ML model consumption.

#### 4. Logistic Model (`models/logistic_model.py`)
Implements logistic regression with confidence scoring and quality assessment.

### Technical Indicators

#### Research Indicators (`indicators/research_indicators.py`)
- **RSI(14)**: Momentum oscillator for overbought/oversold conditions
- **Bollinger Bands(20,2)**: Volatility-based support/resistance levels
- **EMA(20)/SMA(50)**: Trend identification and crossover signals
- **Volume Ratio**: Current volume vs. 20-period average for confirmation

### NinjaTrader Integration

#### ResearchStrategy.cs Features
- Multi-timeframe data collection (1m, 5m, 15m)
- Real-time TCP communication with Python
- Professional risk management with stop-loss/take-profit orders
- Position sizing based on ML confidence levels
- Visual signal indicators on charts

## 📊 Signal Generation Process

1. **Data Collection**: NinjaTrader collects tick-by-tick data across multiple timeframes
2. **Feature Extraction**: Python calculates normalized technical indicators
3. **ML Prediction**: Logistic regression generates buy/sell/hold signals with confidence scores
4. **Quality Assessment**: Signals rated as excellent/good/fair/poor based on confidence
5. **Risk Management**: Position sizing adjusted based on signal quality and confidence
6. **Order Execution**: NinjaTrader executes trades with predefined stop-loss/take-profit levels

## 🧪 Testing and Validation

### Backtesting
```bash
# Run historical analysis (implement your backtesting framework)
python -m pytest tests/ -v
```

### Performance Metrics
Monitor these key performance indicators:
- **Sharpe Ratio**: Risk-adjusted returns (target: >1.0)
- **Maximum Drawdown**: Largest peak-to-trough decline (target: <20%)
- **Profit Factor**: Gross profits / Gross losses (target: >1.5)
- **Win Rate**: Percentage of profitable trades (target: >55%)

### Strategy Validation
- **Walk-Forward Analysis**: Test on rolling time windows
- **Out-of-Sample Testing**: Reserve 20% of data for final validation
- **Monte Carlo Simulation**: Assess strategy robustness across market conditions

## 🚨 Risk Management

### Built-in Safeguards
- **Position Limits**: Maximum 2 contracts per signal (configurable)
- **Daily Loss Limits**: Automatic shutdown on excessive losses
- **Connection Monitoring**: Automatic reconnection on TCP failures
- **Signal Quality Filtering**: Only execute high-confidence signals

### Risk Controls
```python
# Example risk parameters
RISK_PERCENT = 0.02        # 2% of account per trade
MAX_POSITION_SIZE = 2      # Maximum contracts
STOP_LOSS_TICKS = 20       # 20 tick stop loss
TAKE_PROFIT_TICKS = 40     # 40 tick take profit (2:1 ratio)
MIN_CONFIDENCE = 0.6       # 60% minimum ML confidence
```

## 📚 Research Foundation

This system is built on extensive academic research demonstrating that simple, well-designed algorithms often outperform complex alternatives. Key research findings:

- **Quantopian Study**: Analysis of 888 algorithms showed backtest Sharpe ratios had virtually no predictive value (R² < 0.025)
- **Cryptocurrency Research**: Single indicators outperformed complex combinations in 17 of 20 test cases
- **Professional Validation**: Simple fixed stops and 2:1 reward-to-risk ratios align with institutional best practices

See `research.txt` for detailed academic references and methodology.

## 🤝 Contributing

We welcome contributions from the trading and quantitative finance community!

### Development Setup
1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Install development dependencies: `pip install -r requirements-dev.txt`
4. Run tests: `python -m pytest tests/`
5. Commit changes: `git commit -m 'Add amazing feature'`
6. Push to branch: `git push origin feature/amazing-feature`
7. Open a Pull Request

### Contribution Guidelines
- **Code Quality**: Follow PEP 8 style guidelines
- **Documentation**: Update docstrings and README for new features
- **Testing**: Add unit tests for new functionality
- **Performance**: Benchmark changes that affect signal generation speed

### Areas for Contribution
- Alternative ML models (Random Forest, Gradient Boosting)
- Additional technical indicators
- Enhanced risk management features
- Performance optimization
- Extended backtesting frameworks

## 📁 Project Structure

```
research-trading-system/
├── 📄 README.md                     # This file
├── 📄 requirements.txt              # Python dependencies
├── 📄 config.py                     # System configuration
├── 📄 main.py                       # Application entry point
├── 📄 ResearchStrategy.cs           # NinjaTrader strategy
├── 📄 research.txt                  # Academic research foundation
├── 📄 .gitignore                    # Git ignore patterns
├── 📁 communication/
│   ├── 📄 __init__.py
│   └── 📄 tcp_bridge.py             # TCP communication layer
├── 📁 core/
│   ├── 📄 __init__.py
│   └── 📄 trading_system.py         # Main system orchestrator
├── 📁 features/
│   ├── 📄 __init__.py
│   └── 📄 feature_extractor.py      # Technical indicator features
├── 📁 indicators/
│   ├── 📄 __init__.py
│   └── 📄 research_indicators.py    # Technical analysis functions
├── 📁 models/
│   ├── 📄 __init__.py
│   ├── 📄 logistic_model.py         # ML model implementation
│   ├── 📄 logistic_model.joblib     # Saved model (auto-generated)
│   └── 📄 feature_scaler.joblib     # Feature scaler (auto-generated)
└── 📁 tests/                        # Unit tests (future implementation)
    ├── 📄 __init__.py
    ├── 📄 test_indicators.py
    ├── 📄 test_features.py
    └── 📄 test_models.py
```

## ⚠️ Disclaimers

**TRADING RISK DISCLOSURE**: Trading futures carries substantial risk of loss and is not suitable for all investors. Past performance is not indicative of future results. This software is provided for educational and research purposes only.

**SOFTWARE LIABILITY**: This software is provided "as is" without warranty of any kind. Users assume full responsibility for trading decisions and outcomes.

**REGULATORY COMPLIANCE**: Ensure compliance with all applicable financial regulations in your jurisdiction before using this system for live trading.

## 🎯 Future Roadmap

- **Enhanced ML Models**: Integration of ensemble methods and deep learning
- **Alternative Data**: Satellite imagery and sentiment analysis for improved signals
- **Multi-Asset Support**: Extension to forex, commodities, and cryptocurrency futures
- **Cloud Deployment**: AWS/Azure hosting for institutional scalability
- **Advanced Analytics**: Real-time performance dashboards and trade analysis

## 📞 Support and Community

- **Issues**: Report bugs and feature requests via GitHub Issues
- **Discussions**: Join community discussions in GitHub Discussions
- **Documentation**: Comprehensive docs available in `/docs` folder
- **Wiki**: Strategy guides and tutorials in the GitHub Wiki

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Academic researchers whose work on algorithmic trading simplification forms the foundation of this project
- The NinjaTrader community for platform integration expertise
- Open-source contributors to scikit-learn and numpy for ML infrastructure
- Professional traders who shared insights on institutional-grade risk management

---

**Built with 💡 by the Quantitative Trading Community**

*"In the world of algorithmic trading, elegance lies not in complexity, but in the sophisticated simplicity that consistently profits."*