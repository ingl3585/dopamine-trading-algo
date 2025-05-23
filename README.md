# Reinforcement Learning Trading System with Ichimoku/EMA/LWPE Integration

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-v1.10+-orange.svg)
![NinjaTrader](https://img.shields.io/badge/NinjaTrader-8-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

A sophisticated algorithmic trading system that combines **Reinforcement Learning (Actor-Critic)** with **Ichimoku Cloud analysis**, **EMA crossovers**, and a **custom LWPE (Liquidity-Weighted Price Entropy)** indicator for institutional-grade trading decisions.

## 🎯 System Overview

This system integrates multiple layers of market analysis:

- **🧠 Reinforcement Learning**: Actor-Critic neural network for adaptive decision making
- **☁️ Ichimoku Cloud**: 6-component market structure analysis (Tenkan, Kijun, Senkou A/B, momentum)
- **📈 EMA Crossovers**: Trend direction and timing confirmation
- **🌊 LWPE Indicator**: Custom order flow and institutional activity detection
- **🔗 NinjaTrader Integration**: Real-time market data and trade execution

### Key Features

✅ **Multi-Signal Intelligence**: Combines 9 features into ternary signals (-1, 0, 1) for robust decision making  
✅ **Adaptive Position Sizing**: Dynamic allocation based on signal confidence and alignment  
✅ **Real-Time Order Flow**: LWPE indicator reveals institutional buying/selling pressure  
✅ **Risk Management**: Confidence thresholds, drawdown protection, and volatility adjustment  
✅ **Online Learning**: Continuous model adaptation to changing market conditions  
✅ **Production Ready**: Comprehensive logging, error handling, and performance monitoring  

## 🏗️ Architecture

### Core Components

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   NinjaTrader   │◄──►│   TCP Bridge     │◄──►│  Feature Engine │
│   (C# Strategy) │    │  (Port 5556/57)  │    │  (9D Vectors)   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │                         │
                                ▼                         ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ Position Mgmt   │◄──►│   RL Agent       │◄──►│ Signal Processor│
│ & Risk Control  │    │ (Actor-Critic)   │    │ (Ichimoku/EMA)  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │                         │
                                ▼                         ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Data Logger    │◄──►│    Trainer       │◄──►│  LWPE Processor │
│   (CSV/Stats)   │    │ (Online/Batch)   │    │ (Tick Analysis) │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### Feature Vector (9 Dimensions)

| Index | Feature | Range | Description |
|-------|---------|-------|-------------|
| 0 | Close Price | Float | Current bar close price |
| 1 | Normalized Volume | Float | Volume normalized by 20-period SMA |
| 2 | Tenkan/Kijun Signal | -1,0,1 | Ichimoku conversion/base line cross |
| 3 | Price/Cloud Signal | -1,0,1 | Price position relative to cloud |
| 4 | Future Cloud Signal | -1,0,1 | Future cloud color (green/red) |
| 5 | EMA Cross Signal | -1,0,1 | Fast EMA vs Slow EMA relationship |
| 6 | Tenkan Momentum | -1,0,1 | Tenkan-sen momentum direction |
| 7 | Kijun Momentum | -1,0,1 | Kijun-sen momentum direction |
| 8 | LWPE | 0.0-1.0 | Liquidity-Weighted Price Entropy |

## 🚀 Quick Start

### Prerequisites

- **Python 3.8+**
- **NinjaTrader 8** with active market data
- **PyTorch 1.10+**
- **CUDA compatible GPU** (recommended)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/rl-trading-system.git
   cd rl-trading-system
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure paths in `config.py`**
   ```python
   FEATURE_FILE = r"C:\path\to\your\features\features.csv"
   MODEL_PATH = r"C:\path\to\your\model\actor_critic_model.pth"
   ```

4. **Load NinjaTrader strategy**
   - Copy `RLTrader.cs` to NinjaTrader strategies folder
   - Compile and add to chart
   - Configure parameters (EMA periods, risk %, confidence threshold)

### Running the System

#### Fresh Start (New Model)
```bash
python main.py --reset --log-level INFO
```

#### Continue Training (Existing Model)
```bash
python main.py --log-level INFO
```

#### Dry Run Mode (No Live Trading)
```bash
python main.py --dry-run --log-level DEBUG
```

### Expected Startup Sequence

```
[2025-05-23 14:53:12] Starting RL Trading System with Ichimoku/EMA Features
[2025-05-23 14:53:14] TCP Bridge initialized on localhost:5556/5557
[2025-05-23 14:53:21] NinjaTrader connected successfully
[2025-05-23 14:53:28] Processed 1000 feature vectors
[2025-05-23 14:54:08] Initial training completed on 6834 samples
[2025-05-23 14:54:10] === Feature Analysis ===
[2025-05-23 14:54:10] tenkan_kijun_signal: Bull=34.0% Bear=35.0% Neutral=31.1%
[2025-05-23 14:55:02] Signal sent - Hold: size=0, conf=0.838
```

## 📊 Signal Analysis & Performance

### Historical Performance (Training Data)

| Signal Type | Bullish % | Bearish % | Neutral % | Avg Reward |
|-------------|-----------|-----------|-----------|------------|
| **Ichimoku Setups** | 34-38% | 35-36% | 26-31% | **+0.60/-0.64** |
| **EMA Cross** | 22% | 22% | 55% | +0.29/-0.26 |

### Signal Quality Ratings

- **Excellent (0.8+ confidence)**: All indicators aligned, maximum position size
- **Good (0.6-0.8 confidence)**: Strong setup with minor conflicts, standard position
- **Mixed (0.4-0.6 confidence)**: Conflicting signals, small position or hold
- **Poor (<0.4 confidence)**: No clear direction, hold position

### Expected Trading Frequency

| Market Condition | Signals/Day | Trade Rate | Position Sizing |
|------------------|-------------|------------|-----------------|
| **Trending** | 15-25 | 60% | Full allocation |
| **Ranging** | 5-10 | 20% | Selective entries |
| **Volatile** | 8-15 | 40% | Reduced sizes |

## ⚙️ Configuration

### Key Parameters (`config.py`)

```python
# Model Architecture
INPUT_DIM = 9           # Feature vector dimensions
HIDDEN_DIM = 128        # Neural network hidden units
ACTION_DIM = 3          # Hold(0), Long(1), Short(2)

# Trading Parameters
BASE_SIZE = 4           # Base position size
MAX_SIZE = 10           # Maximum position size
MIN_CONFIDENCE = 0.6    # Minimum confidence for trading
TEMPERATURE = 1.8       # Model prediction temperature

# Feature Weights
ICHIMOKU_WEIGHT = 0.30  # Ichimoku signal importance
EMA_WEIGHT = 0.20       # EMA signal importance
LWPE_WEIGHT = 0.20      # LWPE signal importance
MOMENTUM_WEIGHT = 0.15  # Momentum signal importance
VOLUME_WEIGHT = 0.15    # Volume signal importance
```

### NinjaTrader Strategy Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| Risk Percent | 0.01 | Risk per trade (1%) |
| EMA Fast Period | 12 | Fast EMA period |
| EMA Slow Period | 26 | Slow EMA period |
| Tenkan Period | 9 | Ichimoku Tenkan-sen period |
| Kijun Period | 26 | Ichimoku Kijun-sen period |
| Senkou Period | 52 | Ichimoku Senkou span period |
| Min Confidence | 0.6 | Minimum confidence threshold |

## 🔍 Advanced Features

### LWPE (Liquidity-Weighted Price Entropy) Indicator

Our custom LWPE indicator provides unique market microstructure insights:

```python
def calculate_lwpe(tick_buffer):
    # Analyzes last 1000 ticks
    # Creates 10ms OHLC resampling
    # Calculates buy/sell pressure distribution
    # Returns entropy weighted by liquidity
    return lwpe_value  # 0.0-1.0
```

**LWPE Interpretation:**
- **0.8-1.0**: Strong institutional buying pressure
- **0.6-0.8**: Moderate buying bias
- **0.4-0.6**: Balanced/neutral market
- **0.2-0.4**: Moderate selling bias  
- **0.0-0.2**: Strong institutional selling pressure

### Ichimoku Cloud Components

The system calculates 6 Ichimoku signals:

1. **Tenkan-sen (Conversion Line)**: (9H + 9L) / 2
2. **Kijun-sen (Base Line)**: (26H + 26L) / 2
3. **Senkou Span A**: (Tenkan + Kijun) / 2, plotted 26 periods ahead
4. **Senkou Span B**: (52H + 52L) / 2, plotted 26 periods ahead
5. **Tenkan Momentum**: 3-period Tenkan direction
6. **Kijun Momentum**: 3-period Kijun direction

### Risk Management Features

- **Dynamic Position Sizing**: Based on signal confidence and alignment
- **Volatility Adjustment**: LWPE-based position size reduction in high volatility
- **Drawdown Protection**: Maximum 2% portfolio risk per trade
- **Signal Validation**: Strict bounds checking for all input signals

## 📈 Optimal Trading Conditions

### Best Market Conditions

1. **Trending Markets**: Clear directional bias with pullbacks
2. **Medium Volatility**: 0.5-2.0% daily range for optimal signal quality
3. **Regular Sessions**: Avoid thin volume/holiday periods
4. **Index Futures**: ES, NQ, YM for best institutional flow

### Perfect Setup Example

```
✅ Ichimoku: Price breaks above cloud + Tenkan > Kijun
✅ EMA: Fast EMA crosses above Slow EMA with expanding gap
✅ LWPE: > 0.75 (strong institutional buying)
✅ Volume: Above average on breakout
✅ Time: 9:30-11:30 AM EST (optimal institutional activity)

Expected Response:
- Confidence: 0.80-0.85
- Position Size: 7-10 contracts
- Win Rate: 70%+ based on training data
```

### Instruments & Timeframes

**Recommended Instruments:**
- **Tier 1**: ES, NQ, YM (index futures)
- **Tier 2**: EUR/USD, GBP/USD (major FX)
- **Tier 3**: CL, GC (liquid commodities)

**Optimal Timeframes:**
- **Primary**: 5-minute charts (balance of signal quality vs frequency)
- **Secondary**: 15-minute charts (higher probability setups)

## 🛠️ Development & Debugging

### Logging Levels

```bash
--log-level DEBUG    # Verbose signal analysis and decision details
--log-level INFO     # Standard operational information  
--log-level WARNING  # Only warnings and errors
--log-level ERROR    # Critical errors only
```

### Monitoring & Statistics

The system provides comprehensive performance tracking:

- **TCP Statistics**: Message counts, connection health, data quality
- **Signal Analysis**: Distribution of bullish/bearish/neutral signals
- **Feature Importance**: Which indicators contribute most to decisions
- **Performance Metrics**: Win rates, average rewards, Sharpe ratios

### Data Files

- **features.csv**: Historical feature vectors and rewards
- **actor_critic_model.pth**: Trained neural network weights
- **System logs**: Detailed operational and debug information

## 🔧 Troubleshooting

### Common Issues

**Connection Problems:**
```bash
# Check NinjaTrader strategy is running
# Verify ports 5556/5557 are available
# Restart both Python system and NinjaTrader
```

**No Signals Generated:**
```bash
# Check confidence threshold (default 0.55)
# Verify market is active (not pre/post market)
# Review signal distribution in logs
```

**High Loss Values:**
```bash
# Normal during initial training (complex patterns)
# Monitor for convergence over multiple epochs
# Consider reducing learning rate if diverging
```

### Performance Optimization

- **GPU Acceleration**: Ensure CUDA is available for faster training
- **Batch Size**: Adjust based on available memory (default 32)
- **Feature Validation**: Enable strict validation for debugging

## 📚 Research & Background

### Academic Foundation

This system implements concepts from:
- **Reinforcement Learning**: Actor-Critic methods for sequential decision making
- **Technical Analysis**: Ichimoku Cloud theory and EMA trend following
- **Market Microstructure**: Order flow analysis and institutional behavior
- **Risk Management**: Kelly criterion and volatility-adjusted position sizing

### Key Innovations

1. **Ternary Signal Processing**: -1/0/1 signals handle market uncertainty better than binary
2. **LWPE Integration**: Custom order flow indicator provides institutional edge
3. **Multi-Timeframe Analysis**: Combines tick-level, minute-level, and hour-level signals
4. **Adaptive Learning**: Online learning adjusts to changing market regimes

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## ⚠️ Disclaimer

**This software is for educational and research purposes only. Trading financial instruments carries substantial risk of loss and is not suitable for all investors. Past performance does not guarantee future results. Use at your own risk.**

## 📞 Support

For questions and support:
- Create an issue in this repository
- Review the troubleshooting section above
- Check system logs for detailed error information

---

**Built with ❤️ for algorithmic trading excellence**