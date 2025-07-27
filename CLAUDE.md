# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Setup and Installation
```bash
# Install dependencies
pip install -r requirements.txt

# Reset workspace (clears models/, data/, logs/ directories)
python main.py --reset
```

### Running the System
```bash
# Start the trading system
python main.py

# Run from src module
python -m src
```

### Testing
```bash
# Run tests
pytest

# Run tests with coverage
pytest --cov

# Run async tests
pytest-asyncio
```

### Type Checking and Linting
```bash
# Type checking (optional dependency)
mypy src/

# Note: No specific linting commands found - add as needed
```

## Architecture Overview

### Core System Design
The Dopamine Trading System is a bio-inspired algorithmic trading platform organized around several key architectural patterns:

**Event-Driven Architecture**: The system uses `src/core/event_bus.py` as the central nervous system, coordinating communication between all components through event publishing/subscription.

**Dependency Injection**: `src/core/dependency_registry.py` manages service registration and dependency resolution across the system.

**State Management**: `src/core/state_coordinator.py` and `src/core/system_state_manager.py` handle centralized state management with persistence to `data/` directory.

### Primary System Components

**TradingSystem (`src/core/trading_system.py`)**: Main orchestrator that coordinates all subsystems. Entry point for the entire system.

**Intelligence Engine (`src/intelligence/intelligence_engine.py`)**: Bio-inspired decision-making system with multiple specialized subsystems:
- DNA Subsystem: Pattern recognition and genetic algorithms
- Enhanced Dopamine Subsystem: Reward-based learning
- Immune Subsystem: Anomaly detection and system protection
- Temporal Subsystem: Time-series analysis and trend detection

**Trading Agent (`src/agent/trading_agent_v2.py`)**: Neural network-based trading decision engine with:
- Multi-objective optimization
- Real-time adaptation
- Confidence scoring
- Experience management

**Risk Management (`src/risk/`)**: Multi-layered risk control with advanced risk manager and portfolio-level risk calculation.

**Communication (`src/communication/tcp_bridge.py`)**: TCP server for external data feeds and signal communication (ports 5556-5557).

### Key Data Flow
1. Market data flows through `MarketDataProcessor`
2. Intelligence subsystems analyze and generate signals
3. Trading agent processes signals and makes decisions
4. Risk manager validates decisions
5. Portfolio manager executes trades
6. State manager persists system state

### Configuration
- Development config: `config/development.json`
- Production config: `config/production.json`
- TCP ports: 5556 (data), 5557 (signals)
- Key parameters: dopamine_sensitivity, momentum_factor, max_position_size

### Neural Network Components
Located in `src/neural/`: Adaptive networks, dynamic pruning, uncertainty estimation, and specialized trading networks with PyTorch integration.

### Directory Structure
- `src/agent/`: Trading decision components
- `src/core/`: System orchestration and infrastructure
- `src/intelligence/`: Bio-inspired analysis subsystems
- `src/neural/`: Neural network implementations
- `src/portfolio/`: Portfolio and position management
- `src/risk/`: Risk management and calculation
- `src/communication/`: External system interfaces
- `data/`: Runtime state and system snapshots
- `logs/`: System logging output
- `models/`: Trained model storage