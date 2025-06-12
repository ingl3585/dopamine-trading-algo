# pattern_learning.py

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import joblib
import os

log = logging.getLogger(__name__)

@dataclass
class TradeRecord:
    """Simple trade record"""
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    direction: str  # 'long' or 'short'
    confidence: float
    time_of_day: int  # hour 0-23
    day_of_week: int  # 0=Monday, 6=Sunday
    pnl: float
    duration_minutes: int
    exit_reason: str

@dataclass
class PatternInsight:
    """Discovered pattern"""
    name: str
    description: str
    sample_size: int
    impact: float
    action: str

class PatternLearningSystem:
    """Simple pattern discovery from trades"""
    
    def __init__(self, min_samples: int = 15):
        self.min_samples = min_samples
        self.trades: List[TradeRecord] = []
        self.insights: List[PatternInsight] = []
        
        # Create data directory
        os.makedirs('data', exist_ok=True)
        
        log.info(f"Pattern learning initialized (min samples: {min_samples})")
    
    def add_trade(self, trade: TradeRecord):
        """Add completed trade"""
        self.trades.append(trade)
        self._save_data()
        
        # Analyze every 10 trades
        if len(self.trades) >= self.min_samples and len(self.trades) % 10 == 0:
            self.analyze_patterns()
    
    def analyze_patterns(self):
        """Find simple patterns"""
        if len(self.trades) < self.min_samples:
            return
        
        log.info(f"Analyzing {len(self.trades)} trades for patterns...")
        
        df = pd.DataFrame([{
            'hour': t.time_of_day,
            'day': t.day_of_week, 
            'confidence': t.confidence,
            'pnl': t.pnl,
            'win': t.pnl > 0,
            'duration': t.duration_minutes
        } for t in self.trades])
        
        self.insights = []
        overall_win_rate = df['win'].mean()
        
        # 1. Time of day patterns
        for hour in [9, 10, 11, 13, 14, 15]:  # Key trading hours
            hour_trades = df[df['hour'] == hour]
            if len(hour_trades) >= 5:
                hour_win_rate = hour_trades['win'].mean()
                if abs(hour_win_rate - overall_win_rate) > 0.15:  # 15% difference
                    action = "FAVOR" if hour_win_rate > overall_win_rate else "AVOID"
                    self.insights.append(PatternInsight(
                        name=f"Hour {hour}:00",
                        description=f"{hour}:00 hour shows {hour_win_rate:.1%} win rate vs {overall_win_rate:.1%} overall",
                        sample_size=len(hour_trades),
                        impact=hour_win_rate - overall_win_rate,
                        action=f"{action} trading at {hour}:00"
                    ))
        
        # 2. Day of week patterns  
        for day in range(5):  # Monday-Friday
            day_trades = df[df['day'] == day]
            if len(day_trades) >= 5:
                day_win_rate = day_trades['win'].mean()
                if abs(day_win_rate - overall_win_rate) > 0.12:  # 12% difference
                    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
                    action = "FAVOR" if day_win_rate > overall_win_rate else "AVOID"
                    self.insights.append(PatternInsight(
                        name=f"{days[day]} Trading",
                        description=f"{days[day]} shows {day_win_rate:.1%} win rate vs {overall_win_rate:.1%} overall",
                        sample_size=len(day_trades),
                        impact=day_win_rate - overall_win_rate,
                        action=f"{action} trading on {days[day]}"
                    ))
        
        # 3. Confidence threshold patterns
        for threshold in [0.65, 0.75, 0.85]:
            high_conf = df[df['confidence'] >= threshold]
            if len(high_conf) >= 8:
                high_win_rate = high_conf['win'].mean()
                if high_win_rate - overall_win_rate > 0.1:  # 10% improvement
                    self.insights.append(PatternInsight(
                        name=f"Confidence {threshold:.0%}+",
                        description=f"Signals ≥{threshold:.0%} confidence show {high_win_rate:.1%} win rate vs {overall_win_rate:.1%} overall",
                        sample_size=len(high_conf),
                        impact=high_win_rate - overall_win_rate,
                        action=f"Increase minimum confidence to {threshold:.0%}"
                    ))
        
        log.info(f"Found {len(self.insights)} actionable patterns")
        self._save_data()
    
    def get_insights(self) -> List[PatternInsight]:
        """Get current insights sorted by impact"""
        return sorted(self.insights, key=lambda x: abs(x.impact), reverse=True)
    
    def generate_report(self) -> str:
        """Generate simple report"""
        if not self.trades:
            return "No trades recorded yet."
        
        df = pd.DataFrame([{'pnl': t.pnl, 'win': t.pnl > 0} for t in self.trades])
        win_rate = df['win'].mean()
        avg_pnl = df['pnl'].mean()
        total_pnl = df['pnl'].sum()
        
        report = [
            "="*50,
            "PATTERN ANALYSIS REPORT", 
            "="*50,
            f"Total Trades: {len(self.trades)}",
            f"Win Rate: {win_rate:.1%}",
            f"Average PnL: ${avg_pnl:.2f}",
            f"Total PnL: ${total_pnl:.2f}",
            "",
            "TOP INSIGHTS:",
            "-"*20
        ]
        
        insights = self.get_insights()[:3]  # Top 3
        for i, insight in enumerate(insights, 1):
            report.extend([
                f"{i}. {insight.name}",
                f"   Impact: {insight.impact:+.1%}",
                f"   Sample: {insight.sample_size} trades", 
                f"   Action: {insight.action}",
                ""
            ])
        
        if not insights:
            report.append("No significant patterns found yet.")
        
        return "\n".join(report)
    
    def _save_data(self):
        """Save data to files"""
        try:
            # Save trades
            trade_dicts = [
                {
                    'entry_time': t.entry_time.isoformat(),
                    'exit_time': t.exit_time.isoformat(),
                    'entry_price': t.entry_price,
                    'exit_price': t.exit_price,
                    'direction': t.direction,
                    'confidence': t.confidence,
                    'time_of_day': t.time_of_day,
                    'day_of_week': t.day_of_week,
                    'pnl': t.pnl,
                    'duration_minutes': t.duration_minutes,
                    'exit_reason': t.exit_reason
                }
                for t in self.trades
            ]
            joblib.dump(trade_dicts, 'data/trades.joblib')
            
            # Save insights
            insight_dicts = [
                {
                    'name': i.name,
                    'description': i.description,
                    'sample_size': i.sample_size,
                    'impact': i.impact,
                    'action': i.action
                }
                for i in self.insights
            ]
            joblib.dump(insight_dicts, 'data/insights.joblib')
            
        except Exception as e:
            log.error(f"Save error: {e}")
    
    def load_data(self):
        """Load existing data"""
        try:
            # Load trades
            if os.path.exists('data/trades.joblib'):
                trade_dicts = joblib.load('data/trades.joblib')
                self.trades = [
                    TradeRecord(
                        entry_time=datetime.fromisoformat(t['entry_time']),
                        exit_time=datetime.fromisoformat(t['exit_time']),
                        entry_price=t['entry_price'],
                        exit_price=t['exit_price'],
                        direction=t['direction'],
                        confidence=t['confidence'],
                        time_of_day=t['time_of_day'],
                        day_of_week=t['day_of_week'],
                        pnl=t['pnl'],
                        duration_minutes=t['duration_minutes'],
                        exit_reason=t['exit_reason']
                    )
                    for t in trade_dicts
                ]
                log.info(f"Loaded {len(self.trades)} trade records")
            
            # Load insights
            if os.path.exists('data/insights.joblib'):
                insight_dicts = joblib.load('data/insights.joblib')
                self.insights = [
                    PatternInsight(
                        name=i['name'],
                        description=i['description'],
                        sample_size=i['sample_size'],
                        impact=i['impact'],
                        action=i['action']
                    )
                    for i in insight_dicts
                ]
                log.info(f"Loaded {len(self.insights)} insights")
                
        except Exception as e:
            log.error(f"Load error: {e}")