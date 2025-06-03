// NinjaScript - RLTrader

using System;
using System.Globalization;
using System.IO;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using NinjaTrader.Cbi;
using NinjaTrader.Data;
using NinjaTrader.NinjaScript;
using NinjaTrader.NinjaScript.Strategies;
using NinjaTrader.Gui.Chart;
using NinjaTrader.NinjaScript.DrawingTools;
using System.Collections.Generic;
using System.Web.Script.Serialization;
using System.ComponentModel;
using System.ComponentModel.DataAnnotations;
using NinjaTrader.NinjaScript.Indicators;
using System.Windows.Media;

namespace NinjaTrader.NinjaScript.Strategies
{
    public class RLTrader : Strategy
    {
        #region Private Fields
        
        // Position Management
        private double entryPrice = 0;
        private double stopLossPrice = 0;
        private double takeProfitPrice = 0;
        private double trailingStopPrice = 0;
        private bool isTrailingStopActive = false;
        private double lastSignalConfidence = 0;
        private string lastSignalQuality = "";
        private int currentPositionSize = 0;
        private DateTime lastEntryTime = DateTime.MinValue;
        
        // Exit tracking
        private bool hasStopLoss = false;
        private bool hasTakeProfit = false;
        private List<string> activeOrders = new List<string>();
        
        // Indicators
        private EMA emaFast;
        private EMA emaSlow;
        private Series<double> lwpeSeries;
        
        // LWPE handling
        private double currentLWPE = 0.5;
        private readonly object lwpeLock = new object();
        
        // Network connections
        private TcpClient sendSock, recvSock, tickSock, lwpeSock;
        private Thread recvThread, lwpeThread;
        private volatile bool running;
        private bool socketsStarted = false;
		
        // Signal handling
        private SignalData latestSignal;
		private long lastProcessedTimestamp = 0;
        private DateTime lastSignalTime = DateTime.MinValue;
        private readonly object signalLock = new object();
        
        // Performance tracking
        private int signalCount = 0;
        private int tradesExecuted = 0;
        private int stopLossHits = 0;
        private int takeProfitHits = 0;
        private int trailingStopHits = 0;
        private int scaleOutExecutions = 0;
        private DateTime strategyStartTime;
        
        // Serialization
        private readonly JavaScriptSerializer serializer = new JavaScriptSerializer();
        
        // State tracking
        private bool isTerminated = false;
        private static int instanceCounter = 0;
        private int instanceId;
        
        #endregion
        
        #region Properties
        
        [NinjaScriptProperty]
        [Range(0.001, 0.1)]
        [Display(Name = "Risk Percent", Description = "Risk percentage per trade", Order = 1, GroupName = "Risk Management")]
        public double RiskPercent { get; set; } = 0.01;

        [NinjaScriptProperty]
        [Range(1, 100)]
        [Display(Name = "Stop Loss Ticks", Description = "Stop loss distance in ticks", Order = 2, GroupName = "Exit Management")]
        public int StopLossTicks { get; set; } = 20;

        [NinjaScriptProperty]
        [Range(1, 200)]
        [Display(Name = "Take Profit Ticks", Description = "Take profit distance in ticks", Order = 3, GroupName = "Exit Management")]
        public int TakeProfitTicks { get; set; } = 40;

        [NinjaScriptProperty]
        [Range(1, 50)]
        [Display(Name = "Trailing Stop Ticks", Description = "Trailing stop distance in ticks", Order = 4, GroupName = "Exit Management")]
        public int TrailingStopTicks { get; set; } = 15;

        [NinjaScriptProperty]
        [Range(0.1, 1.0)]
        [Display(Name = "High Confidence Threshold", Description = "Confidence threshold for scaling vs full exit", Order = 5, GroupName = "Exit Management")]
        public double HighConfidenceThreshold { get; set; } = 0.75;

        [NinjaScriptProperty]
        [Range(10, 90)]
        [Display(Name = "Scale Out Percentage", Description = "Percentage to scale out on partial exits", Order = 6, GroupName = "Exit Management")]
        public int ScaleOutPercentage { get; set; } = 50;

        [NinjaScriptProperty]
        [Range(1, 20)]
        [Display(Name = "Max Position Size", Description = "Maximum position size", Order = 7, GroupName = "Position Sizing")]
        public int MaxPositionSize { get; set; } = 10;

        [NinjaScriptProperty]
        [Range(1, 10)]
        [Display(Name = "Base Position Size", Description = "Base position size for low confidence", Order = 8, GroupName = "Position Sizing")]
        public int BasePositionSize { get; set; } = 3;

        [NinjaScriptProperty]
        [Range(5, 50)]
        [Display(Name = "EMA Fast Period", Description = "Fast EMA period", Order = 9, GroupName = "Indicators")]
        public int EmaFastPeriod { get; set; } = 12;

        [NinjaScriptProperty]
        [Range(10, 100)]
        [Display(Name = "EMA Slow Period", Description = "Slow EMA period", Order = 10, GroupName = "Indicators")]
        public int EmaSlowPeriod { get; set; } = 26;

        [NinjaScriptProperty]
        [Range(5, 20)]
        [Display(Name = "Tenkan Period", Description = "Ichimoku Tenkan period", Order = 11, GroupName = "Indicators")]
        public int TenkanPeriod { get; set; } = 9;

        [NinjaScriptProperty]
        [Range(15, 50)]
        [Display(Name = "Kijun Period", Description = "Ichimoku Kijun period", Order = 12, GroupName = "Indicators")]
        public int KijunPeriod { get; set; } = 26;

        [NinjaScriptProperty]
        [Range(25, 100)]
        [Display(Name = "Senkou Period", Description = "Ichimoku Senkou period", Order = 13, GroupName = "Indicators")]
        public int SenkouPeriod { get; set; } = 52;

        [NinjaScriptProperty]
        [Range(0.1, 1.0)]
        [Display(Name = "Min Confidence", Description = "Minimum confidence threshold for trading", Order = 14, GroupName = "Signal Filtering")]
        public double MinConfidence { get; set; } = 0.45;
		
		[NinjaScriptProperty]
		[Display(Name = "Enable Trend Filter", Description = "Block counter-trend trades", Order = 15, GroupName = "Signal Filtering")]
		public bool EnableTrendFilter { get; set; } = true;
		
		[NinjaScriptProperty]
		[Range(20, 100)]
		[Display(Name = "Trend Period", Description = "Period for trend analysis", Order = 16, GroupName = "Signal Filtering")]
		public int TrendPeriod { get; set; } = 50;

        [NinjaScriptProperty]
        [Display(Name = "Enable Trailing Stops", Description = "Enable trailing stop functionality", Order = 17, GroupName = "Exit Management")]
        public bool EnableTrailingStops { get; set; } = true;

        [NinjaScriptProperty]
        [Display(Name = "Enable Scale Outs", Description = "Enable partial position scaling", Order = 18, GroupName = "Exit Management")]
        public bool EnableScaleOuts { get; set; } = true;

        [NinjaScriptProperty]
        [Range(5, 300)]
        [Display(Name = "Min Hold Time Seconds", Description = "Minimum time to hold position", Order = 19, GroupName = "Exit Management")]
        public int MinHoldTimeSeconds { get; set; } = 30;

        [NinjaScriptProperty]
        [Display(Name = "Enable Logging", Description = "Enable detailed logging", Order = 20, GroupName = "Debug")]
        public bool EnableLogging { get; set; } = true;
        
        #endregion
        
        #region State Management
        
        protected override void OnStateChange()
        {
            try
            {
                switch (State)
                {
                    case State.SetDefaults:
                        ConfigureDefaults();
                        break;
                        
                    case State.DataLoaded:
                        InitializeIndicators();
                        break;
                        
                    case State.Historical:
                        if (!socketsStarted)
                        {
                            InitializeSockets();
                        }
                        break;
                        
                    case State.Realtime:
                        if (!socketsStarted)
                        {
                            InitializeSockets();
                        }
                        strategyStartTime = DateTime.Now;
                        Print($"RLTrader #{instanceId} started with complete exit management");
                        LogExitParameters();
                        break;
                        
                    case State.Terminated:
                        Cleanup();
                        break;
                }
            }
            catch (Exception ex)
            {
                Print($"Instance #{instanceId} OnStateChange error in {State}: {ex.Message}");
            }
        }
        
		private void ConfigureDefaults()
		{
		    instanceId = ++instanceCounter;
		    
		    Name = "RLTrader";
		    Description = "RL Trading Strategy with Complete Exit Management v3.0";
		    Calculate = Calculate.OnBarClose;
		    
		    // Chart configuration
		    IsOverlay = false;
		    DisplayInDataBox = true;
		    
		    // Enhanced plots for exit management
		    AddPlot(Brushes.Blue, "LWPE");
		    AddPlot(Brushes.Green, "Signal Quality");
		    AddPlot(Brushes.Orange, "Position Size");
		    AddPlot(Brushes.Red, "Stop Loss");
		    AddPlot(Brushes.Lime, "Take Profit");
		    
		    // Entry configuration for multiple exits
		    BarsRequiredToTrade = Math.Max(SenkouPeriod + 5, EmaSlowPeriod + 5);
		    EntriesPerDirection = 1; // Single entry, multiple exits
		    EntryHandling = EntryHandling.AllEntries;
		    
		    // Reset state
		    isTerminated = false;
		    socketsStarted = false;
		    running = false;
		    ResetPositionTracking();
		}

        private void LogExitParameters()
        {
            if (!EnableLogging) return;
            
            Print("=== Exit Management Configuration ===");
            Print($"Stop Loss: {StopLossTicks} ticks");
            Print($"Take Profit: {TakeProfitTicks} ticks");
            Print($"Trailing Stop: {TrailingStopTicks} ticks (Enabled: {EnableTrailingStops})");
            Print($"High Confidence Threshold: {HighConfidenceThreshold:F2}");
            Print($"Scale Out: {ScaleOutPercentage}% (Enabled: {EnableScaleOuts})");
            Print($"Max Position: {MaxPositionSize}, Base: {BasePositionSize}");
            Print($"Min Hold Time: {MinHoldTimeSeconds} seconds");
        }

        private void ResetPositionTracking()
        {
            entryPrice = 0;
            stopLossPrice = 0;
            takeProfitPrice = 0;
            trailingStopPrice = 0;
            isTrailingStopActive = false;
            currentPositionSize = 0;
            lastSignalConfidence = 0;
            lastSignalQuality = "";
            hasStopLoss = false;
            hasTakeProfit = false;
            activeOrders.Clear();
            lastEntryTime = DateTime.MinValue;
        }
        
		private void InitializeIndicators()
		{
		    try
		    {
		        emaFast = EMA(EmaFastPeriod);
		        emaSlow = EMA(EmaSlowPeriod);
		        lwpeSeries = new Series<double>(this);
		        
		        AddChartIndicator(emaFast);
		        AddChartIndicator(emaSlow);
		        
		        if (EnableLogging)
		        {
		            Print($"Indicators initialized with exit management");
		        }
		    }
		    catch (Exception ex)
		    {
		        Print($"Indicator initialization error: {ex.Message}");
		    }
		}
        
        private void InitializeSockets()
        {
            if (socketsStarted) return;
            
            try
            {
                ConnectToSockets();
                StartBackgroundThreads();
                socketsStarted = true;
                Print($"RLTrader #{instanceId} connected - Ready for ML signals with exit management");
            }
            catch (Exception ex)
            {
                Print($"Socket connection failed: {ex.Message}");
                socketsStarted = false;
            }
        }
        
        private void Cleanup()
        {
            bool shouldCleanup = false;
            
            lock (signalLock)
            {
                if (!isTerminated)
                {
                    isTerminated = true;
                    shouldCleanup = true;
                }
            }
            
            if (!shouldCleanup) return;
            
            if (socketsStarted)
            {
                LogFinalPerformance();
            }
            
            running = false;
            
            if (recvThread?.IsAlive == true)
            {
                recvThread.Join(1000);
            }
            
            if (lwpeThread?.IsAlive == true)
            {
                lwpeThread.Join(1000);
            }
            
            DisposeSockets();
        }

        private void LogFinalPerformance()
        {
            try
            {
                if (strategyStartTime != DateTime.MinValue)
                {
                    TimeSpan uptime = DateTime.Now - strategyStartTime;
                    Print($"=== Final Performance ===");
                    Print($"Uptime: {uptime.TotalHours:F1} hours");
                    Print($"Signals: {signalCount}, Trades: {tradesExecuted}");
                    Print($"Stop Losses: {stopLossHits}, Take Profits: {takeProfitHits}");
                    Print($"Trailing Stops: {trailingStopHits}, Scale Outs: {scaleOutExecutions}");
                    Print($"Current Position: {GetCurrentPosition()}");
                }
            }
            catch (Exception ex)
            {
                Print($"Performance summary error: {ex.Message}");
            }
        }
        
        #endregion
        
        #region Socket Management (Simplified)
        
        private void ConnectToSockets()
        {
            const string host = "localhost";
            
            sendSock = new TcpClient(host, 5556);
            recvSock = new TcpClient(host, 5557);
            tickSock = new TcpClient(host, 5558);
            lwpeSock = new TcpClient(host, 5559);    
        }
        
        private void StartBackgroundThreads()
        {
            running = true;
            
            recvThread = new Thread(SignalReceiveLoop) 
            { 
                IsBackground = true, 
                Name = $"SignalReceiver_{instanceId}" 
            };
            
            lwpeThread = new Thread(LwpeReceiveLoop) 
            { 
                IsBackground = true, 
                Name = $"LwpeReceiver_{instanceId}" 
            };
            
            recvThread.Start();
            lwpeThread.Start();
        }
        
        private void DisposeSockets()
        {
            var sockets = new[] { sendSock, recvSock, tickSock, lwpeSock };
            
            foreach (var socket in sockets)
            {
                try { socket?.Close(); }
                catch (Exception ex) { Print($"Error closing socket: {ex.Message}"); }
            }
        }
        
        #endregion
        
        #region Main Trading Logic with Exit Management
        
		protected override void OnBarUpdate()
		{
		    try
		    {
		        UpdatePlots();
		        SendFeatureVector();
		        
		        if (!IsReadyForTrading())
		            return;

		        // Manage existing positions first
		        ManageExistingPositions();
		        
		        // Process new signals
		        ProcessLatestSignal();
		        SendPositionUpdate();
		        
		        // Visual updates
		        if (CurrentBar >= BarsRequiredToTrade)
		        {
		            PlotPositionInfo();
		        }
		        
		        // Periodic logging
		        if (CurrentBar % 100 == 0 && EnableLogging)
		        {
		            LogCurrentStatus();
		        }
		    }
		    catch (Exception ex)
		    {
		        Print($"OnBarUpdate error: {ex.Message}");
		    }
		}

        private void ManageExistingPositions()
        {
            if (Position.MarketPosition == MarketPosition.Flat)
            {
                if (currentPositionSize != 0)
                {
                    // Position was closed, reset tracking
                    ResetPositionTracking();
                    if (EnableLogging)
                        Print("Position closed - reset tracking");
                }
                return;
            }

            // Update trailing stops
            if (EnableTrailingStops && isTrailingStopActive)
            {
                UpdateTrailingStop();
            }

            // Check for scale out opportunities
            if (EnableScaleOuts && ShouldScaleOut())
            {
                ExecuteScaleOut();
            }

            // Emergency exit on conflicting signals
            if (ShouldEmergencyExit())
            {
                ExecuteEmergencyExit();
            }
        }

        private void UpdateTrailingStop()
        {
            if (Position.MarketPosition == MarketPosition.Flat) return;

            try
            {
                double currentPrice = Close[0];
                double newTrailingPrice = 0;

                if (Position.MarketPosition == MarketPosition.Long)
                {
                    newTrailingPrice = currentPrice - (TrailingStopTicks * TickSize);
                    
                    if (newTrailingPrice > trailingStopPrice || trailingStopPrice == 0)
                    {
                        trailingStopPrice = newTrailingPrice;
                        
                        if (EnableLogging)
                            Print($"Trailing stop updated to {trailingStopPrice:F2} (price: {currentPrice:F2})");
                    }

                    // Check if trailing stop hit
                    if (currentPrice <= trailingStopPrice)
                    {
                        ExitLong("TrailingStop");
                        trailingStopHits++;
                        if (EnableLogging)
                            Print($"Trailing stop executed at {currentPrice:F2}");
                    }
                }
                else if (Position.MarketPosition == MarketPosition.Short)
                {
                    newTrailingPrice = currentPrice + (TrailingStopTicks * TickSize);
                    
                    if (newTrailingPrice < trailingStopPrice || trailingStopPrice == 0)
                    {
                        trailingStopPrice = newTrailingPrice;
                        
                        if (EnableLogging)
                            Print($"Trailing stop updated to {trailingStopPrice:F2} (price: {currentPrice:F2})");
                    }

                    // Check if trailing stop hit
                    if (currentPrice >= trailingStopPrice)
                    {
                        ExitShort("TrailingStop");
                        trailingStopHits++;
                        if (EnableLogging)
                            Print($"Trailing stop executed at {currentPrice:F2}");
                    }
                }
            }
            catch (Exception ex)
            {
                Print($"Trailing stop update error: {ex.Message}");
            }
        }

        private bool ShouldScaleOut()
        {
            if (Position.MarketPosition == MarketPosition.Flat) return false;
            if (Position.Quantity <= 1) return false;
            if (lastSignalConfidence < HighConfidenceThreshold) return false;

            double currentPrice = Close[0];
            double profitTicks = 0;

            if (Position.MarketPosition == MarketPosition.Long)
            {
                profitTicks = (currentPrice - entryPrice) / TickSize;
            }
            else
            {
                profitTicks = (entryPrice - currentPrice) / TickSize;
            }

            // Scale out when halfway to take profit target
            return profitTicks >= (TakeProfitTicks * 0.5);
        }

        private void ExecuteScaleOut()
        {
            try
            {
                int scaleOutQuantity = Math.Max(1, (Position.Quantity * ScaleOutPercentage) / 100);
                
                if (Position.MarketPosition == MarketPosition.Long)
                {
                    ExitLong(scaleOutQuantity, "ScaleOut", "ML_Long");
                }
                else if (Position.MarketPosition == MarketPosition.Short)
                {
                    ExitShort(scaleOutQuantity, "ScaleOut", "ML_Short");
                }

                scaleOutExecutions++;
                
                if (EnableLogging)
                    Print($"Scale out executed: {scaleOutQuantity} contracts at {Close[0]:F2}");

                // Activate trailing stop on remaining position
                if (EnableTrailingStops && !isTrailingStopActive)
                {
                    isTrailingStopActive = true;
                    trailingStopPrice = 0; // Will be set on next update
                    if (EnableLogging)
                        Print("Trailing stop activated after scale out");
                }
            }
            catch (Exception ex)
            {
                Print($"Scale out execution error: {ex.Message}");
            }
        }

        private bool ShouldEmergencyExit()
        {
            // Exit if confidence drops significantly or signal quality becomes poor
            return (lastSignalConfidence > 0 && lastSignalConfidence < 0.3) ||
                   lastSignalQuality == "poor";
        }

        private void ExecuteEmergencyExit()
        {
            try
            {
                if (Position.MarketPosition == MarketPosition.Long)
                {
                    ExitLong("EmergencyExit");
                }
                else if (Position.MarketPosition == MarketPosition.Short)
                {
                    ExitShort("EmergencyExit");
                }

                if (EnableLogging)
                    Print($"Emergency exit executed due to poor signal quality");
            }
            catch (Exception ex)
            {
                Print($"Emergency exit error: {ex.Message}");
            }
        }
        
		private void UpdatePlots()
		{
		    try
		    {
		        // LWPE
		        lock (lwpeLock)
		        {
		            Values[0][0] = currentLWPE;
		        }
		        
		        // Signal Quality (convert string to numeric)
		        double qualityValue = 0.5;
		        switch (lastSignalQuality.ToLower())
		        {
		            case "excellent": qualityValue = 1.0; break;
		            case "good": qualityValue = 0.75; break;
		            case "poor": qualityValue = 0.25; break;
		            default: qualityValue = 0.5; break;
		        }
		        Values[1][0] = qualityValue;
		        
		        // Position Size
		        Values[2][0] = Math.Abs(GetCurrentPosition());
		        
		        // Stop Loss Price
		        Values[3][0] = stopLossPrice;
		        
		        // Take Profit Price
		        Values[4][0] = takeProfitPrice;
		    }
		    catch (Exception ex)
		    {
		        if (EnableLogging)
		            Print($"Plot update error: {ex.Message}");
		    }
		}
        
		private bool IsReadyForTrading()
		{
		    return CurrentBar >= Math.Max(SenkouPeriod, EmaSlowPeriod) && 
		           socketsStarted && 
		           running;
		}

        private void ProcessLatestSignal()
        {
            var signal = GetLatestSignal();
            
            if (signal == null) return;
            
            if (IsSignalValid(signal))
            {
                ExecuteSignal(signal);
                UpdateLastSignalTime(signal);
                signalCount++;
            }
        }

        private SignalData GetLatestSignal()
        {
            lock (signalLock)
            {
                return latestSignal;
            }
        }

        private bool IsSignalValid(SignalData signal)
        {
            try
            {
                // Time validation
                var signalDateTime = new DateTime(1970, 1, 1, 0, 0, 0, DateTimeKind.Utc).AddSeconds(signal.Timestamp);
                var signalLocalTime = signalDateTime.ToLocalTime();
                var timeDiff = (DateTime.Now - signalLocalTime).TotalSeconds;

                if (Math.Abs(timeDiff) > 120)
                {
                    if (EnableLogging)
                        Print($"Signal expired: {timeDiff:F1}s old");
                    return false;
                }
                
                // Confidence validation
                if (signal.Confidence < MinConfidence)
                {
                    if (EnableLogging)
                        Print($"Signal confidence {signal.Confidence:F3} below threshold {MinConfidence:F3}");
                    return false;
                }

                // Don't take new signals too quickly
                if (lastEntryTime != DateTime.MinValue && 
                    (DateTime.Now - lastEntryTime).TotalSeconds < MinHoldTimeSeconds)
                {
                    if (EnableLogging)
                        Print($"Signal blocked by minimum hold time");
                    return false;
                }
                
                // Trend filter validation
                if (EnableTrendFilter)
                {
                    int trendDirection = GetTrendDirection();
                    
                    if ((signal.Action == 1 && trendDirection == -1) || 
                        (signal.Action == 2 && trendDirection == 1))
                    {
                        if (EnableLogging)
                            Print($"Signal blocked by trend filter");
                        return false;
                    }
                }
                
                return true;
            }
            catch (Exception ex)
            {
                Print($"Signal validation error: {ex.Message}");
                return false;
            }
        }

        private void ExecuteSignal(SignalData signal)
        {
            try
            {
                // Store signal info for position management
                lastSignalConfidence = signal.Confidence;
                lastSignalQuality = signal.Quality ?? "unknown";

                // Calculate position size based on confidence
                int positionSize = CalculatePositionSize(signal.Confidence);
                
                if (positionSize <= 0)
                {
                    if (EnableLogging)
                        Print($"Calculated position size is 0 for confidence {signal.Confidence:F3}");
                    return;
                }

                switch (signal.Action)
                {
                    case 1: // BUY
                        if (Position.MarketPosition != MarketPosition.Long)
                        {
                            // Close any short position first
                            if (Position.MarketPosition == MarketPosition.Short)
                            {
                                ExitShort("ReverseToLong");
                            }
                            
                            EnterLong(positionSize, "ML_Long");
                            SetupExitOrders(signal, positionSize, true);
                            
                            if (EnableLogging)
                                Print($"LONG Entry: size={positionSize}, conf={signal.Confidence:F3}, quality={signal.Quality}");
                        }
                        break;
                        
                    case 2: // SELL
                        if (Position.MarketPosition != MarketPosition.Short)
                        {
                            // Close any long position first
                            if (Position.MarketPosition == MarketPosition.Long)
                            {
                                ExitLong("ReverseToShort");
                            }
                            
                            EnterShort(positionSize, "ML_Short");
                            SetupExitOrders(signal, positionSize, false);
                            
                            if (EnableLogging)
                                Print($"SHORT Entry: size={positionSize}, conf={signal.Confidence:F3}, quality={signal.Quality}");
                        }
                        break;
                        
                    case 0: // HOLD/EXIT
                        if (Position.MarketPosition != MarketPosition.Flat)
                        {
                            // Determine exit strategy based on confidence
                            if (signal.Confidence < HighConfidenceThreshold)
                            {
                                // Low confidence - full exit
                                if (Position.MarketPosition == MarketPosition.Long)
                                    ExitLong("ML_Exit");
                                else
                                    ExitShort("ML_Exit");
                                
                                if (EnableLogging)
                                    Print($"FULL EXIT: conf={signal.Confidence:F3}, quality={signal.Quality}");
                            }
                            else if (EnableScaleOuts && Position.Quantity > 1)
                            {
                                // High confidence - partial exit
                                int exitSize = Math.Max(1, Position.Quantity / 2);
                                
                                if (Position.MarketPosition == MarketPosition.Long)
                                    ExitLong(exitSize, "ML_PartialExit", "ML_Long");
                                else
                                    ExitShort(exitSize, "ML_PartialExit", "ML_Short");
                                
                                if (EnableLogging)
                                    Print($"PARTIAL EXIT: size={exitSize}, conf={signal.Confidence:F3}");
                            }
                        }
                        break;
                }
            }
            catch (Exception ex)
            {
                Print($"Signal execution error: {ex.Message}");
            }
        }

        private int CalculatePositionSize(double confidence)
        {
            // Simple confidence-based sizing
            if (confidence < MinConfidence) return 0;
            
            double sizeMultiplier = 1.0;
            
            if (confidence >= 0.8)
                sizeMultiplier = 2.0;
            else if (confidence >= 0.7)
                sizeMultiplier = 1.5;
            else if (confidence >= 0.6)
                sizeMultiplier = 1.2;
            else
                sizeMultiplier = 1.0;
            
            int calculatedSize = (int)(BasePositionSize * sizeMultiplier);
            return Math.Min(calculatedSize, MaxPositionSize);
        }

        private void SetupExitOrders(SignalData signal, int positionSize, bool isLong)
        {
            try
            {
                entryPrice = Close[0];
                currentPositionSize = positionSize;
                lastEntryTime = DateTime.Now;
                
                // Calculate stop loss and take profit prices
                if (isLong)
                {
                    stopLossPrice = entryPrice - (StopLossTicks * TickSize);
                    takeProfitPrice = entryPrice + (TakeProfitTicks * TickSize);
                }
                else
                {
                    stopLossPrice = entryPrice + (StopLossTicks * TickSize);
                    takeProfitPrice = entryPrice - (TakeProfitTicks * TickSize);
                }

                // Set exit orders
                if (isLong)
                {
                    SetStopLoss("ML_Long", CalculationMode.Price, stopLossPrice, false);
                    SetProfitTarget("ML_Long", CalculationMode.Price, takeProfitPrice);
                }
                else
                {
                    SetStopLoss("ML_Short", CalculationMode.Price, stopLossPrice, false);
                    SetProfitTarget("ML_Short", CalculationMode.Price, takeProfitPrice);
                }

                hasStopLoss = true;
                hasTakeProfit = true;

                // Initialize trailing stop if high confidence
                if (EnableTrailingStops && signal.Confidence >= HighConfidenceThreshold)
                {
                    isTrailingStopActive = true;
                    trailingStopPrice = 0; // Will be set on first update
                    
                    if (EnableLogging)
                        Print($"Trailing stop will activate for high confidence trade");
                }

                if (EnableLogging)
                {
                    Print($"Exit orders set - SL: {stopLossPrice:F2}, TP: {takeProfitPrice:F2}, Entry: {entryPrice:F2}");
                }
            }
            catch (Exception ex)
            {
                Print($"Exit order setup error: {ex.Message}");
            }
        }

		protected override void OnExecutionUpdate(Execution execution, string executionId, double price, int quantity, MarketPosition marketPosition, string orderId, DateTime time)
		{
		    try
		    {
		        if (execution.Order != null)
		        {
		            string orderName = execution.Order.Name;
		            tradesExecuted++;

		            // Track exit types
		            if (orderName.Contains("Stop"))
		            {
		                stopLossHits++;
		                if (EnableLogging)
		                    Print($"STOP LOSS HIT #{stopLossHits}: {quantity} @ {price:F2}");
		            }
		            else if (orderName.Contains("Profit") || orderName.Contains("Target"))
		            {
		                takeProfitHits++;
		                if (EnableLogging)
		                    Print($"TAKE PROFIT HIT #{takeProfitHits}: {quantity} @ {price:F2}");
		            }
		            else if (orderName.Contains("TrailingStop"))
		            {
		                trailingStopHits++;
		                if (EnableLogging)
		                    Print($"TRAILING STOP HIT #{trailingStopHits}: {quantity} @ {price:F2}");
		            }
		            else if (orderName.Contains("ScaleOut"))
		            {
		                if (EnableLogging)
		                    Print($"SCALE OUT EXECUTED: {quantity} @ {price:F2}");
		            }
		            else
		            {
		                if (EnableLogging)
		                    Print($"ENTRY FILL #{tradesExecuted}: {orderName} {quantity} @ {price:F2}");
		            }

		            // Update position tracking
		            if (marketPosition == MarketPosition.Flat)
		            {
		                ResetPositionTracking();
		                if (EnableLogging)
		                    Print($"Position FLAT - tracking reset");
		            }

		            // Send position update to Python
		            Core.Globals.RandomDispatcher.BeginInvoke(new Action(() =>
		            {
		                SendPositionUpdate();
		            }));
		        }
		    }
		    catch (Exception ex)
		    {
		        Print($"Execution update error: {ex.Message}");
		    }
		}

        private void PlotPositionInfo()
        {
            try
            {
                if (Position.MarketPosition != MarketPosition.Flat && entryPrice > 0)
                {
                    // Plot entry line
                    Draw.HorizontalLine(this, "EntryLine", entryPrice, Brushes.Yellow);
                    
                    // Plot stop loss line
                    if (stopLossPrice > 0)
                    {
                        Draw.HorizontalLine(this, "StopLossLine", stopLossPrice, Brushes.Red);
                    }
                    
                    // Plot take profit line
                    if (takeProfitPrice > 0)
                    {
                        Draw.HorizontalLine(this, "TakeProfitLine", takeProfitPrice, Brushes.Lime);
                    }
                    
                    // Plot trailing stop if active
                    if (isTrailingStopActive && trailingStopPrice > 0)
                    {
                        Draw.HorizontalLine(this, "TrailingStopLine", trailingStopPrice, Brushes.Orange);
                    }

                    // Add position info text
                    string positionInfo = $"Pos: {Position.Quantity} | Conf: {lastSignalConfidence:F2} | Qual: {lastSignalQuality}";
                    Draw.TextFixed(this, "PositionInfo", positionInfo, TextPosition.TopLeft, 
                                 Brushes.White, new NinjaTrader.Gui.Tools.SimpleFont("Arial", 10), Brushes.Black, Brushes.Transparent, 0);
                }
                else
                {
                    // Remove lines when flat
                    RemoveDrawObject("EntryLine");
                    RemoveDrawObject("StopLossLine");
                    RemoveDrawObject("TakeProfitLine");
                    RemoveDrawObject("TrailingStopLine");
                    RemoveDrawObject("PositionInfo");
                }
            }
            catch (Exception ex)
            {
                if (EnableLogging)
                    Print($"Position plotting error: {ex.Message}");
            }
        }

        private void LogCurrentStatus()
        {
            try
            {
                var features = CalculateFeatures();
                
                string positionStatus = Position.MarketPosition == MarketPosition.Flat ? "FLAT" :
                                      $"{Position.MarketPosition} {Position.Quantity}";

                Print($"Bar {CurrentBar}: {positionStatus} | LWPE={features.LWPE:F3} | " +
                      $"LastConf={lastSignalConfidence:F3} | Signals={signalCount} | Trades={tradesExecuted}");
                      
                if (Position.MarketPosition != MarketPosition.Flat)
                {
                    double unrealizedPnL = Position.GetUnrealizedProfitLoss(PerformanceUnit.Currency, Close[0]);
                    Print($"  PnL: {unrealizedPnL:C} | Entry: {entryPrice:F2} | Current: {Close[0]:F2}");
                    
                    if (isTrailingStopActive)
                    {
                        Print($"  Trailing Stop: {trailingStopPrice:F2} (Active)");
                    }
                }
            }
            catch (Exception ex)
            {
                Print($"Status logging error: {ex.Message}");
            }
        }

        private void UpdateLastSignalTime(SignalData signal)
        {
            lastProcessedTimestamp = signal.Timestamp;
            lastSignalTime = Time[0];
        }
        
        #endregion
        
        #region Market Data and Feature Processing (Unchanged)
        
        protected override void OnMarketData(MarketDataEventArgs e)
        {
            if (!IsRelevantMarketData(e.MarketDataType))
                return;
                
            ForwardTickData(e);
        }
        
        private bool IsRelevantMarketData(MarketDataType dataType)
        {
            return dataType == MarketDataType.Bid || 
                   dataType == MarketDataType.Ask || 
                   dataType == MarketDataType.Last;
        }
        
        private void ForwardTickData(MarketDataEventArgs e)
        {
            try
            {
                long unixMs = GetUnixTimestamp(e.Time);
                string tickData = FormatTickData(unixMs, e.Price, e.Volume, e.MarketDataType);
                
                SendTickData(tickData);
            }
            catch (Exception ex)
            {
                if (EnableLogging)
                    Print($"Tick forwarding error: {ex.Message}");
            }
        }
        
        private long GetUnixTimestamp(DateTime dateTime)
        {
            return (long)(dateTime.ToUniversalTime() - new DateTime(1970, 1, 1)).TotalMilliseconds;
        }
        
        private string FormatTickData(long timestamp, double price, long volume, MarketDataType dataType)
        {
            return $"{timestamp},{price},{volume},{dataType}\n";
        }
        
        private void SendTickData(string tickData)
        {
            try
            {
                if (tickSock?.Connected == true)
                {
                    byte[] data = Encoding.UTF8.GetBytes(tickData);
                    tickSock.GetStream().Write(data, 0, data.Length);
                }
            }
            catch (Exception ex)
            {
                if (EnableLogging)
                    Print($"Tick send error: {ex.Message}");
            }
        }

        private void SendFeatureVector()
        {
            if (sendSock?.Connected != true)
                return;
                
            try
            {
                var features = CalculateFeatures();
                string payload = CreateFeaturePayload(features);
                TransmitData(sendSock, payload);
            }
            catch (Exception ex)
            {
                if (EnableLogging)
                    Print($"Feature transmission error: {ex.Message}");
            }
        }
        
        private FeatureVector CalculateFeatures()
        {
            try
            {
                double volMean = SMA(Volume, 20)[0];
                double volStd = StdDev(Volume, 20)[0];
                double normalizedVolume = volStd != 0 ? (Volume[0] - volMean) / volStd : 0;
                
                double lwpeValue;
                lock (lwpeLock)
                {
                    lwpeValue = currentLWPE;
                }
                
                return new FeatureVector
                {
                    Close = Close[0],
                    NormalizedVolume = normalizedVolume,
                    TenkanKijunSignal = GetTenkanKijunSignal(),
                    PriceCloudSignal = GetPriceCloudSignal(),
                    FutureCloudSignal = GetFutureCloudSignal(),
                    EmaCrossSignal = GetEmaCrossSignal(),
                    TenkanMomentum = GetTenkanMomentum(),
                    KijunMomentum = GetKijunMomentum(),
                    LWPE = lwpeValue,
                    IsLive = State == State.Realtime
                };
            }
            catch (Exception ex)
            {
                if (EnableLogging)
                    Print($"Feature calculation error: {ex.Message}");
                return GetDefaultFeatures();
            }
        }
        
        private FeatureVector GetDefaultFeatures()
        {
            double lwpeValue;
            lock (lwpeLock)
            {
                lwpeValue = currentLWPE;
            }
            
            return new FeatureVector
            {
                Close = Close[0],
                NormalizedVolume = 0,
                TenkanKijunSignal = 0,
                PriceCloudSignal = 0,
                FutureCloudSignal = 0,
                EmaCrossSignal = 0,
                TenkanMomentum = 0,
                KijunMomentum = 0,
                LWPE = lwpeValue,
                IsLive = State == State.Realtime
            };
        }

        // Ichimoku calculation methods (unchanged from original)
		private double GetTenkanKijunSignal()
		{
		    try
		    {
		        if (CurrentBar < Math.Max(TenkanPeriod, KijunPeriod))
		            return 0;
		            
		        double tenkanHigh = MAX(High, TenkanPeriod)[0];
		        double tenkanLow = MIN(Low, TenkanPeriod)[0];
		        double tenkan = (tenkanHigh + tenkanLow) / 2;
		        
		        double kijunHigh = MAX(High, KijunPeriod)[0];
		        double kijunLow = MIN(Low, KijunPeriod)[0];
		        double kijun = (kijunHigh + kijunLow) / 2;
		        
		        if (tenkan == 0 || kijun == 0)
		            return 0;
		            
		        double diff = tenkan - kijun;
		        double threshold = Close[0] * 0.001;
		        
		        if (diff > threshold)
		            return 1.0;
		        else if (diff < -threshold)
		            return -1.0;
		        else
		            return 0.0;
		    }
		    catch
		    {
		        return 0;
		    }
		}
        
		private double GetPriceCloudSignal()
		{
		    try
		    {
		        if (CurrentBar < SenkouPeriod + 26)
		            return 0;
		            
		        double tenkanHigh26 = MAX(High, TenkanPeriod)[26];
		        double tenkanLow26 = MIN(Low, TenkanPeriod)[26];
		        double tenkan26 = (tenkanHigh26 + tenkanLow26) / 2;
		        
		        double kijunHigh26 = MAX(High, KijunPeriod)[26];
		        double kijunLow26 = MIN(Low, KijunPeriod)[26];
		        double kijun26 = (kijunHigh26 + kijunLow26) / 2;
		        
		        double senkouA = (tenkan26 + kijun26) / 2;
		        
		        double senkouBHigh26 = MAX(High, SenkouPeriod)[26];
		        double senkouBLow26 = MIN(Low, SenkouPeriod)[26];
		        double senkouB = (senkouBHigh26 + senkouBLow26) / 2;
		        
		        if (senkouA == 0 || senkouB == 0)
		            return 0;
		            
		        double cloudTop = Math.Max(senkouA, senkouB);
		        double cloudBottom = Math.Min(senkouA, senkouB);
		        double buffer = Close[0] * 0.002;
		        
		        if (Close[0] > cloudTop + buffer)
		            return 1.0;
		        else if (Close[0] < cloudBottom - buffer)
		            return -1.0;
		        else
		            return 0.0;
		    }
		    catch
		    {
		        return 0;
		    }
		}
		
		private double GetFutureCloudSignal()
		{
		    try
		    {
		        if (CurrentBar < SenkouPeriod)
		            return 0;
		            
		        double tenkanHigh = MAX(High, TenkanPeriod)[0];
		        double tenkanLow = MIN(Low, TenkanPeriod)[0];
		        double tenkan = (tenkanHigh + tenkanLow) / 2;
		        
		        double kijunHigh = MAX(High, KijunPeriod)[0];
		        double kijunLow = MIN(Low, KijunPeriod)[0];
		        double kijun = (kijunHigh + kijunLow) / 2;
		        
		        double futureA = (tenkan + kijun) / 2;
		        
		        double futureBHigh = MAX(High, SenkouPeriod)[0];
		        double futureBLow = MIN(Low, SenkouPeriod)[0];
		        double futureB = (futureBHigh + futureBLow) / 2;
		        
		        if (futureA == 0 || futureB == 0)
		            return 0;
		            
		        double diff = futureA - futureB;
		        double avgPrice = (futureA + futureB) / 2;
		        double threshold = avgPrice * 0.0001;
		        
		        if (diff > threshold)
		            return 1.0;
		        else if (diff < -threshold)
		            return -1.0;
		        else
		            return 0.0;
		    }
		    catch
		    {
		        return 0;
		    }
		}
        
		private double GetEmaCrossSignal()
		{
		    try
		    {
		        if (CurrentBar < EmaSlowPeriod)
		            return 0;
		            
		        double fastEma = emaFast[0];
		        double slowEma = emaSlow[0];
		        
		        if (fastEma == 0 || slowEma == 0)
		            return 0;
		            
		        double diff = fastEma - slowEma;
		        double threshold = Close[0] * 0.0005;
		        
		        if (diff > threshold)
		            return 1.0;
		        else if (diff < -threshold)
		            return -1.0;
		        else
		            return 0.0;
		    }
		    catch
		    {
		        return 0;
		    }
		}
        
		private double GetTenkanMomentum()
		{
		    try
		    {
		        if (CurrentBar < TenkanPeriod + 8)
		            return 0;
		            
		        double currentTenkanHigh = MAX(High, TenkanPeriod)[0];
		        double currentTenkanLow = MIN(Low, TenkanPeriod)[0];
		        double currentTenkan = (currentTenkanHigh + currentTenkanLow) / 2;
		        
		        double previousTenkanHigh = MAX(High, TenkanPeriod)[5];
		        double previousTenkanLow = MIN(Low, TenkanPeriod)[5];
		        double previousTenkan = (previousTenkanHigh + previousTenkanLow) / 2;
		        
		        if (currentTenkan == 0 || previousTenkan == 0)
		            return 0;
		            
		        double change = currentTenkan - previousTenkan;
		        double threshold = currentTenkan * 0.0005;
		        
		        if (change > threshold)
		            return 1.0;
		        else if (change < -threshold)
		            return -1.0;
		        else
		            return 0.0;
		    }
		    catch
		    {
		        return 0;
		    }
		}
		
		private double GetKijunMomentum()
		{
		    try
		    {
		        if (CurrentBar < KijunPeriod + 5)
		            return 0;
		            
		        double currentKijunHigh = MAX(High, KijunPeriod)[0];
		        double currentKijunLow = MIN(Low, KijunPeriod)[0];
		        double currentKijun = (currentKijunHigh + currentKijunLow) / 2;
		        
		        double previousKijunHigh = MAX(High, KijunPeriod)[3];
		        double previousKijunLow = MIN(Low, KijunPeriod)[3];
		        double previousKijun = (previousKijunHigh + previousKijunLow) / 2;
		        
		        if (currentKijun == 0 || previousKijun == 0)
		            return 0;
		            
		        double change = currentKijun - previousKijun;
		        double threshold = currentKijun * 0.0001;
		        
		        if (change > threshold)
		            return 1.0;
		        else if (change < -threshold)
		            return -1.0;
		        else
		            return 0.0;
		    }
		    catch
		    {
		        return 0;
		    }
		}
        
        private string CreateFeaturePayload(FeatureVector features)
        {
            return string.Format(CultureInfo.InvariantCulture,
                @"{{
                    ""features"":[{0:F6},{1:F6},{2:F1},{3:F1},{4:F1},{5:F1},{6:F1},{7:F1},{8:F6}],
                    ""live"":{9}
                }}",
                features.Close, 
                features.NormalizedVolume, 
                features.TenkanKijunSignal,
                features.PriceCloudSignal,
                features.FutureCloudSignal,
                features.EmaCrossSignal,
                features.TenkanMomentum,
                features.KijunMomentum,
                features.LWPE,
                features.IsLive ? 1 : 0);
        }
		
		private int GetTrendDirection()
		{
		    try
		    {
		        if (CurrentBar < 50)
		            return 0;
		            
		        double highestHigh = MAX(High, 50)[0];
		        double lowestLow = MIN(Low, 50)[0];
		        double pricePosition = (Close[0] - lowestLow) / (highestHigh - lowestLow);
		        
		        if (pricePosition > 0.8)
		            return 1;
		        else if (pricePosition < 0.2)
		            return -1;
		        else
		            return 0;
		    }
		    catch
		    {
		        return 0;
		    }
		}
        
        #endregion
        
        #region Position Management and Communication
        
		private void SendPositionUpdate()
		{
		    try
		    {
		        int actualPosition = GetCurrentPosition();
		        string positionJson = $"{{\"position\":{actualPosition}}}";
		        TransmitData(sendSock, positionJson);
		    }
		    catch (Exception ex)
		    {
		        if (EnableLogging)
		            Print($"Position update error: {ex.Message}");
		    }
		}
		
		private int GetCurrentPosition()
		{
		    if (Position == null)
		        return 0;
		        
		    switch (Position.MarketPosition)
		    {
		        case MarketPosition.Long:
		            return Position.Quantity;
		        case MarketPosition.Short:
		            return -Position.Quantity;
		        default:
		            return 0;
		    }
		}
        
        #endregion
        
        #region Network Communication (Simplified)
        
        private void SignalReceiveLoop()
        {
            var stream = recvSock.GetStream();
            byte[] lengthBuffer = new byte[4];
            
            while (running)
            {
                try
                {
                    if (!ReadExactBytes(stream, lengthBuffer, 4))
                        break;
                        
                    int messageLength = BitConverter.ToInt32(lengthBuffer, 0);
                    if (messageLength > 10000)
                        continue;
                        
                    byte[] messageBuffer = new byte[messageLength];
                    
                    if (!ReadExactBytes(stream, messageBuffer, messageLength))
                        continue;
                        
                    ProcessReceivedSignal(messageBuffer);
                }
                catch (Exception ex)
                {
                    if (EnableLogging)
                        Print($"Signal receive error: {ex.Message}");
                    Thread.Sleep(1000);
                }
            }
        }
        
        private void LwpeReceiveLoop()
        {
            var stream = lwpeSock.GetStream();
            byte[] buffer = new byte[1024];
            
            while (running)
            {
                try
                {
                    int bytesRead = stream.Read(buffer, 0, buffer.Length);
                    if (bytesRead > 0)
                    {
                        ProcessLwpeData(buffer, bytesRead);
                    }
                }
                catch (Exception ex)
                {
                    if (EnableLogging)
                        Print($"LWPE receive error: {ex.Message}");
                    Thread.Sleep(1000);
                }
            }
        }
        
        private bool ReadExactBytes(NetworkStream stream, byte[] buffer, int count)
        {
            int totalRead = 0;
            while (totalRead < count)
            {
                int bytesRead = stream.Read(buffer, totalRead, count - totalRead);
                if (bytesRead == 0)
                    return false;
                totalRead += bytesRead;
            }
            return true;
        }
        
		private void ProcessReceivedSignal(byte[] messageBuffer)
		{
		    try
		    {
		        string jsonString = Encoding.UTF8.GetString(messageBuffer);
		        var signalDict = serializer.Deserialize<Dictionary<string, object>>(jsonString);
		        
			    lock (signalLock)
			    {
			        latestSignal = new SignalData
			        {
			            Action = Convert.ToInt32(signalDict["action"]),
			            Confidence = Convert.ToDouble(signalDict["confidence"]),
			            Quality = signalDict.ContainsKey("signal_quality") ? signalDict["signal_quality"].ToString() : "unknown",
			            Timestamp = Convert.ToInt64(signalDict["timestamp"])
			        };
			
			        if (EnableLogging)
			        {
			            string actionName = latestSignal.Action == 1 ? "Long" : (latestSignal.Action == 2 ? "Short" : "Hold/Exit");
			            Print($"Signal: {actionName}, conf={latestSignal.Confidence:F3}, quality={latestSignal.Quality}");
			        }
			    }
		    }
		    catch (Exception ex)
		    {
		        Print($"Signal processing error: {ex.Message}");
		    }
		}
        
        private void ProcessLwpeData(byte[] buffer, int bytesRead)
        {
            try
            {
                string valueString = Encoding.UTF8.GetString(buffer, 0, bytesRead).Trim();
                
                if (double.TryParse(valueString, NumberStyles.Any, CultureInfo.InvariantCulture, out double lwpeValue))
                {
                    lock (lwpeLock)
                    {
                        currentLWPE = Math.Max(0, Math.Min(1, lwpeValue));
                    }
                }
            }
            catch (Exception ex)
            {
                if (EnableLogging)
                    Print($"LWPE processing error: {ex.Message}");
            }
        }
        
        private void TransmitData(TcpClient client, string data)
        {
            try
            {
                if (client?.Connected != true)
                    return;
                    
                byte[] payload = Encoding.UTF8.GetBytes(data);
                byte[] header = BitConverter.GetBytes(payload.Length);
                
                var stream = client.GetStream();
                stream.Write(header, 0, 4);
                stream.Write(payload, 0, payload.Length);
            }
            catch (Exception ex)
            {
                if (EnableLogging)
                    Print($"Data transmission error: {ex.Message}");
            }
        }
        
        #endregion
        
        #region Helper Classes
        
        private class SignalData
        {
            public int Action { get; set; }
            public double Confidence { get; set; }
            public string Quality { get; set; }
            public long Timestamp { get; set; }
        }
        
        private class FeatureVector
        {
            public double Close { get; set; }
            public double NormalizedVolume { get; set; }
            public double TenkanKijunSignal { get; set; }
            public double PriceCloudSignal { get; set; }
            public double FutureCloudSignal { get; set; }
            public double EmaCrossSignal { get; set; }
            public double TenkanMomentum { get; set; }
            public double KijunMomentum { get; set; }
            public double LWPE { get; set; }
            public bool IsLive { get; set; }
        }

        #endregion
        
        #region Public Methods for Monitoring
        
        public string GetStrategyStatus()
        {
            try
            {
                return $"RLTrader #{instanceId} | " +
                       $"Signals: {signalCount} | " +
                       $"Trades: {tradesExecuted} | " +
                       $"Position: {GetCurrentPosition()} | " +
                       $"SL Hits: {stopLossHits} | " +
                       $"TP Hits: {takeProfitHits} | " +
                       $"TS Hits: {trailingStopHits} | " +
                       $"Scale Outs: {scaleOutExecutions}";
            }
            catch
            {
                return "Status unavailable";
            }
        }
        
        #endregion
    }
}