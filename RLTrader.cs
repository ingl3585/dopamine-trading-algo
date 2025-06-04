// RLTrader.cs

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
using NinjaTrader.Gui.Tools; 

namespace NinjaTrader.NinjaScript.Strategies
{
    public class RLTrader : Strategy
    {
        #region Private Fields
        
        // Multi-timeframe data series
        private Series<double> series15Min;
        private Series<double> series5Min;
        private Series<double> series1Min;
        
        // Multi-timeframe indicators
        // 15-minute timeframe
        private EMA emaFast15;
        private EMA emaSlow15;
        
        // 5-minute timeframe
        private EMA emaFast5;
        private EMA emaSlow5;
        
        // 1-minute timeframe (primary)
        private EMA emaFast1;
        private EMA emaSlow1;
        
        // Position Management (unchanged)
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
        
        // Multi-timeframe tracking
        private DateTime last15MinUpdate = DateTime.MinValue;
        private DateTime last5MinUpdate = DateTime.MinValue;
        private MultiTimeframeFeatures lastFeatures = new MultiTimeframeFeatures();
        
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
        
        #region Properties (expanded for multi-timeframe)
        
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

        // 15-minute timeframe parameters
        [NinjaScriptProperty]
        [Range(5, 50)]
        [Display(Name = "EMA Fast Period 15m", Description = "Fast EMA period for 15-minute", Order = 9, GroupName = "15-Minute Indicators")]
        public int EmaFastPeriod15 { get; set; } = 12;

        [NinjaScriptProperty]
        [Range(10, 100)]
        [Display(Name = "EMA Slow Period 15m", Description = "Slow EMA period for 15-minute", Order = 10, GroupName = "15-Minute Indicators")]
        public int EmaSlowPeriod15 { get; set; } = 26;

        [NinjaScriptProperty]
        [Range(5, 20)]
        [Display(Name = "Tenkan Period 15m", Description = "Ichimoku Tenkan period for 15-minute", Order = 11, GroupName = "15-Minute Indicators")]
        public int TenkanPeriod15 { get; set; } = 9;

        [NinjaScriptProperty]
        [Range(15, 50)]
        [Display(Name = "Kijun Period 15m", Description = "Ichimoku Kijun period for 15-minute", Order = 12, GroupName = "15-Minute Indicators")]
        public int KijunPeriod15 { get; set; } = 26;

        [NinjaScriptProperty]
        [Range(25, 100)]
        [Display(Name = "Senkou Period 15m", Description = "Ichimoku Senkou period for 15-minute", Order = 13, GroupName = "15-Minute Indicators")]
        public int SenkouPeriod15 { get; set; } = 52;

        // 5-minute timeframe parameters
        [NinjaScriptProperty]
        [Range(5, 50)]
        [Display(Name = "EMA Fast Period 5m", Description = "Fast EMA period for 5-minute", Order = 14, GroupName = "5-Minute Indicators")]
        public int EmaFastPeriod5 { get; set; } = 12;

        [NinjaScriptProperty]
        [Range(10, 100)]
        [Display(Name = "EMA Slow Period 5m", Description = "Slow EMA period for 5-minute", Order = 15, GroupName = "5-Minute Indicators")]
        public int EmaSlowPeriod5 { get; set; } = 26;

        [NinjaScriptProperty]
        [Range(5, 20)]
        [Display(Name = "Tenkan Period 5m", Description = "Ichimoku Tenkan period for 5-minute", Order = 16, GroupName = "5-Minute Indicators")]
        public int TenkanPeriod5 { get; set; } = 9;

        [NinjaScriptProperty]
        [Range(15, 50)]
        [Display(Name = "Kijun Period 5m", Description = "Ichimoku Kijun period for 5-minute", Order = 17, GroupName = "5-Minute Indicators")]
        public int KijunPeriod5 { get; set; } = 26;

        [NinjaScriptProperty]
        [Range(25, 100)]
        [Display(Name = "Senkou Period 5m", Description = "Ichimoku Senkou period for 5-minute", Order = 18, GroupName = "5-Minute Indicators")]
        public int SenkouPeriod5 { get; set; } = 52;

        // 1-minute timeframe parameters (primary)
        [NinjaScriptProperty]
        [Range(5, 50)]
        [Display(Name = "EMA Fast Period 1m", Description = "Fast EMA period for 1-minute", Order = 19, GroupName = "1-Minute Indicators")]
        public int EmaFastPeriod1 { get; set; } = 12;

        [NinjaScriptProperty]
        [Range(10, 100)]
        [Display(Name = "EMA Slow Period 1m", Description = "Slow EMA period for 1-minute", Order = 20, GroupName = "1-Minute Indicators")]
        public int EmaSlowPeriod1 { get; set; } = 26;

        [NinjaScriptProperty]
        [Range(5, 20)]
        [Display(Name = "Tenkan Period 1m", Description = "Ichimoku Tenkan period for 1-minute", Order = 21, GroupName = "1-Minute Indicators")]
        public int TenkanPeriod1 { get; set; } = 9;

        [NinjaScriptProperty]
        [Range(15, 50)]
        [Display(Name = "Kijun Period 1m", Description = "Ichimoku Kijun period for 1-minute", Order = 22, GroupName = "1-Minute Indicators")]
        public int KijunPeriod1 { get; set; } = 26;

        [NinjaScriptProperty]
        [Range(25, 100)]
        [Display(Name = "Senkou Period 1m", Description = "Ichimoku Senkou period for 1-minute", Order = 23, GroupName = "1-Minute Indicators")]
        public int SenkouPeriod1 { get; set; } = 52;

        [NinjaScriptProperty]
        [Range(0.1, 1.0)]
        [Display(Name = "Min Confidence", Description = "Minimum confidence threshold for trading", Order = 24, GroupName = "Signal Filtering")]
        public double MinConfidence { get; set; } = 0.45;
		
		[NinjaScriptProperty]
		[Display(Name = "Enable Trend Filter", Description = "Block counter-trend trades", Order = 25, GroupName = "Signal Filtering")]
		public bool EnableTrendFilter { get; set; } = true;
		
		[NinjaScriptProperty]
		[Range(20, 100)]
		[Display(Name = "Trend Period", Description = "Period for trend analysis", Order = 26, GroupName = "Signal Filtering")]
		public int TrendPeriod { get; set; } = 50;

        [NinjaScriptProperty]
        [Display(Name = "Enable Trailing Stops", Description = "Enable trailing stop functionality", Order = 27, GroupName = "Exit Management")]
        public bool EnableTrailingStops { get; set; } = true;

        [NinjaScriptProperty]
        [Display(Name = "Enable Scale Outs", Description = "Enable partial position scaling", Order = 28, GroupName = "Exit Management")]
        public bool EnableScaleOuts { get; set; } = true;

        [NinjaScriptProperty]
        [Range(5, 300)]
        [Display(Name = "Min Hold Time Seconds", Description = "Minimum time to hold position", Order = 29, GroupName = "Exit Management")]
        public int MinHoldTimeSeconds { get; set; } = 30;

        [NinjaScriptProperty]
        [Display(Name = "Enable Logging", Description = "Enable detailed logging", Order = 30, GroupName = "Debug")]
        public bool EnableLogging { get; set; } = true;

        [NinjaScriptProperty]
        [Display(Name = "Enable Multi-Timeframe", Description = "Enable multi-timeframe analysis", Order = 31, GroupName = "Multi-Timeframe")]
        public bool EnableMultiTimeframe { get; set; } = true;
		
		[NinjaScriptProperty]
		[Range(0, 60)]
		[Display(Name = "Warm‑up Bars", Description = "How many 1‑min bars to wait after real‑time starts", Order = 99, GroupName = "Debug")]
		public int WarmUpBars { get; set; } = 5;   // set 0 for immediate trading
        
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
		
		            case State.Configure:
		                AddDataSeries(BarsPeriodType.Minute, 15);
		                AddDataSeries(BarsPeriodType.Minute, 5);
		                BarsRequiredToTrade = 1;
		                break;
		
		            case State.DataLoaded:
		                InitializeMultiTimeframeIndicators();
		                break;
		
		            case State.Historical:
		                if (!socketsStarted) InitializeSockets();
		                break;
		
		            case State.Realtime:
		                if (!socketsStarted) InitializeSockets();
		                strategyStartTime = DateTime.Now;
		                Print($"Multi‑Timeframe RLTrader #{instanceId} started");
		                LogMultiTimeframeParameters();
		                
		                // NEW: Process historical data when transitioning to real-time
		                Print("Starting historical data processing...");
		                ProcessHistoricalData();
		                Print("Historical data processing complete, starting real-time mode");
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
		    instanceId   = ++instanceCounter;
		    Name         = "RLTraderMultiTimeframe";
		    Description  = "Multi‑TF RL Trader — instant start";
		    Calculate    = Calculate.OnBarClose;
		
		    IsOverlay        = false;
		    DisplayInDataBox = true;
		
		    EntriesPerDirection = 1;
		    EntryHandling       = EntryHandling.AllEntries;
		
		    AddPlot(Brushes.Orange, "LWPE");
		    AddPlot(Brushes.Cyan, "Signal_Quality");
		    AddPlot(Brushes.Yellow, "Position_Size");
		    AddPlot(Brushes.Red, "Trend_15m");
		    AddPlot(Brushes.Green, "Momentum_5m");
		    AddPlot(Brushes.Blue, "Entry_1m");
		
		    isTerminated   = false;
		    socketsStarted = false;
		    running        = false;
		    ResetPositionTracking();
		}

        private void LogMultiTimeframeParameters()
        {
            if (!EnableLogging) return;
            
            Print("=== Multi-Timeframe Configuration (27 Features) ===");
            Print($"15-minute: EMA({EmaFastPeriod15}/{EmaSlowPeriod15}), Ichimoku({TenkanPeriod15}/{KijunPeriod15}/{SenkouPeriod15})");
            Print($"5-minute:  EMA({EmaFastPeriod5}/{EmaSlowPeriod5}), Ichimoku({TenkanPeriod5}/{KijunPeriod5}/{SenkouPeriod5})");
            Print($"1-minute:  EMA({EmaFastPeriod1}/{EmaSlowPeriod1}), Ichimoku({TenkanPeriod1}/{KijunPeriod1}/{SenkouPeriod1})");
            Print($"Feature Vector: 27 elements (9 per timeframe)");
            Print($"Trend Context: 15m → Momentum: 5m → Entry: 1m");
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
        
		private void InitializeMultiTimeframeIndicators()
		{
		    try
		    {
		        // Wait for data series to be available
		        if (BarsArray.Length < 3)
		        {
		            Print("Multi-timeframe data series not yet available");
		            return;
		        }
		        
		        // 15-minute indicators (BarsArray[1])
		        if (BarsArray[1] != null)
		        {
		            emaFast15 = EMA(BarsArray[1], EmaFastPeriod15);
		            emaSlow15 = EMA(BarsArray[1], EmaSlowPeriod15);
		        }
		        
		        // 5-minute indicators (BarsArray[2])
		        if (BarsArray[2] != null)
		        {
		            emaFast5 = EMA(BarsArray[2], EmaFastPeriod5);
		            emaSlow5 = EMA(BarsArray[2], EmaSlowPeriod5);
		        }
		        
		        // 1-minute indicators (primary BarsArray[0])
		        emaFast1 = EMA(BarsArray[0], EmaFastPeriod1);
		        emaSlow1 = EMA(BarsArray[0], EmaSlowPeriod1);
		        
		        if (EnableLogging)
		        {
		            Print($"Multi-timeframe indicators initialized for 27-feature analysis");
		        }
		    }
		    catch (Exception ex)
		    {
		        Print($"Multi-timeframe indicator initialization error: {ex.Message}");
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
                Print($"Multi-Timeframe RLTrader #{instanceId} connected - Ready for 27-feature ML signals");
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
                    Print($"=== Final Multi-Timeframe Performance ===");
                    Print($"Uptime: {uptime.TotalHours:F1} hours");
                    Print($"27-Feature Signals: {signalCount}, Trades: {tradesExecuted}");
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
        
        #region Socket Management (unchanged)
        
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
        
        #region Multi-Timeframe Trading Logic
        
		protected override void OnBarUpdate()
		{
		    try
		    {
				if (BarsInProgress != 0 || State != State.Realtime || CurrentBar < WarmUpBars)
				    return;
		            
		        UpdatePlots();
		        
		        // Calculate and send multi-timeframe feature vector
		        if (EnableMultiTimeframe)
		        {
		            SendMultiTimeframeFeatureVector();
		        }
		        else
		        {
		            SendSingleTimeframeFeatureVector();
		        }
		        
		        if (!IsReadyForTrading())
		            return;
		
		        // Manage existing positions first
		        ManageExistingPositions();
		        
		        // Process new signals
		        ProcessLatestSignal();
		        SendPositionUpdate();
		        
		        // Visual updates - ONLY if sufficient data
		        if (IsDataSeriesReady() && CurrentBar >= BarsRequiredToTrade)
		        {
		            PlotMultiTimeframeInfo();
		        }
		        
		        // Periodic logging
		        if (CurrentBar % 100 == 0 && EnableLogging)
		        {
		            LogMultiTimeframeStatus();
		        }
		    }
		    catch (Exception ex)
		    {
		        Print($"OnBarUpdate error: {ex.Message}");
		    }
		}
		
		private bool IsDataSeriesReady()
		{
		    try
		    {
		        // Check if we have enough bars on all timeframes
		        int requiredBars = Math.Max(Math.Max(SenkouPeriod15, SenkouPeriod5), SenkouPeriod1) + 30;
		        
		        // Check primary series (1-minute) 
		        if (CurrentBar < requiredBars)
		        {
		            if (EnableLogging && CurrentBar % 50 == 0)
		                Print($"Waiting for 1m data: {CurrentBar}/{requiredBars} bars");
		            return false;
		        }
		        
		        // Check if we have all data series
		        if (BarsArray == null || BarsArray.Length < 3)
		        {
		            if (EnableLogging && CurrentBar % 50 == 0)
		                Print($"BarsArray not ready - have {BarsArray?.Length ?? 0}, need 3 timeframes");
		            return false;
		        }
		        
		        // Check 15-minute series (BarsArray[1])
		        if (CurrentBars.Length < 2 || CurrentBars[1] < (requiredBars / 15))
		        {
		            if (EnableLogging && CurrentBar % 50 == 0)
		                Print($"Waiting for 15m data: {(CurrentBars.Length > 1 ? CurrentBars[1] : 0)}/{requiredBars/15} bars");
		            return false;
		        }
		            
		        // Check 5-minute series (BarsArray[2])  
		        if (CurrentBars.Length < 3 || CurrentBars[2] < (requiredBars / 5))
		        {
		            if (EnableLogging && CurrentBar % 50 == 0)
		                Print($"Waiting for 5m data: {(CurrentBars.Length > 2 ? CurrentBars[2] : 0)}/{requiredBars/5} bars");
		            return false;
		        }
		        
		        // Check that all required arrays exist
		        if (Closes[1] == null || Closes[2] == null ||
		            Highs[1] == null || Highs[2] == null ||
		            Lows[1] == null || Lows[2] == null ||
		            Volumes[1] == null || Volumes[2] == null)
		        {
		            if (EnableLogging && CurrentBar % 50 == 0)
		                Print("Multi-timeframe data arrays not ready");
		            return false;
		        }
		        
		        // Check indicators are ready
		        if (emaFast15 == null || emaSlow15 == null ||
		            emaFast5 == null || emaSlow5 == null ||
		            emaFast1 == null || emaSlow1 == null)
		        {
		            if (EnableLogging && CurrentBar % 50 == 0)
		                Print("Multi-timeframe indicators not ready");
		            return false;
		        }
		        
		        // Additional safety check - make sure we can actually access the values
		        try
		        {
		            double test15 = Closes[1][0];
		            double test5 = Closes[2][0];
		            double test1 = Close[0];
		            
		            if (test15 <= 0 || test5 <= 0 || test1 <= 0)
		            {
		                if (EnableLogging && CurrentBar % 50 == 0)
		                    Print($"Invalid price data: 15m={test15}, 5m={test5}, 1m={test1}");
		                return false;
		            }
		        }
		        catch (Exception ex)
		        {
		            if (EnableLogging && CurrentBar % 50 == 0)
		                Print($"Data access test failed: {ex.Message}");
		            return false;
		        }
		        
		        return true;
		    }
		    catch (Exception ex)
		    {
		        if (EnableLogging)
		            Print($"IsDataSeriesReady error: {ex.Message}");
		        return false;
		    }
		}
		
		private void ProcessHistoricalData()
		{
		    if (!IsDataSeriesReady())
		    {
		        Print("Data series not ready for historical processing");
		        return;
		    }
		    
		    try
		    {
		        // Determine how many historical bars to process
		        int barsToProcess = Math.Min(500, CurrentBar); // Process up to 500 historical bars
		        int startBar = Math.Max(0, CurrentBar - barsToProcess);
		        
		        Print($"Processing {barsToProcess} historical bars from bar {startBar} to {CurrentBar}");
		        
		        int processedCount = 0;
		        
		        // Loop through historical bars
		        for (int i = startBar; i < CurrentBar; i++)
		        {
		            try
		            {
		                // Calculate features for this historical bar
		                var historicalFeatures = CalculateHistoricalMultiTimeframeFeatures(i);
		                
		                if (historicalFeatures != null)
		                {
		                    string payload = CreateMultiTimeframeFeaturePayload(historicalFeatures);
		                    
		                    // Send historical data (live = 0)
		                    payload = payload.Replace("\"live\":1", "\"live\":0");
		                    
		                    TransmitData(sendSock, payload);
		                    processedCount++;
		                    
		                    // Small delay to avoid overwhelming the Python side
		                    if (processedCount % 50 == 0)
		                    {
		                        Thread.Sleep(10);
		                        Print($"Processed {processedCount} historical bars...");
		                    }
		                }
		            }
		            catch (Exception ex)
		            {
		                if (EnableLogging)
		                    Print($"Error processing historical bar {i}: {ex.Message}");
		            }
		        }
		        
		        Print($"Historical data processing complete: {processedCount} bars sent");
		        
		        // Small delay before starting real-time processing
		        Thread.Sleep(100);
		    }
		    catch (Exception ex)
		    {
		        Print($"Historical data processing error: {ex.Message}");
		    }
		}

		private void SendMultiTimeframeFeatureVector()
		{
		    if (sendSock?.Connected != true)
		        return;
		        
		    try
		    {
		        var features = CalculateMultiTimeframeFeatures();
		        
		        // CRITICAL FIX: Update lastFeatures for use in plotting and trend filtering
		        lastFeatures = features;
		        
		        string payload = CreateMultiTimeframeFeaturePayload(features);
		        TransmitData(sendSock, payload);
		        
		        if (EnableLogging && CurrentBar % 500 == 0)
		        {
		            Print($"Sent 27-feature vector: 15m trend={features.Trend15m:F1}, 5m momentum={features.Momentum5m:F1}, 1m entry={features.Entry1m:F1}");
		        }
		    }
		    catch (Exception ex)
		    {
		        if (EnableLogging)
		            Print($"Multi-timeframe feature transmission error: {ex.Message}");
		    }
		}

        private void SendSingleTimeframeFeatureVector()
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
                    Print($"Single timeframe feature transmission error: {ex.Message}");
            }
        }
		
		private MultiTimeframeFeatures CalculateHistoricalMultiTimeframeFeatures(int barIndex)
		{
		    try
		    {
		        var features = new MultiTimeframeFeatures();
		        
		        // Check if we can safely access all required historical data
		        if (barIndex < 0 || 
		            CurrentBars[0] <= barIndex ||
		            CurrentBars[1] <= (barIndex / 15) ||  // 15-minute bar index
		            CurrentBars[2] <= (barIndex / 5))     // 5-minute bar index
		        {
		            return null;
		        }
		        
		        // Calculate corresponding indices for other timeframes
		        int bar15Index = Math.Min(barIndex / 15, CurrentBars[1] - 1);
		        int bar5Index = Math.Min(barIndex / 5, CurrentBars[2] - 1);
		        
		        // 15-minute features (using barIndex / 15)
		        features.Close15m = Closes[1][CurrentBars[1] - bar15Index - 1];
		        features.NormalizedVolume15m = CalculateHistoricalNormalizedVolume(1, 20, bar15Index);
		        features.TenkanKijunSignal15m = GetHistoricalTenkanKijunSignal(1, TenkanPeriod15, KijunPeriod15, bar15Index);
		        features.PriceCloudSignal15m = GetHistoricalPriceCloudSignal(1, TenkanPeriod15, KijunPeriod15, SenkouPeriod15, bar15Index);
		        features.FutureCloudSignal15m = GetHistoricalFutureCloudSignal(1, TenkanPeriod15, KijunPeriod15, SenkouPeriod15, bar15Index);
		        features.EmaCrossSignal15m = GetHistoricalEmaCrossSignal(emaFast15, emaSlow15, bar15Index);
		        features.TenkanMomentum15m = GetHistoricalTenkanMomentum(1, TenkanPeriod15, bar15Index);
		        features.KijunMomentum15m = GetHistoricalKijunMomentum(1, KijunPeriod15, bar15Index);
		        features.LWPE15m = 0.5; // Use default for historical
		        
		        // 5-minute features (using barIndex / 5)
		        features.Close5m = Closes[2][CurrentBars[2] - bar5Index - 1];
		        features.NormalizedVolume5m = CalculateHistoricalNormalizedVolume(2, 20, bar5Index);
		        features.TenkanKijunSignal5m = GetHistoricalTenkanKijunSignal(2, TenkanPeriod5, KijunPeriod5, bar5Index);
		        features.PriceCloudSignal5m = GetHistoricalPriceCloudSignal(2, TenkanPeriod5, KijunPeriod5, SenkouPeriod5, bar5Index);
		        features.FutureCloudSignal5m = GetHistoricalFutureCloudSignal(2, TenkanPeriod5, KijunPeriod5, SenkouPeriod5, bar5Index);
		        features.EmaCrossSignal5m = GetHistoricalEmaCrossSignal(emaFast5, emaSlow5, bar5Index);
		        features.TenkanMomentum5m = GetHistoricalTenkanMomentum(2, TenkanPeriod5, bar5Index);
		        features.KijunMomentum5m = GetHistoricalKijunMomentum(2, KijunPeriod5, bar5Index);
		        features.LWPE5m = 0.5; // Use default for historical
		        
		        // 1-minute features (using barIndex directly)
		        features.Close1m = Closes[0][CurrentBar - barIndex];
		        features.NormalizedVolume1m = CalculateHistoricalNormalizedVolume(0, 20, barIndex);
		        features.TenkanKijunSignal1m = GetHistoricalTenkanKijunSignal(0, TenkanPeriod1, KijunPeriod1, barIndex);
		        features.PriceCloudSignal1m = GetHistoricalPriceCloudSignal(0, TenkanPeriod1, KijunPeriod1, SenkouPeriod1, barIndex);
		        features.FutureCloudSignal1m = GetHistoricalFutureCloudSignal(0, TenkanPeriod1, KijunPeriod1, SenkouPeriod1, barIndex);
		        features.EmaCrossSignal1m = GetHistoricalEmaCrossSignal(emaFast1, emaSlow1, barIndex);
		        features.TenkanMomentum1m = GetHistoricalTenkanMomentum(0, TenkanPeriod1, barIndex);
		        features.KijunMomentum1m = GetHistoricalKijunMomentum(0, KijunPeriod1, barIndex);
		        features.LWPE1m = 0.5; // Use default for historical
		        
		        // Calculate multi-timeframe alignment signals
		        features.Trend15m = CalculateTrendAlignment15m(features);
		        features.Momentum5m = CalculateMomentumAlignment5m(features);
		        features.Entry1m = CalculateEntryAlignment1m(features);
		        
		        features.IsLive = false; // Historical data
		        
		        return features;
		    }
		    catch (Exception ex)
		    {
		        if (EnableLogging)
		            Print($"Historical feature calculation error for bar {barIndex}: {ex.Message}");
		        return null;
		    }
		}
		
		private double CalculateHistoricalNormalizedVolume(int seriesIndex, int lookback, int barIndex)
		{
		    try
		    {
		        if (barIndex < lookback || CurrentBars[seriesIndex] <= barIndex)
		            return 0;
		            
		        double currentVol = Volumes[seriesIndex][barIndex];
		        double avgVol = 0;
		        
		        for (int i = 0; i < lookback; i++)
		        {
		            if (barIndex + i < CurrentBars[seriesIndex])
		                avgVol += Volumes[seriesIndex][barIndex + i];
		        }
		        avgVol /= lookback;
		        
		        return avgVol != 0 ? (currentVol - avgVol) / avgVol : 0;
		    }
		    catch
		    {
		        return 0;
		    }
		}
		
		private double GetHistoricalTenkanKijunSignal(int seriesIndex, int tenkanPeriod, int kijunPeriod, int barIndex)
		{
		    try
		    {
		        if (barIndex < Math.Max(tenkanPeriod, kijunPeriod) || CurrentBars[seriesIndex] <= barIndex)
		            return 0;
		            
		        double tenkanHigh = MAX(Highs[seriesIndex], tenkanPeriod)[barIndex];
		        double tenkanLow = MIN(Lows[seriesIndex], tenkanPeriod)[barIndex];
		        double tenkan = (tenkanHigh + tenkanLow) / 2;
		        
		        double kijunHigh = MAX(Highs[seriesIndex], kijunPeriod)[barIndex];
		        double kijunLow = MIN(Lows[seriesIndex], kijunPeriod)[barIndex];
		        double kijun = (kijunHigh + kijunLow) / 2;
		        
		        if (tenkan == 0 || kijun == 0)
		            return 0;
		            
		        double diff = tenkan - kijun;
		        double threshold = Closes[seriesIndex][barIndex] * 0.001;
		        
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
		
		private double GetHistoricalEmaCrossSignal(EMA emaFast, EMA emaSlow, int barIndex)
		{
		    try
		    {
		        if (emaFast == null || emaSlow == null || barIndex >= emaFast.Count || barIndex >= emaSlow.Count)
		            return 0;
		            
		        double fastEma = emaFast[barIndex];
		        double slowEma = emaSlow[barIndex];
		        
		        if (fastEma == 0 || slowEma == 0)
		            return 0;
		            
		        double diff = fastEma - slowEma;
		        double threshold = Math.Abs(fastEma) * 0.0005;
		        
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
		
		private double GetHistoricalPriceCloudSignal(int seriesIndex, int tenkanPeriod, int kijunPeriod, int senkouPeriod, int barIndex)
		{
		    try
		    {
		        if (barIndex < senkouPeriod + 26 || CurrentBars[seriesIndex] <= barIndex)
		            return 0;
		            
		        // Calculate cloud components 26 bars ago from the reference point
		        int cloudBarIndex = barIndex + 26;
		        if (cloudBarIndex >= CurrentBars[seriesIndex])
		            return 0;
		            
		        double tenkanHigh26 = MAX(Highs[seriesIndex], tenkanPeriod)[cloudBarIndex];
		        double tenkanLow26 = MIN(Lows[seriesIndex], tenkanPeriod)[cloudBarIndex];
		        double tenkan26 = (tenkanHigh26 + tenkanLow26) / 2;
		        
		        double kijunHigh26 = MAX(Highs[seriesIndex], kijunPeriod)[cloudBarIndex];
		        double kijunLow26 = MIN(Lows[seriesIndex], kijunPeriod)[cloudBarIndex];
		        double kijun26 = (kijunHigh26 + kijunLow26) / 2;
		        
		        double senkouA = (tenkan26 + kijun26) / 2;
		        
		        double senkouBHigh26 = MAX(Highs[seriesIndex], senkouPeriod)[cloudBarIndex];
		        double senkouBLow26 = MIN(Lows[seriesIndex], senkouPeriod)[cloudBarIndex];
		        double senkouB = (senkouBHigh26 + senkouBLow26) / 2;
		        
		        if (senkouA == 0 || senkouB == 0)
		            return 0;
		            
		        double cloudTop = Math.Max(senkouA, senkouB);
		        double cloudBottom = Math.Min(senkouA, senkouB);
		        double currentPrice = Closes[seriesIndex][barIndex];
		        double buffer = currentPrice * 0.002;
		        
		        if (currentPrice > cloudTop + buffer)
		            return 1.0;
		        else if (currentPrice < cloudBottom - buffer)
		            return -1.0;
		        else
		            return 0.0;
		    }
		    catch
		    {
		        return 0;
		    }
		}
		
		private double GetHistoricalFutureCloudSignal(int seriesIndex, int tenkanPeriod, int kijunPeriod, int senkouPeriod, int barIndex)
		{
		    try
		    {
		        if (barIndex < senkouPeriod || CurrentBars[seriesIndex] <= barIndex)
		            return 0;
		            
		        double tenkanHigh = MAX(Highs[seriesIndex], tenkanPeriod)[barIndex];
		        double tenkanLow = MIN(Lows[seriesIndex], tenkanPeriod)[barIndex];
		        double tenkan = (tenkanHigh + tenkanLow) / 2;
		        
		        double kijunHigh = MAX(Highs[seriesIndex], kijunPeriod)[barIndex];
		        double kijunLow = MIN(Lows[seriesIndex], kijunPeriod)[barIndex];
		        double kijun = (kijunHigh + kijunLow) / 2;
		        
		        double futureA = (tenkan + kijun) / 2;
		        
		        double futureBHigh = MAX(Highs[seriesIndex], senkouPeriod)[barIndex];
		        double futureBLow = MIN(Lows[seriesIndex], senkouPeriod)[barIndex];
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
		
		private double GetHistoricalTenkanMomentum(int seriesIndex, int tenkanPeriod, int barIndex)
		{
		    try
		    {
		        if (barIndex < tenkanPeriod + 8 || CurrentBars[seriesIndex] <= barIndex + 5)
		            return 0;
		            
		        // Current Tenkan at barIndex
		        double currentTenkanHigh = MAX(Highs[seriesIndex], tenkanPeriod)[barIndex];
		        double currentTenkanLow = MIN(Lows[seriesIndex], tenkanPeriod)[barIndex];
		        double currentTenkan = (currentTenkanHigh + currentTenkanLow) / 2;
		        
		        // Previous Tenkan 5 bars earlier
		        double previousTenkanHigh = MAX(Highs[seriesIndex], tenkanPeriod)[barIndex + 5];
		        double previousTenkanLow = MIN(Lows[seriesIndex], tenkanPeriod)[barIndex + 5];
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
		
		private double GetHistoricalKijunMomentum(int seriesIndex, int kijunPeriod, int barIndex)
		{
		    try
		    {
		        if (barIndex < kijunPeriod + 5 || CurrentBars[seriesIndex] <= barIndex + 3)
		            return 0;
		            
		        // Current Kijun at barIndex
		        double currentKijunHigh = MAX(Highs[seriesIndex], kijunPeriod)[barIndex];
		        double currentKijunLow = MIN(Lows[seriesIndex], kijunPeriod)[barIndex];
		        double currentKijun = (currentKijunHigh + currentKijunLow) / 2;
		        
		        // Previous Kijun 3 bars earlier
		        double previousKijunHigh = MAX(Highs[seriesIndex], kijunPeriod)[barIndex + 3];
		        double previousKijunLow = MIN(Lows[seriesIndex], kijunPeriod)[barIndex + 3];
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

		private MultiTimeframeFeatures CalculateMultiTimeframeFeatures()
		{
		    try
		    {
		        var features = new MultiTimeframeFeatures();
		        
		        // Ensure we have enough data on all timeframes
		        if (!IsDataSeriesReady())
		        {
		            return GetDefaultMultiTimeframeFeatures();
		        }
		        
		        // 15-minute features (BarsArray[1]) - TREND CONTEXT
		        features.Close15m = Closes[1][0];
		        features.NormalizedVolume15m = CalculateNormalizedVolume(1, 20);
		        features.TenkanKijunSignal15m = GetTenkanKijunSignal(1, TenkanPeriod15, KijunPeriod15);
		        features.PriceCloudSignal15m = GetPriceCloudSignal(1, TenkanPeriod15, KijunPeriod15, SenkouPeriod15);
		        features.FutureCloudSignal15m = GetFutureCloudSignal(1, TenkanPeriod15, KijunPeriod15, SenkouPeriod15);
		        features.EmaCrossSignal15m = GetEmaCrossSignal(emaFast15, emaSlow15);
		        features.TenkanMomentum15m = GetTenkanMomentum(1, TenkanPeriod15);
		        features.KijunMomentum15m = GetKijunMomentum(1, KijunPeriod15);
		        features.LWPE15m = currentLWPE; // Use current LWPE for all timeframes
		        
		        // 5-minute features (BarsArray[2]) - MOMENTUM CONTEXT
		        features.Close5m = Closes[2][0];
		        features.NormalizedVolume5m = CalculateNormalizedVolume(2, 20);
		        features.TenkanKijunSignal5m = GetTenkanKijunSignal(2, TenkanPeriod5, KijunPeriod5);
		        features.PriceCloudSignal5m = GetPriceCloudSignal(2, TenkanPeriod5, KijunPeriod5, SenkouPeriod5);
		        features.FutureCloudSignal5m = GetFutureCloudSignal(2, TenkanPeriod5, KijunPeriod5, SenkouPeriod5);
		        features.EmaCrossSignal5m = GetEmaCrossSignal(emaFast5, emaSlow5);
		        features.TenkanMomentum5m = GetTenkanMomentum(2, TenkanPeriod5);
		        features.KijunMomentum5m = GetKijunMomentum(2, KijunPeriod5);
		        features.LWPE5m = currentLWPE;
		        
		        // 1-minute features (BarsArray[0]) - ENTRY TIMING
		        features.Close1m = Close[0];
		        features.NormalizedVolume1m = CalculateNormalizedVolume(0, 20);
		        features.TenkanKijunSignal1m = GetTenkanKijunSignal(0, TenkanPeriod1, KijunPeriod1);
		        features.PriceCloudSignal1m = GetPriceCloudSignal(0, TenkanPeriod1, KijunPeriod1, SenkouPeriod1);
		        features.FutureCloudSignal1m = GetFutureCloudSignal(0, TenkanPeriod1, KijunPeriod1, SenkouPeriod1);
		        features.EmaCrossSignal1m = GetEmaCrossSignal(emaFast1, emaSlow1);
		        features.TenkanMomentum1m = GetTenkanMomentum(0, TenkanPeriod1);
		        features.KijunMomentum1m = GetKijunMomentum(0, KijunPeriod1);
		        features.LWPE1m = currentLWPE;
		        
		        // Calculate multi-timeframe alignment signals
		        features.Trend15m = CalculateTrendAlignment15m(features);
		        features.Momentum5m = CalculateMomentumAlignment5m(features);
		        features.Entry1m = CalculateEntryAlignment1m(features);
		        
		        features.IsLive = State == State.Realtime;
		        
		        return features;
		    }
		    catch (Exception ex)
		    {
		        if (EnableLogging)
		            Print($"Multi-timeframe feature calculation error: {ex.Message}");
		        return GetDefaultMultiTimeframeFeatures();
		    }
		}

        private double CalculateTrendAlignment15m(MultiTimeframeFeatures features)
        {
            // Strong trend alignment on 15-minute timeframe
            double alignment = 0.0;
            double signals = 0;
            
            if (features.TenkanKijunSignal15m != 0)
            {
                alignment += features.TenkanKijunSignal15m;
                signals++;
            }
            
            if (features.PriceCloudSignal15m != 0)
            {
                alignment += features.PriceCloudSignal15m * 1.5; // Higher weight for cloud
                signals += 1.5;
            }
            
            if (features.EmaCrossSignal15m != 0)
            {
                alignment += features.EmaCrossSignal15m;
                signals++;
            }
            
            return signals > 0 ? Math.Max(-1, Math.Min(1, alignment / signals)) : 0;
        }

        private double CalculateMomentumAlignment5m(MultiTimeframeFeatures features)
        {
            // Momentum alignment on 5-minute timeframe
            double alignment = 0.0;
            double signals = 0;
            
            if (features.TenkanKijunSignal5m != 0)
            {
                alignment += features.TenkanKijunSignal5m;
                signals++;
            }
            
            if (features.EmaCrossSignal5m != 0)
            {
                alignment += features.EmaCrossSignal5m;
                signals++;
            }
            
            // Add momentum factors
            if (features.TenkanMomentum5m != 0)
            {
                alignment += features.TenkanMomentum5m * 0.5;
                signals += 0.5;
            }
            
            if (features.KijunMomentum5m != 0)
            {
                alignment += features.KijunMomentum5m * 0.5;
                signals += 0.5;
            }
            
            return signals > 0 ? Math.Max(-1, Math.Min(1, alignment / signals)) : 0;
        }

        private double CalculateEntryAlignment1m(MultiTimeframeFeatures features)
        {
            // Entry timing alignment on 1-minute timeframe
            double alignment = 0.0;
            int signals = 0;
            
            if (features.TenkanKijunSignal1m != 0)
            {
                alignment += features.TenkanKijunSignal1m;
                signals++;
            }
            
            if (features.PriceCloudSignal1m != 0)
            {
                alignment += features.PriceCloudSignal1m;
                signals++;
            }
            
            if (features.EmaCrossSignal1m != 0)
            {
                alignment += features.EmaCrossSignal1m;
                signals++;
            }
            
            return signals > 0 ? Math.Max(-1, Math.Min(1, alignment / signals)) : 0;
        }

		private double CalculateNormalizedVolume(int seriesIndex, int lookback)
		{
		    try
		    {
		        // SAFETY: Check if we have enough data
		        if (seriesIndex >= CurrentBars.Length)
		            return 0;
		            
		        if (CurrentBars[seriesIndex] < lookback)
		            return 0;
		            
		        // SAFETY: Check if volumes array exists
		        if (Volumes[seriesIndex] == null)
		            return 0;
		            
		        double currentVol = Volumes[seriesIndex][0];
		        double avgVol = 0;
		        
		        for (int i = 0; i < lookback && i <= CurrentBars[seriesIndex]; i++)
		        {
		            avgVol += Volumes[seriesIndex][i];
		        }
		        avgVol /= lookback;
		        
		        if (avgVol == 0) return 0;
		        
		        return (currentVol - avgVol) / avgVol;
		    }
		    catch (Exception ex)
		    {
		        if (EnableLogging)
		            Print($"CalculateNormalizedVolume error for series {seriesIndex}: {ex.Message}");
		        return 0;
		    }
		}

		private MultiTimeframeFeatures GetDefaultMultiTimeframeFeatures()
		{
		    double lwpeValue;
		    lock (lwpeLock)
		    {
		        lwpeValue = currentLWPE;
		    }
		    
		    return new MultiTimeframeFeatures
		    {
		        // 15-minute defaults - SAFE ACCESS ONLY
		        Close15m = (CurrentBars.Length > 1 && CurrentBars[1] >= 0) ? Closes[1][0] : Close[0],
		        NormalizedVolume15m = 0,
		        TenkanKijunSignal15m = 0,
		        PriceCloudSignal15m = 0,
		        FutureCloudSignal15m = 0,
		        EmaCrossSignal15m = 0,
		        TenkanMomentum15m = 0,
		        KijunMomentum15m = 0,
		        LWPE15m = lwpeValue,
		        
		        // 5-minute defaults - SAFE ACCESS ONLY  
		        Close5m = (CurrentBars.Length > 2 && CurrentBars[2] >= 0) ? Closes[2][0] : Close[0],
		        NormalizedVolume5m = 0,
		        TenkanKijunSignal5m = 0,
		        PriceCloudSignal5m = 0,
		        FutureCloudSignal5m = 0,
		        EmaCrossSignal5m = 0,
		        TenkanMomentum5m = 0,
		        KijunMomentum5m = 0,
		        LWPE5m = lwpeValue,
		        
		        // 1-minute defaults (unchanged)
		        Close1m = Close[0],
		        NormalizedVolume1m = 0,
		        TenkanKijunSignal1m = 0,
		        PriceCloudSignal1m = 0,
		        FutureCloudSignal1m = 0,
		        EmaCrossSignal1m = 0,
		        TenkanMomentum1m = 0,
		        KijunMomentum1m = 0,
		        LWPE1m = lwpeValue,
		        
		        // Alignment signals
		        Trend15m = 0,
		        Momentum5m = 0,
		        Entry1m = 0,
		        
		        IsLive = State == State.Realtime
		    };
		}

        // Multi-timeframe Ichimoku calculations
		private double GetTenkanKijunSignal(int seriesIndex, int tenkanPeriod, int kijunPeriod)
		{
		    try
		    {
		        // ONLY ADD: Basic bounds check to prevent crash
		        if (seriesIndex >= CurrentBars.Length || CurrentBars[seriesIndex] < Math.Max(tenkanPeriod, kijunPeriod))
		            return 0;
		            
		        if (Highs[seriesIndex] == null || Lows[seriesIndex] == null)
		            return 0;
		            
		        // YOUR ORIGINAL LOGIC UNCHANGED
		        double tenkanHigh = MAX(Highs[seriesIndex], tenkanPeriod)[0];
		        double tenkanLow = MIN(Lows[seriesIndex], tenkanPeriod)[0];
		        double tenkan = (tenkanHigh + tenkanLow) / 2;
		        
		        double kijunHigh = MAX(Highs[seriesIndex], kijunPeriod)[0];
		        double kijunLow = MIN(Lows[seriesIndex], kijunPeriod)[0];
		        double kijun = (kijunHigh + kijunLow) / 2;
		        
		        if (tenkan == 0 || kijun == 0)
		            return 0;
		            
		        double diff = tenkan - kijun;
		        double threshold = Closes[seriesIndex][0] * 0.001;
		        
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
        
		private double GetPriceCloudSignal(int seriesIndex, int tenkanPeriod, int kijunPeriod, int senkouPeriod)
		{
		    try
		    {
		        // ONLY ADD: Basic bounds check to prevent crash
		        if (seriesIndex >= CurrentBars.Length || CurrentBars[seriesIndex] < senkouPeriod + 26)
		            return 0;
		            
		        if (Highs[seriesIndex] == null || Lows[seriesIndex] == null || Closes[seriesIndex] == null)
		            return 0;
		            
		        // YOUR ORIGINAL LOGIC UNCHANGED
		        double tenkanHigh26 = MAX(Highs[seriesIndex], tenkanPeriod)[26];
		        double tenkanLow26 = MIN(Lows[seriesIndex], tenkanPeriod)[26];
		        double tenkan26 = (tenkanHigh26 + tenkanLow26) / 2;
		        
		        double kijunHigh26 = MAX(Highs[seriesIndex], kijunPeriod)[26];
		        double kijunLow26 = MIN(Lows[seriesIndex], kijunPeriod)[26];
		        double kijun26 = (kijunHigh26 + kijunLow26) / 2;
		        
		        double senkouA = (tenkan26 + kijun26) / 2;
		        
		        double senkouBHigh26 = MAX(Highs[seriesIndex], senkouPeriod)[26];
		        double senkouBLow26 = MIN(Lows[seriesIndex], senkouPeriod)[26];
		        double senkouB = (senkouBHigh26 + senkouBLow26) / 2;
		        
		        if (senkouA == 0 || senkouB == 0)
		            return 0;
		            
		        double cloudTop = Math.Max(senkouA, senkouB);
		        double cloudBottom = Math.Min(senkouA, senkouB);
		        double buffer = Closes[seriesIndex][0] * 0.002;
		        
		        if (Closes[seriesIndex][0] > cloudTop + buffer)
		            return 1.0;
		        else if (Closes[seriesIndex][0] < cloudBottom - buffer)
		            return -1.0;
		        else
		            return 0.0;
		    }
		    catch
		    {
		        return 0;
		    }
		}
        
        private double GetFutureCloudSignal(int seriesIndex, int tenkanPeriod, int kijunPeriod, int senkouPeriod)
        {
            try
            {
                if (CurrentBars[seriesIndex] < senkouPeriod)
                    return 0;
                    
                double tenkanHigh = MAX(Highs[seriesIndex], tenkanPeriod)[0];
                double tenkanLow = MIN(Lows[seriesIndex], tenkanPeriod)[0];
                double tenkan = (tenkanHigh + tenkanLow) / 2;
                
                double kijunHigh = MAX(Highs[seriesIndex], kijunPeriod)[0];
                double kijunLow = MIN(Lows[seriesIndex], kijunPeriod)[0];
                double kijun = (kijunHigh + kijunLow) / 2;
                
                double futureA = (tenkan + kijun) / 2;
                
                double futureBHigh = MAX(Highs[seriesIndex], senkouPeriod)[0];
                double futureBLow = MIN(Lows[seriesIndex], senkouPeriod)[0];
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
        
        private double GetEmaCrossSignal(EMA emaFast, EMA emaSlow)
        {
            try
            {
                if (emaFast == null || emaSlow == null)
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
        
        private double GetTenkanMomentum(int seriesIndex, int tenkanPeriod)
        {
            try
            {
                if (CurrentBars[seriesIndex] < tenkanPeriod + 8)
                    return 0;
                    
                double currentTenkanHigh = MAX(Highs[seriesIndex], tenkanPeriod)[0];
                double currentTenkanLow = MIN(Lows[seriesIndex], tenkanPeriod)[0];
                double currentTenkan = (currentTenkanHigh + currentTenkanLow) / 2;
                
                double previousTenkanHigh = MAX(Highs[seriesIndex], tenkanPeriod)[5];
                double previousTenkanLow = MIN(Lows[seriesIndex], tenkanPeriod)[5];
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
        
        private double GetKijunMomentum(int seriesIndex, int kijunPeriod)
        {
            try
            {
                if (CurrentBars[seriesIndex] < kijunPeriod + 5)
                    return 0;
                    
                double currentKijunHigh = MAX(Highs[seriesIndex], kijunPeriod)[0];
                double currentKijunLow = MIN(Lows[seriesIndex], kijunPeriod)[0];
                double currentKijun = (currentKijunHigh + currentKijunLow) / 2;
                
                double previousKijunHigh = MAX(Highs[seriesIndex], kijunPeriod)[3];
                double previousKijunLow = MIN(Lows[seriesIndex], kijunPeriod)[3];
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

		private string CreateMultiTimeframeFeaturePayload(MultiTimeframeFeatures features)
		{
		    try
		    {
		        // Validate all features before creating JSON
		        var validatedFeatures = new double[]
		        {
		            // 15-minute features (indices 0-8)
		            ValidatePrice(features.Close15m), 
		            ValidateFloat(features.NormalizedVolume15m), 
		            ValidateSignal(features.TenkanKijunSignal15m),
		            ValidateSignal(features.PriceCloudSignal15m), 
		            ValidateSignal(features.FutureCloudSignal15m), 
		            ValidateSignal(features.EmaCrossSignal15m),
		            ValidateSignal(features.TenkanMomentum15m), 
		            ValidateSignal(features.KijunMomentum15m), 
		            ValidateLWPE(features.LWPE15m),
		            
		            // 5-minute features (indices 9-17)
		            ValidatePrice(features.Close5m), 
		            ValidateFloat(features.NormalizedVolume5m), 
		            ValidateSignal(features.TenkanKijunSignal5m),
		            ValidateSignal(features.PriceCloudSignal5m), 
		            ValidateSignal(features.FutureCloudSignal5m), 
		            ValidateSignal(features.EmaCrossSignal5m),
		            ValidateSignal(features.TenkanMomentum5m), 
		            ValidateSignal(features.KijunMomentum5m), 
		            ValidateLWPE(features.LWPE5m),
		            
		            // 1-minute features (indices 18-26)
		            ValidatePrice(features.Close1m), 
		            ValidateFloat(features.NormalizedVolume1m), 
		            ValidateSignal(features.TenkanKijunSignal1m),
		            ValidateSignal(features.PriceCloudSignal1m), 
		            ValidateSignal(features.FutureCloudSignal1m), 
		            ValidateSignal(features.EmaCrossSignal1m),
		            ValidateSignal(features.TenkanMomentum1m), 
		            ValidateSignal(features.KijunMomentum1m), 
		            ValidateLWPE(features.LWPE1m)
		        };
		
		        // Double-check we have exactly 27 features
		        if (validatedFeatures.Length != 27)
		        {
		            if (EnableLogging)
		                Print($"ERROR: Feature array has {validatedFeatures.Length} elements, expected 27");
		            return "{}"; // Return empty JSON to prevent crash
		        }
		
		        // Create feature array string manually to ensure proper formatting
		        var featureStrings = new string[27];
		        for (int i = 0; i < 27; i++)
		        {
		            if (i >= 2 && i <= 7 || i >= 11 && i <= 16 || i >= 20 && i <= 25) // Signal indices
		            {
		                featureStrings[i] = ((int)validatedFeatures[i]).ToString();
		            }
		            else if (i == 8 || i == 17 || i == 26) // LWPE indices
		            {
		                featureStrings[i] = validatedFeatures[i].ToString("F6", CultureInfo.InvariantCulture);
		            }
		            else // Price and volume indices
		            {
		                featureStrings[i] = validatedFeatures[i].ToString("F6", CultureInfo.InvariantCulture);
		            }
		        }
		
		        string featuresJson = "[" + string.Join(",", featureStrings) + "]";
		
		        return string.Format(CultureInfo.InvariantCulture,
		            @"{{
		                ""features"":{0},
		                ""live"":{1},
		                ""timeframe_alignment"":{{
		                    ""trend_15m"":{2:F3},
		                    ""momentum_5m"":{3:F3},
		                    ""entry_1m"":{4:F3}
		                }}
		            }}",
		            featuresJson,
		            features.IsLive ? 1 : 0,
		            ValidateFloat(features.Trend15m), 
		            ValidateFloat(features.Momentum5m), 
		            ValidateFloat(features.Entry1m));
		    }
		    catch (Exception ex)
		    {
		        if (EnableLogging)
		            Print($"Feature payload creation error: {ex.Message}");
		        return "{}"; // Return empty JSON to prevent crash
		    }
		}
		
		private double ValidatePrice(double price)
		{
		    return (price > 0) ? price : Close[0];
		}
		
		private double ValidateFloat(double value)
		{
		    return double.IsNaN(value) || double.IsInfinity(value) ? 0.0 : value;
		}
		
		private double ValidateSignal(double signal)
		{
		    if (double.IsNaN(signal) || double.IsInfinity(signal))
		        return 0.0;
		    
		    // Use exact integer comparison for ternary signals
		    int roundedSignal = (int)Math.Round(signal);
		    
		    if (roundedSignal > 1)
		        return 1.0;
		    else if (roundedSignal < -1)
		        return -1.0;
		    else
		        return (double)roundedSignal; // Will be -1.0, 0.0, or 1.0
		}
		
		private double ValidateLWPE(double lwpe)
		{
		    if (double.IsNaN(lwpe) || double.IsInfinity(lwpe))
		        return 0.5;
		    
		    return Math.Max(0.0, Math.Min(1.0, lwpe));
		}

        // Legacy single timeframe support
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
                    TenkanKijunSignal = GetTenkanKijunSignal(0, TenkanPeriod1, KijunPeriod1),
                    PriceCloudSignal = GetPriceCloudSignal(0, TenkanPeriod1, KijunPeriod1, SenkouPeriod1),
                    FutureCloudSignal = GetFutureCloudSignal(0, TenkanPeriod1, KijunPeriod1, SenkouPeriod1),
                    EmaCrossSignal = GetEmaCrossSignal(emaFast1, emaSlow1),
                    TenkanMomentum = GetTenkanMomentum(0, TenkanPeriod1),
                    KijunMomentum = GetKijunMomentum(0, KijunPeriod1),
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

        // Position management (same as before)
        private void ManageExistingPositions()
        {
            if (Position.MarketPosition == MarketPosition.Flat)
            {
                if (currentPositionSize != 0)
                {
                    ResetPositionTracking();
                    if (EnableLogging)
                        Print("Position closed - reset tracking");
                }
                return;
            }

            if (EnableTrailingStops && isTrailingStopActive)
            {
                UpdateTrailingStop();
            }

            if (EnableScaleOuts && ShouldScaleOut())
            {
                ExecuteScaleOut();
            }

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

                if (EnableTrailingStops && !isTrailingStopActive)
                {
                    isTrailingStopActive = true;
                    trailingStopPrice = 0;
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
		        lock (lwpeLock)
		        {
		            Values[0][0] = currentLWPE;
		        }
		        
		        double qualityValue = 0.5;
		        switch (lastSignalQuality.ToLower())
		        {
		            case "excellent": qualityValue = 1.0; break;
		            case "good": qualityValue = 0.75; break;
		            case "poor": qualityValue = 0.25; break;
		            default: qualityValue = 0.5; break;
		        }
		        Values[1][0] = qualityValue;
		        Values[2][0] = Math.Abs(GetCurrentPosition());
		        
		        // Multi-timeframe plots - only if enabled and data available
		        if (EnableMultiTimeframe && lastFeatures != null)
		        {
		            Values[3][0] = lastFeatures.Trend15m;     // 15m trend
		            Values[4][0] = lastFeatures.Momentum5m;   // 5m momentum
		            Values[5][0] = lastFeatures.Entry1m;      // 1m entry
		        }
		        else
		        {
		            // Default values when multi-timeframe not ready
		            Values[3][0] = 0;
		            Values[4][0] = 0; 
		            Values[5][0] = 0;
		        }
		    }
		    catch (Exception ex)
		    {
		        if (EnableLogging)
		            Print($"Plot update error: {ex.Message}");
		    }
		}
        
		private bool IsReadyForTrading()
		{
		    return State == State.Realtime
		           && socketsStarted
		           && running
		           && CurrentBar >= WarmUpBars;
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
                var signalDateTime = new DateTime(1970, 1, 1, 0, 0, 0, DateTimeKind.Utc).AddSeconds(signal.Timestamp);
                var signalLocalTime = signalDateTime.ToLocalTime();
                var timeDiff = (DateTime.Now - signalLocalTime).TotalSeconds;

                if (Math.Abs(timeDiff) > 120)
                {
                    if (EnableLogging)
                        Print($"Signal expired: {timeDiff:F1}s old");
                    return false;
                }
                
                if (signal.Confidence < MinConfidence)
                {
                    if (EnableLogging)
                        Print($"Signal confidence {signal.Confidence:F3} below threshold {MinConfidence:F3}");
                    return false;
                }

                if (lastEntryTime != DateTime.MinValue && 
                    (DateTime.Now - lastEntryTime).TotalSeconds < MinHoldTimeSeconds)
                {
                    if (EnableLogging)
                        Print($"Signal blocked by minimum hold time");
                    return false;
                }
                
                if (EnableTrendFilter && EnableMultiTimeframe)
                {
                    // Enhanced trend filter using multi-timeframe data
                    if (!IsSignalAlignedWithTrend(signal))
                    {
                        if (EnableLogging)
                            Print($"Signal blocked by multi-timeframe trend filter");
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

        private bool IsSignalAlignedWithTrend(SignalData signal)
        {
            try
            {
                if (lastFeatures == null) return true; // Allow if no multi-timeframe data
                
                // Don't trade against strong 15-minute trend
                if (Math.Abs(lastFeatures.Trend15m) > 0.6)
                {
                    if ((signal.Action == 1 && lastFeatures.Trend15m < -0.6) || 
                        (signal.Action == 2 && lastFeatures.Trend15m > 0.6))
                    {
                        if (EnableLogging)
                            Print($"Signal blocked: Action={signal.Action}, 15m Trend={lastFeatures.Trend15m:F2}");
                        return false;
                    }
                }
                
                return true;
            }
            catch
            {
                return true; // Allow on error
            }
        }

        private void ExecuteSignal(SignalData signal)
        {
            try
            {
                lastSignalConfidence = signal.Confidence;
                lastSignalQuality = signal.Quality ?? "unknown";

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
                            if (signal.Confidence < HighConfidenceThreshold)
                            {
                                if (Position.MarketPosition == MarketPosition.Long)
                                    ExitLong("ML_Exit");
                                else
                                    ExitShort("ML_Exit");
                                
                                if (EnableLogging)
                                    Print($"FULL EXIT: conf={signal.Confidence:F3}, quality={signal.Quality}");
                            }
                            else if (EnableScaleOuts && Position.Quantity > 1)
                            {
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

                if (EnableTrailingStops && signal.Confidence >= HighConfidenceThreshold)
                {
                    isTrailingStopActive = true;
                    trailingStopPrice = 0;
                    
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

		            if (marketPosition == MarketPosition.Flat)
		            {
		                ResetPositionTracking();
		                if (EnableLogging)
		                    Print($"Position FLAT - tracking reset");
		            }

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

		private void PlotMultiTimeframeInfo()
		{
		    try
		    {
		        // SAFETY: Only plot if we have sufficient data
		        if (!IsDataSeriesReady())
		            return;
		            
		        if (Position.MarketPosition != MarketPosition.Flat && entryPrice > 0)
		        {
		            Draw.HorizontalLine(this, "EntryLine", entryPrice, Brushes.Yellow);
		            
		            if (stopLossPrice > 0)
		            {
		                Draw.HorizontalLine(this, "StopLossLine", stopLossPrice, Brushes.Red);
		            }
		            
		            if (takeProfitPrice > 0)
		            {
		                Draw.HorizontalLine(this, "TakeProfitLine", takeProfitPrice, Brushes.Lime);
		            }
		            
		            if (isTrailingStopActive && trailingStopPrice > 0)
		            {
		                Draw.HorizontalLine(this, "TrailingStopLine", trailingStopPrice, Brushes.Orange);
		            }
		
		            string positionInfo = $"Pos: {Position.Quantity} | Conf: {lastSignalConfidence:F2} | Qual: {lastSignalQuality}";
		            
		            if (EnableMultiTimeframe && lastFeatures != null)
		            {
		                positionInfo += $"\n15m: {lastFeatures.Trend15m:F2} | 5m: {lastFeatures.Momentum5m:F2} | 1m: {lastFeatures.Entry1m:F2}";
		            }
		            
		            Draw.TextFixed(this, "PositionInfo", positionInfo, TextPosition.TopLeft, 
		                         Brushes.White, new NinjaTrader.Gui.Tools.SimpleFont("Arial", 10), Brushes.Black, Brushes.Transparent, 0);
		        }
		        else
		        {
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
		            Print($"Multi-timeframe plotting error: {ex.Message}");
		    }
		}

        private void LogMultiTimeframeStatus()
        {
            try
            {
                string positionStatus = Position.MarketPosition == MarketPosition.Flat ? "FLAT" :
                                      $"{Position.MarketPosition} {Position.Quantity}";

                string status = $"Bar {CurrentBar}: {positionStatus} | LWPE={currentLWPE:F3} | " +
                               $"LastConf={lastSignalConfidence:F3} | Signals={signalCount} | Trades={tradesExecuted}";

                if (EnableMultiTimeframe && lastFeatures != null)
                {
                    status += $" | 15m={lastFeatures.Trend15m:F2} | 5m={lastFeatures.Momentum5m:F2} | 1m={lastFeatures.Entry1m:F2}";
                }

                Print(status);
                      
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
                Print($"Multi-timeframe status logging error: {ex.Message}");
            }
        }

        private void UpdateLastSignalTime(SignalData signal)
        {
            lastProcessedTimestamp = signal.Timestamp;
            lastSignalTime = Time[0];
        }
        
        #endregion
        
        #region Market Data Processing (unchanged)
        
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
        
        #region Network Communication (unchanged)
        
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
			            Print($"Multi-TF Signal: {actionName}, conf={latestSignal.Confidence:F3}, quality={latestSignal.Quality}");
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

        private class MultiTimeframeFeatures
        {
            // 15-minute features (trend context)
            public double Close15m { get; set; }
            public double NormalizedVolume15m { get; set; }
            public double TenkanKijunSignal15m { get; set; }
            public double PriceCloudSignal15m { get; set; }
            public double FutureCloudSignal15m { get; set; }
            public double EmaCrossSignal15m { get; set; }
            public double TenkanMomentum15m { get; set; }
            public double KijunMomentum15m { get; set; }
            public double LWPE15m { get; set; }
            
            // 5-minute features (momentum context)
            public double Close5m { get; set; }
            public double NormalizedVolume5m { get; set; }
            public double TenkanKijunSignal5m { get; set; }
            public double PriceCloudSignal5m { get; set; }
            public double FutureCloudSignal5m { get; set; }
            public double EmaCrossSignal5m { get; set; }
            public double TenkanMomentum5m { get; set; }
            public double KijunMomentum5m { get; set; }
            public double LWPE5m { get; set; }
            
            // 1-minute features (entry timing)
            public double Close1m { get; set; }
            public double NormalizedVolume1m { get; set; }
            public double TenkanKijunSignal1m { get; set; }
            public double PriceCloudSignal1m { get; set; }
            public double FutureCloudSignal1m { get; set; }
            public double EmaCrossSignal1m { get; set; }
            public double TenkanMomentum1m { get; set; }
            public double KijunMomentum1m { get; set; }
            public double LWPE1m { get; set; }
            
            // Multi-timeframe alignment signals
            public double Trend15m { get; set; }     // Overall trend strength
            public double Momentum5m { get; set; }   // Momentum alignment
            public double Entry1m { get; set; }      // Entry timing quality
            
            public bool IsLive { get; set; }
        }

        #endregion
        
        #region Public Methods for Monitoring
        
        public string GetStrategyStatus()
        {
            try
            {
                string status = $"Multi-TF RLTrader #{instanceId} | " +
                               $"Signals: {signalCount} | " +
                               $"Trades: {tradesExecuted} | " +
                               $"Position: {GetCurrentPosition()} | " +
                               $"SL Hits: {stopLossHits} | " +
                               $"TP Hits: {takeProfitHits} | " +
                               $"TS Hits: {trailingStopHits} | " +
                               $"Scale Outs: {scaleOutExecutions}";
                               
                if (EnableMultiTimeframe && lastFeatures != null)
                {
                    status += $" | 15m: {lastFeatures.Trend15m:F2} | 5m: {lastFeatures.Momentum5m:F2} | 1m: {lastFeatures.Entry1m:F2}";
                }
                
                return status;
            }
            catch
            {
                return "Multi-timeframe status unavailable";
            }
        }
        
        #endregion
    }
}