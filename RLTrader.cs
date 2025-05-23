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
        [Range(5, 50)]
        [Display(Name = "EMA Fast Period", Description = "Fast EMA period", Order = 2, GroupName = "Indicators")]
        public int EmaFastPeriod { get; set; } = 12;

        [NinjaScriptProperty]
        [Range(10, 100)]
        [Display(Name = "EMA Slow Period", Description = "Slow EMA period", Order = 3, GroupName = "Indicators")]
        public int EmaSlowPeriod { get; set; } = 26;

        [NinjaScriptProperty]
        [Range(5, 20)]
        [Display(Name = "Tenkan Period", Description = "Ichimoku Tenkan period", Order = 4, GroupName = "Indicators")]
        public int TenkanPeriod { get; set; } = 9;

        [NinjaScriptProperty]
        [Range(15, 50)]
        [Display(Name = "Kijun Period", Description = "Ichimoku Kijun period", Order = 5, GroupName = "Indicators")]
        public int KijunPeriod { get; set; } = 26;

        [NinjaScriptProperty]
        [Range(25, 100)]
        [Display(Name = "Senkou Period", Description = "Ichimoku Senkou period", Order = 6, GroupName = "Indicators")]
        public int SenkouPeriod { get; set; } = 52;

        [NinjaScriptProperty]
        [Range(0.1, 1.0)]
        [Display(Name = "Min Confidence", Description = "Minimum confidence threshold for trading", Order = 7, GroupName = "Risk Management")]
        public double MinConfidence { get; set; } = 0.6;

        [NinjaScriptProperty]
        [Display(Name = "Enable Logging", Description = "Enable detailed logging", Order = 8, GroupName = "Debug")]
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
                        Print($"RLTrader #{instanceId} started in real-time mode with Ichimoku/EMA features");
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
		    // Assign unique instance ID
		    instanceId = ++instanceCounter;
		    
		    Name = "RLTrader";
		    Description = "Reinforcement Learning Trading Strategy with Ichimoku/EMA Features v2.1";
		    Calculate = Calculate.OnBarClose;
		    
		    // Chart configuration
		    IsOverlay = false;  // Set to FALSE so it shows in a separate panel
		    DisplayInDataBox = true;
		    
		    // LWPE and Signal Quality plots
		    AddPlot(Brushes.Blue, "LWPE");
		    AddPlot(Brushes.Green, "Signal Quality");
		    
		    // Entry configuration
		    BarsRequiredToTrade = Math.Max(SenkouPeriod + 5, EmaSlowPeriod + 5);
		    EntriesPerDirection = 10;
		    EntryHandling = EntryHandling.AllEntries;
		    
		    // Reset state flags for new instance
		    isTerminated = false;
		    socketsStarted = false;
		    running = false;
		    signalCount = 0;
		    tradesExecuted = 0;
		}
        
		private void InitializeIndicators()
		{
		    try
		    {
		        // Initialize EMAs
		        emaFast = EMA(EmaFastPeriod);
		        emaSlow = EMA(EmaSlowPeriod);
		        lwpeSeries = new Series<double>(this);
		        
		        // Add only EMAs to the main price chart
		        // Ichimoku will be calculated manually 
		        AddChartIndicator(emaFast);
		        AddChartIndicator(emaSlow);
		        
		        if (EnableLogging)
		        {
		            Print($"Initialized indicators - EMA({EmaFastPeriod},{EmaSlowPeriod}), Ichimoku calculated manually");
		        }
		    }
		    catch (Exception ex)
		    {
		        Print($"Indicator initialization error: {ex.Message}");
		    }
		}
        
        private void InitializeSockets()
        {
            if (socketsStarted)
            {
                return;
            }
            
            try
            {
                ConnectToSockets();
                StartBackgroundThreads();
                socketsStarted = true;
                Print($"RLTrader #{instanceId} connected to Python service - Ready for Ichimoku/EMA signals");
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
            
            if (!shouldCleanup)
            {
                return;
            }
            
            if (socketsStarted)
            {
                LogPerformanceSummary();
                Print($"RLTrader #{instanceId} shutting down");
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
        
        private void LogPerformanceSummary()
        {
            try
            {
                if (strategyStartTime != DateTime.MinValue)
                {
                    TimeSpan uptime = DateTime.Now - strategyStartTime;
                    Print($"=== Performance Summary ===");
                    Print($"Uptime: {uptime.TotalHours:F1} hours");
                    Print($"Signals processed: {signalCount}");
                    Print($"Trades executed: {tradesExecuted}");
                    Print($"Current position: {GetCurrentPosition()}");
                }
            }
            catch (Exception ex)
            {
                Print($"Performance summary error: {ex.Message}");
            }
        }
        
        #endregion
        
        #region Socket Management
        
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
                try
                {
                    socket?.Close();
                }
                catch (Exception ex)
                {
                    Print($"Error closing socket: {ex.Message}");
                }
            }
        }
        
        #endregion
        
        #region Main Trading Logic
        
		protected override void OnBarUpdate()
		{
		    try
		    {
		        UpdatePlots();
		        SendFeatureVector();
		        
		        // ADD signal arrows to chart
		        if (CurrentBar >= BarsRequiredToTrade)
		        {
		            PlotSignalArrows();  // <-- ADD THIS LINE
		        }
		        
		        if (!IsReadyForTrading())
		            return;
		            
		        ProcessLatestSignal();
		        SendPositionUpdate();
		        
		        // Periodic status logging
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
        
		private void UpdatePlots()
		{
		    try
		    {
		        // Plot LWPE value
		        lock (lwpeLock)
		        {
		            Values[0][0] = currentLWPE;
		        }
		        
		        // Calculate and plot signal quality
		        double signalQuality = CalculateSignalQuality();
		        Values[1][0] = signalQuality;
		    }
		    catch (Exception ex)
		    {
		        if (EnableLogging)
		            Print($"Plot update error: {ex.Message}");
		    }
		}
        
		private double CalculateSignalQuality()
		{
		    try
		    {
		        if (CurrentBar < BarsRequiredToTrade)
		            return 0.5;
		        
		        double quality = 0.0;
		        int signalCount = 0;
		        
		        // Get all signals
		        double tkSignal = GetTenkanKijunSignal();
		        double pcSignal = GetPriceCloudSignal();
		        double fcSignal = GetFutureCloudSignal();
		        double emaSignal = GetEmaCrossSignal();
		        double tmSignal = GetTenkanMomentum();
		        double kmSignal = GetKijunMomentum();
		        
		        // Weight signals by importance
		        var signals = new[]
		        {
		            new { value = tkSignal, weight = 0.25 },
		            new { value = pcSignal, weight = 0.25 },
		            new { value = emaSignal, weight = 0.20 },
		            new { value = fcSignal, weight = 0.15 },
		            new { value = tmSignal, weight = 0.075 },
		            new { value = kmSignal, weight = 0.075 }
		        };
		        
		        double totalWeight = 0;
		        double weightedSum = 0;
		        
		        foreach (var signal in signals)
		        {
		            if (signal.value != 0) // Only count active signals
		            {
		                // Convert signal to quality score (1 for bullish, 0 for bearish, 0.5 for neutral)
		                double signalQuality = signal.value > 0 ? 1.0 : 0.0;
		                weightedSum += signalQuality * signal.weight;
		                totalWeight += signal.weight;
		            }
		        }
		        
		        return totalWeight > 0 ? weightedSum / totalWeight : 0.5;
		    }
		    catch
		    {
		        return 0.5;
		    }
		}
        
		private bool IsReadyForTrading()
		{
		    return CurrentBar >= Math.Max(SenkouPeriod, EmaSlowPeriod) && 
		           socketsStarted && 
		           running;
		}
        
		private void LogCurrentStatus()
		{
		    try
		    {
		        var features = CalculateFeatures();
		        
		        // Create array of all signals
		        var signals = new[] { 
		            features.TenkanKijunSignal, 
		            features.PriceCloudSignal, 
		            features.FutureCloudSignal, 
		            features.EmaCrossSignal,
		            features.TenkanMomentum, 
		            features.KijunMomentum 
		        };
		        
		        // Count signals manually (no LINQ needed)
		        int bullish = 0;
		        int bearish = 0;
		        int neutral = 0;
		        
		        for (int i = 0; i < signals.Length; i++)
		        {
		            if (signals[i] > 0)
		                bullish++;
		            else if (signals[i] < 0)
		                bearish++;
		            else
		                neutral++;
		        }
		        
		        Print($"Bar {CurrentBar}: Bull={bullish}, Bear={bearish}, Neutral={neutral}, " +
		              $"LWPE={features.LWPE:F3}, Pos={GetCurrentPosition()}");
		    }
		    catch (Exception ex)
		    {
		        Print($"Status logging error: {ex.Message}");
		    }
		}
        
		private void ProcessLatestSignal()
		{
		    var signal = GetLatestSignal();
		    
		    if (signal == null)
		    {
		        return;
		    }
		    
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
		        // Check timestamp validity
		        var signalDateTime = new DateTime(1970, 1, 1, 0, 0, 0, DateTimeKind.Utc).AddSeconds(signal.Timestamp);
		        var signalLocalTime = signalDateTime.ToLocalTime();
		        var currentTime = DateTime.Now;
		        var timeDiff = (currentTime - signalLocalTime).TotalSeconds;

		        if (Math.Abs(timeDiff) > 120)
		        {
		            if (EnableLogging)
		                Print($"Signal expired: {timeDiff:F1}s old");
		            return false;
		        }
		        
		        // Check confidence threshold
		        if (signal.Confidence < MinConfidence)
		        {
		            if (EnableLogging)
		                Print($"Signal confidence {signal.Confidence:F3} below threshold {MinConfidence:F3}");
		            return false;
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
		        if (signal.Size <= 0 || signal.Size > 20)
		        {
		            if (EnableLogging)
		                Print($"Invalid signal size: {signal.Size}");
		            return;
		        }

		        switch (signal.Action)
		        {
		            case 1: // BUY
		                EnterLong(signal.Size, "IchimokuEMA_Long");
		                if (EnableLogging)
		                    Print($"Executing LONG: size={signal.Size}, conf={signal.Confidence:F3}");
		                break;
		                
		            case 2: // SELL
		                EnterShort(signal.Size, "IchimokuEMA_Short");
		                if (EnableLogging)
		                    Print($"Executing SHORT: size={signal.Size}, conf={signal.Confidence:F3}");
		                break;
		                
		            default: // HOLD
		                if (EnableLogging)
		                    Print($"HOLD signal: conf={signal.Confidence:F3}");
		                break;
		        }
		    }
		    catch (Exception ex)
		    {
		        Print($"Signal execution error: {ex.Message}");
		    }
		}
		
		protected override void OnExecutionUpdate(Execution execution, string executionId, double price, int quantity, MarketPosition marketPosition, string orderId, DateTime time)
		{
		    if (execution.Order != null)
		    {
		        tradesExecuted++;
		        Print($"FILL #{tradesExecuted}: {execution.Order.Name} {quantity} @ {price:F2} | Position: {GetCurrentPosition()}");
		        
		        Core.Globals.RandomDispatcher.BeginInvoke(new Action(() =>
		        {
		            SendPositionUpdate();
		        }));
		    }
		}
        
		private void UpdateLastSignalTime(SignalData signal)
		{
		    lastProcessedTimestamp = signal.Timestamp;
		    lastSignalTime = Time[0];
		}
        
        #endregion
        
        #region Market Data Handling
        
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
        
        #region Feature Vector Transmission
        
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
                // Volume normalization
                double volMean = SMA(Volume, 20)[0];
                double volStd = StdDev(Volume, 20)[0];
                double normalizedVolume = volStd != 0 ? (Volume[0] - volMean) / volStd : 0;
                
                // LWPE value
                double lwpeValue;
                lock (lwpeLock)
                {
                    lwpeValue = currentLWPE;
                }
                
                // Calculate all Ichimoku signals
                double tenkanKijunSignal = GetTenkanKijunSignal();
                double priceCloudSignal = GetPriceCloudSignal();
                double futureCloudSignal = GetFutureCloudSignal();
                double emaCrossSignal = GetEmaCrossSignal();
                double tenkanMomentum = GetTenkanMomentum();
                double kijunMomentum = GetKijunMomentum();
                
                return new FeatureVector
                {
                    Close = Close[0],
                    NormalizedVolume = normalizedVolume,
                    TenkanKijunSignal = tenkanKijunSignal,
                    PriceCloudSignal = priceCloudSignal,
                    FutureCloudSignal = futureCloudSignal,
                    EmaCrossSignal = emaCrossSignal,
                    TenkanMomentum = tenkanMomentum,
                    KijunMomentum = kijunMomentum,
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
        
		private double GetTenkanKijunSignal()
		{
		    try
		    {
		        if (CurrentBar < Math.Max(TenkanPeriod, KijunPeriod))
		            return 0;
		            
		        // Calculate Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
		        double tenkanHigh = MAX(High, TenkanPeriod)[0];
		        double tenkanLow = MIN(Low, TenkanPeriod)[0];
		        double tenkan = (tenkanHigh + tenkanLow) / 2;
		        
		        // Calculate Kijun-sen (Base Line): (26-period high + 26-period low) / 2
		        double kijunHigh = MAX(High, KijunPeriod)[0];
		        double kijunLow = MIN(Low, KijunPeriod)[0];
		        double kijun = (kijunHigh + kijunLow) / 2;
		        
		        if (tenkan == 0 || kijun == 0)
		            return 0;
		            
		        double diff = tenkan - kijun;
		        
		        // Use percentage-based threshold for neutral signal
		        double priceRange = Math.Max(High[0] - Low[0], Close[0] * 0.0001);
		        double threshold = priceRange * 0.5;
		        
		        if (diff > threshold)
		            return 1.0;   // Bullish
		        else if (diff < -threshold)
		            return -1.0;  // Bearish
		        else
		            return 0.0;   // Neutral
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
		        if (CurrentBar < SenkouPeriod)
		            return 0;
		            
		        // Calculate Senkou Span A: (Tenkan-sen + Kijun-sen) / 2, plotted 26 periods ahead
		        double tenkanHigh = MAX(High, TenkanPeriod)[0];
		        double tenkanLow = MIN(Low, TenkanPeriod)[0];
		        double tenkan = (tenkanHigh + tenkanLow) / 2;
		        
		        double kijunHigh = MAX(High, KijunPeriod)[0];
		        double kijunLow = MIN(Low, KijunPeriod)[0];
		        double kijun = (kijunHigh + kijunLow) / 2;
		        
		        double senkouA = (tenkan + kijun) / 2;
		        
		        // Calculate Senkou Span B: (52-period high + 52-period low) / 2, plotted 26 periods ahead
		        double senkouBHigh = MAX(High, SenkouPeriod)[0];
		        double senkouBLow = MIN(Low, SenkouPeriod)[0];
		        double senkouB = (senkouBHigh + senkouBLow) / 2;
		        
		        if (senkouA == 0 || senkouB == 0)
		            return 0;
		            
		        double cloudTop = Math.Max(senkouA, senkouB);
		        double cloudBottom = Math.Min(senkouA, senkouB);
		        double cloudThickness = cloudTop - cloudBottom;
		        
		        // Add buffer zone for cleaner signals
		        double buffer = cloudThickness * 0.1; // 10% of cloud thickness
		        
		        if (Close[0] > cloudTop + buffer)
		            return 1.0; // Clearly above cloud (bullish)
		        else if (Close[0] < cloudBottom - buffer)
		            return -1.0; // Clearly below cloud (bearish)
		        else
		            return 0.0; // Inside cloud or near edges (neutral)
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
		            
		        // Calculate current Senkou spans (these represent the "future" cloud)
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
		        double threshold = avgPrice * 0.0001; // 0.01% threshold
		        
		        if (diff > threshold)
		            return 1.0;  // Green cloud (bullish)
		        else if (diff < -threshold)
		            return -1.0; // Red cloud (bearish)
		        else
		            return 0.0;  // Neutral cloud
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
		        double avgEma = (fastEma + slowEma) / 2;
		        double threshold = avgEma * 0.0002; // 0.02% threshold for neutral zone
		        
		        if (diff > threshold)
		            return 1.0;   // Fast above slow (bullish)
		        else if (diff < -threshold)
		            return -1.0;  // Fast below slow (bearish)
		        else
		            return 0.0;   // EMAs converging (neutral)
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
		        if (CurrentBar < TenkanPeriod + 5)
		            return 0;
		            
		        // Calculate current Tenkan-sen
		        double currentTenkanHigh = MAX(High, TenkanPeriod)[0];
		        double currentTenkanLow = MIN(Low, TenkanPeriod)[0];
		        double currentTenkan = (currentTenkanHigh + currentTenkanLow) / 2;
		        
		        // Calculate previous Tenkan-sen (3 bars ago)
		        double previousTenkanHigh = MAX(High, TenkanPeriod)[3];
		        double previousTenkanLow = MIN(Low, TenkanPeriod)[3];
		        double previousTenkan = (previousTenkanHigh + previousTenkanLow) / 2;
		        
		        if (currentTenkan == 0 || previousTenkan == 0)
		            return 0;
		            
		        double change = currentTenkan - previousTenkan;
		        double threshold = currentTenkan * 0.0001; // 0.01% momentum threshold
		        
		        if (change > threshold)
		            return 1.0;   // Rising momentum
		        else if (change < -threshold)
		            return -1.0;  // Falling momentum
		        else
		            return 0.0;   // Flat momentum
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
		            
		        // Calculate current Kijun-sen
		        double currentKijunHigh = MAX(High, KijunPeriod)[0];
		        double currentKijunLow = MIN(Low, KijunPeriod)[0];
		        double currentKijun = (currentKijunHigh + currentKijunLow) / 2;
		        
		        // Calculate previous Kijun-sen (3 bars ago)
		        double previousKijunHigh = MAX(High, KijunPeriod)[3];
		        double previousKijunLow = MIN(Low, KijunPeriod)[3];
		        double previousKijun = (previousKijunHigh + previousKijunLow) / 2;
		        
		        if (currentKijun == 0 || previousKijun == 0)
		            return 0;
		            
		        double change = currentKijun - previousKijun;
		        double threshold = currentKijun * 0.0001; // 0.01% momentum threshold
		        
		        if (change > threshold)
		            return 1.0;   // Rising momentum
		        else if (change < -threshold)
		            return -1.0;  // Falling momentum
		        else
		            return 0.0;   // Flat momentum
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
        
        #endregion
        
        #region Position Management
        
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
        
        #region Network Communication
        
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
                    if (messageLength > 10000) // Sanity check
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
			            Size = Convert.ToInt32(signalDict["size"]),
			            Confidence = Convert.ToDouble(signalDict["confidence"]),
			            Timestamp = Convert.ToInt64(signalDict["timestamp"]),
			            SignalId = signalDict.ContainsKey("signal_id") ? Convert.ToInt32(signalDict["signal_id"]) : 0
			        };
			
			        if (EnableLogging)
			        {
			            string actionName = latestSignal.Action == 1 ? "Long" : (latestSignal.Action == 2 ? "Short" : "Hold");
			            Print($"Signal #{latestSignal.SignalId}: {actionName}, size={latestSignal.Size}, conf={latestSignal.Confidence:F3}");
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
                        currentLWPE = Math.Max(0, Math.Min(1, lwpeValue)); // Clamp to [0,1]
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
            public int Size { get; set; }
            public double Confidence { get; set; }
            public long Timestamp { get; set; }
			public int SignalId { get; set; }
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
		
		private double GetTenkanValue()
		{
		    if (CurrentBar < TenkanPeriod) return 0;
		    double high = MAX(High, TenkanPeriod)[0];
		    double low = MIN(Low, TenkanPeriod)[0];
		    return (high + low) / 2;
		}
		
		private double GetKijunValue()
		{
		    if (CurrentBar < KijunPeriod) return 0;
		    double high = MAX(High, KijunPeriod)[0];
		    double low = MIN(Low, KijunPeriod)[0];
		    return (high + low) / 2;
		}
		
		private double GetSenkouSpanA()
		{
		    return (GetTenkanValue() + GetKijunValue()) / 2;
		}
		
		private double GetSenkouSpanB()
		{
		    if (CurrentBar < SenkouPeriod) return 0;
		    double high = MAX(High, SenkouPeriod)[0];
		    double low = MIN(Low, SenkouPeriod)[0];
		    return (high + low) / 2;
		}
		
		private void PlotSignalArrows()
		{
		    try
		    {
		        // Get current signals
		        double tkSignal = GetTenkanKijunSignal();
		        double pcSignal = GetPriceCloudSignal();
		        double emaSignal = GetEmaCrossSignal();
		        double fcSignal = GetFutureCloudSignal();
		        double tmSignal = GetTenkanMomentum();
		        double kmSignal = GetKijunMomentum();
		        
		        // Count aligned signals
		        double bullishCount = 0;
		        double bearishCount = 0;
		        double neutralCount = 0;
		        
		        // Major signals (weighted more heavily)
		        var majorSignals = new[] { tkSignal, pcSignal, emaSignal };
		        var minorSignals = new[] { fcSignal, tmSignal, kmSignal };
		        
		        // Count major signals
		        foreach (double signal in majorSignals)
		        {
		            if (signal > 0) bullishCount++;
		            else if (signal < 0) bearishCount++;
		            else neutralCount++;
		        }
		        
		        // Add minor signals with half weight
		        foreach (double signal in minorSignals)
		        {
		            if (signal > 0) bullishCount += 0.5;
		            else if (signal < 0) bearishCount += 0.5;
		            else neutralCount += 0.5;
		        }
		        
		        // Calculate signal quality score
		        double signalQuality = CalculateSignalQuality();
		        
		        // Strong bullish setup (2+ major bullish OR high signal quality)
		        if (bullishCount >= 2 || (bullishCount >= 1.5 && signalQuality > 0.7))
		        {
		            Draw.ArrowUp(this, "BullArrow" + CurrentBar, false, 0, Low[0] - 4 * TickSize, Brushes.Lime);
		            Draw.Text(this, "BullText" + CurrentBar, $"BULL {bullishCount:F1}", 0, Low[0] - 8 * TickSize, Brushes.Lime);
		            
		            if (EnableLogging && CurrentBar % 20 == 0)
		                Print($"BULLISH SETUP: TK={tkSignal}, PC={pcSignal}, EMA={emaSignal}, Quality={signalQuality:F2}");
		        }
		        // Strong bearish setup (2+ major bearish OR high signal quality)
		        else if (bearishCount >= 2 || (bearishCount >= 1.5 && signalQuality < 0.3))
		        {
		            Draw.ArrowDown(this, "BearArrow" + CurrentBar, false, 0, High[0] + 4 * TickSize, Brushes.Red);
		            Draw.Text(this, "BearText" + CurrentBar, $"BEAR {bearishCount:F1}", 0, High[0] + 8 * TickSize, Brushes.Red);
		            
		            if (EnableLogging && CurrentBar % 20 == 0)
		                Print($"BEARISH SETUP: TK={tkSignal}, PC={pcSignal}, EMA={emaSignal}, Quality={signalQuality:F2}");
		        }
		        // Mixed/neutral signals (1+ neutral signals OR moderate signal quality)
		        else if (neutralCount >= 1 || (signalQuality > 0.4 && signalQuality < 0.6))
		        {
		            Draw.Diamond(this, "NeutralDiamond" + CurrentBar, false, 0, Close[0], Brushes.Yellow);
		            Draw.Text(this, "NeutralText" + CurrentBar, $"NEUTRAL {neutralCount:F1}", 0, Close[0] + 2 * TickSize, Brushes.Yellow);
		            
		            if (EnableLogging && CurrentBar % 50 == 0)
		                Print($"NEUTRAL SETUP: TK={tkSignal}, PC={pcSignal}, EMA={emaSignal}, Quality={signalQuality:F2}");
		        }
		        
		        // Add trend strength indicator
		        if (CurrentBar % 10 == 0) // Every 10 bars
		        {
		            string trendStrength = "";
		            if (signalQuality > 0.8) trendStrength = "STRONG";
		            else if (signalQuality > 0.6) trendStrength = "MODERATE";
		            else if (signalQuality < 0.2) trendStrength = "STRONG BEAR";
		            else if (signalQuality < 0.4) trendStrength = "MODERATE BEAR";
		            else trendStrength = "WEAK/MIXED";
		            
		            if (!string.IsNullOrEmpty(trendStrength))
		            {
						Draw.TextFixed(this, "TrendStrength" + CurrentBar, 
						    $"Trend: {trendStrength} | Quality: {signalQuality:F2}", 
						    TextPosition.TopRight);
		            }
		        }
		    }
		    catch (Exception ex)
		    {
		        if (EnableLogging)
		            Print($"Signal arrow error: {ex.Message}");
		    }
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
                       $"Ready: {IsReadyForTrading()} | " +
                       $"LWPE: {currentLWPE:F3}";
            }
            catch
            {
                return "Status unavailable";
            }
        }
        
        public Dictionary<string, object> GetDetailedStatus()
        {
            try
            {
                var features = CalculateFeatures();
                
                return new Dictionary<string, object>
                {
                    ["instance_id"] = instanceId,
                    ["signals_processed"] = signalCount,
                    ["trades_executed"] = tradesExecuted,
                    ["current_position"] = GetCurrentPosition(),
                    ["is_ready"] = IsReadyForTrading(),
                    ["current_bar"] = CurrentBar,
                    ["ichimoku_signals"] = new Dictionary<string, object>
                    {
                        ["tenkan_kijun"] = features.TenkanKijunSignal,
                        ["price_cloud"] = features.PriceCloudSignal,
                        ["future_cloud"] = features.FutureCloudSignal,
                        ["tenkan_momentum"] = features.TenkanMomentum,
                        ["kijun_momentum"] = features.KijunMomentum
                    },
                    ["ema_signal"] = features.EmaCrossSignal,
                    ["lwpe"] = features.LWPE,
                    ["signal_quality"] = CalculateSignalQuality(),
                    ["uptime_hours"] = strategyStartTime != DateTime.MinValue ? 
                        (DateTime.Now - strategyStartTime).TotalHours : 0
                };
            }
            catch (Exception ex)
            {
                return new Dictionary<string, object> { ["error"] = ex.Message };
            }
        }
        
        #endregion
    }
}