// ResearchStrategy.cs

using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.ComponentModel.DataAnnotations;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using System.Xml.Serialization;
using NinjaTrader.Cbi;
using NinjaTrader.Data;
using NinjaTrader.NinjaScript;
using NinjaTrader.NinjaScript.Strategies;
using NinjaTrader.NinjaScript.Indicators;
using NinjaTrader.Gui.Tools;
using NinjaTrader.NinjaScript.DrawingTools;
using System.Windows.Media;

namespace NinjaTrader.NinjaScript.Strategies
{
    public class ResearchStrategy : Strategy
    {
        #region Properties
        
        [NinjaScriptProperty]
        [Range(0.001, 0.1)]
        [Display(Name = "Risk Percent", Description = "Risk percentage per trade", Order = 1, GroupName = "Risk Management")]
        public double RiskPercent { get; set; }
        
        [NinjaScriptProperty] 
        [Range(1, 100)]
        [Display(Name = "Stop Loss Ticks", Description = "Stop loss distance in ticks", Order = 2, GroupName = "Risk Management")]
        public int StopLossTicks { get; set; }
        
        [NinjaScriptProperty]
        [Range(1, 200)]
        [Display(Name = "Take Profit Ticks", Description = "Take profit distance in ticks", Order = 3, GroupName = "Risk Management")]
        public int TakeProfitTicks { get; set; }
		
		[NinjaScriptProperty]
		[Display(Name = "Allow Scale In", Description = "Allow scaling into positions", Order = 7, GroupName = "Risk Management")]
		public bool AllowScaleIn { get; set; }
		
		[NinjaScriptProperty]
		[Range(2, 5)]
		[Display(Name = "Max Position Pieces", Description = "Maximum scaling pieces", Order = 8, GroupName = "Risk Management")]
		public int MaxPositionPieces { get; set; }
        
        [NinjaScriptProperty]
        [Range(0.1, 1.0)]
        [Display(Name = "Min Confidence", Description = "Minimum ML confidence threshold", Order = 4, GroupName = "ML Settings")]
        public double MinConfidence { get; set; }
        
        [NinjaScriptProperty]
        [Range(1, 10)]
        [Display(Name = "Max Position Size", Description = "Maximum position size", Order = 5, GroupName = "Risk Management")]
        public int MaxPositionSize { get; set; }
		
		[NinjaScriptProperty]
		[Display(Name = "Show Indicators", Description = "Show indicators on chart", Order = 6, GroupName = "Display")]
		public bool ShowIndicators { get; set; }
        
        #endregion
        
        #region Private Fields
        
        private TcpClient featureClient;
        private TcpClient signalClient;
        private Thread signalThread;
        private bool isRunning;
        
        // Data storage for Python communication
        private List<double> prices15m = new List<double>();
        private List<double> volumes15m = new List<double>();
        private List<double> prices5m = new List<double>();
        private List<double> volumes5m = new List<double>();
		private List<double> prices1m = new List<double>();
        private List<double> volumes1m = new List<double>();
        
        // Position tracking
        private double entryPrice;
        private bool hasPosition;
        private int signalCount;
        private int tradesExecuted;
        private int connectionAttempts;
        private DateTime lastConnectionAttempt;
		private DateTime lastTradeEntry;
		private int tradeIdCounter = 0;
		private string currentTradeId = "";
        
        // Connection status
        private bool isConnectedToFeatureServer;
        private bool isConnectedToSignalServer;
        
        #endregion
		
		#region Indicator Variables
		
		private EMA ema20;
		private SMA sma50;
		private Bollinger bb;
		private RSI rsi14;
		private VOL volumeIndicator;
		private ATR atr14;
		
		#endregion
        
        #region Strategy Lifecycle
        
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
                        ConfigureStrategy();
                        break;
                        
                    case State.DataLoaded:
		                if (BarsArray?.Length > 0 && CurrentBars[0] > 0)
		                {
		                    Print("Research Strategy data loaded - preparing for connection");
		                }
		                break;
                        
		            case State.Realtime:
		                // Only print connection info ONCE when actually going live
		                if (!isConnectedToFeatureServer || !isConnectedToSignalServer)
		                {
		                    Print("Research Strategy entering real-time mode");
		                    ConnectToPython();
		                    StartSignalReceiver();
		                    
		                    // Send initial historical data for training
		                    if (isConnectedToFeatureServer)
		                    {
		                        Print($"Sending historical data: 15m={prices15m.Count}, 5m={prices5m.Count}, 1m={prices1m.Count}");
		                        SendMarketDataToPython();
		                    }
		                }
		                break;
                        
		            case State.Terminated:
		                // Only print cleanup message if we were actually running
		                if (signalCount > 0 || tradesExecuted > 0)
		                {
		                    CleanupWithStats();
		                }
		                else
		                {
		                    QuietCleanup(); // Silent cleanup for startup failures
		                }
		                break;
                }
            }
            catch (Exception ex)
            {
                Print($"OnStateChange error in {State}: {ex.Message}");
            }
        }
        
		private void ConfigureDefaults()
		{
		    Description = "Research-aligned strategy: RSI + BB + EMA/SMA + Volume (15m/5m)";
		    Name = "ResearchStrategy";
		    Calculate = Calculate.OnBarClose;
		    
		    // Research-backed default values
		    RiskPercent = 0.02;        // 2% risk per trade (institutional standard)
		    StopLossTicks = 50;        // Simple fixed stop
		    TakeProfitTicks = 100;      // 2:1 reward-to-risk (research optimal)
		    MinConfidence = 0.5;       // Research: 60% accuracy threshold
		    MaxPositionSize = 10;       // Simple position limits
			AllowScaleIn = true;
			MaxPositionPieces = 3;
		    
		    // NinjaTrader settings aligned with research
		    EntriesPerDirection = 1;                           // Simplicity
		    EntryHandling = EntryHandling.AllEntries;         
		    ExitOnSessionCloseSeconds = 30;                   
		    IsFillLimitOnTouch = false;                       
		    MaximumBarsLookBack = MaximumBarsLookBack.TwoHundredFiftySix; // Research: sufficient history
		    OrderFillResolution = OrderFillResolution.Standard;          
		    Slippage = 0;                                     
		    StartBehavior = StartBehavior.WaitUntilFlat;      // Risk management
		    TimeInForce = TimeInForce.Gtc;                    
		    TraceOrders = false;                              // Reduce noise
		    RealtimeErrorHandling = RealtimeErrorHandling.StopCancelClose; // Safety
		    StopTargetHandling = StopTargetHandling.PerEntryExecution;     
		    BarsRequiredToTrade = 0;
			ShowIndicators = true;
		}
        
        private void ConfigureStrategy()
        {
            // Add multi-timeframe data series
            AddDataSeries(BarsPeriodType.Minute, 15);  // BarsArray[1] - 15-minute
            AddDataSeries(BarsPeriodType.Minute, 5);   // BarsArray[2] - 5-minute
			AddDataSeries(BarsPeriodType.Minute, 1);   // BarsArray[3] - 1-minute
			
			if (ShowIndicators)
			{
			    // Initialize indicators
			    ema20 = EMA(BarsArray[0], 20);
			    sma50 = SMA(BarsArray[0], 50);
			    bb = Bollinger(BarsArray[0], 2.0, 20);
			    rsi14 = RSI(BarsArray[0], 14, 3);
			    volumeIndicator = VOL(BarsArray[0]);
				atr14 = ATR(BarsArray[0], 14);
			    
			    // Configure EMA plot
			    ema20.Plots[0].Brush = Brushes.Orange;
			    ema20.Plots[0].Width = 2;
			    ema20.Plots[0].PlotStyle = PlotStyle.Line;
			    
			    // Configure SMA plot
			    sma50.Plots[0].Brush = Brushes.Blue;
			    sma50.Plots[0].Width = 2;
			    sma50.Plots[0].PlotStyle = PlotStyle.Line;
			    
			    // Configure Bollinger Bands
			    bb.Plots[0].Brush = Brushes.Red;  // Upper band
			    bb.Plots[1].Brush = Brushes.Gray;  // Middle band
			    bb.Plots[2].Brush = Brushes.Purple;  // Lower band
			    bb.Plots[0].Width = 1;
			    bb.Plots[1].Width = 1;
			    bb.Plots[2].Width = 1;
			    
			    // Configure RSI
			    rsi14.Plots[0].Brush = Brushes.Purple;
			    rsi14.Plots[0].Width = 2;
				
				// Configure ATR
				atr14.Plots[0].Brush = Brushes.Gray;
			    
			    // Add indicators to chart
			    AddChartIndicator(ema20);
			    AddChartIndicator(sma50);
			    AddChartIndicator(bb);
				AddChartIndicator(atr14);
			    
			    // RSI goes in separate panel
			    AddChartIndicator(rsi14);
			}
        }
        
		protected override void OnBarUpdate()
		{
		    try
		    {
		        if (CurrentBars[0] < 1 || CurrentBars[1] < 1 || CurrentBars[2] < 1 || CurrentBars[3] < 1)
		            return;
		
		        if (State == State.Historical || State == State.Realtime)
		        {
		            if (IsFirstTickOfBar)
		            {
		                switch (BarsInProgress)
		                {
		                    case 1: Update15mData(); break;
		                    case 2: Update5mData(); break;
		                    case 3: Update1mData(); break;
		                }
		            }
		        }
		
		        if (State == State.Realtime && BarsInProgress == 0)
		        {
		            if (ShowIndicators)
		                UpdateChartIndicators();
		
		            if (isConnectedToFeatureServer)
		                SendMarketDataToPython();
		        }
		    }
		    catch (Exception ex)
		    {
		        Print($"OnBarUpdate error: {ex.Message}");
		    }
		}
		
		private void Update15mData()
		{
		    prices15m.Add(Closes[1][0]);
		    volumes15m.Add(Volumes[1][0]);
		    TrimList(prices15m, 415);
		    TrimList(volumes15m, 415);
		}
		
		private void Update5mData()
		{
		    prices5m.Add(Closes[2][0]);
		    volumes5m.Add(Volumes[2][0]);
		    TrimList(prices5m, 1248);
		    TrimList(volumes5m, 1248);
		}
		
		private void Update1mData()
		{
		    prices1m.Add(Closes[3][0]);
		    volumes1m.Add(Volumes[3][0]);
		    TrimList(prices1m, 6247);
		    TrimList(volumes1m, 6247);
		}
        
		protected override void OnExecutionUpdate(Execution execution, string executionId, 
		                                        double price, int quantity, MarketPosition marketPosition, 
		                                        string orderId, DateTime time)
		{
		    try
		    {
		        if (execution.Order != null)
		        {
		            tradesExecuted++;
		            
		            // Create trade ID for ML entry orders
		            if (execution.Order.Name.Contains("ML_") && 
		                string.IsNullOrEmpty(currentTradeId) &&
		                (execution.Order.OrderAction == OrderAction.Buy || execution.Order.OrderAction == OrderAction.SellShort))
		            {
		                lastTradeEntry = DateTime.Now;
		                tradeIdCounter++;
		                currentTradeId = $"trade_{tradeIdCounter}";
		                entryPrice = price;
		                hasPosition = true;
		                Print($"Trade started: {currentTradeId} at {price:F2}");
		            }
		            
		            // Handle position going flat
		            if (marketPosition == MarketPosition.Flat)
		            {
		                hasPosition = false;
		                entryPrice = 0;
		                
		                if (!string.IsNullOrEmpty(currentTradeId))
		                {
		                    int duration = (int)(DateTime.Now - lastTradeEntry).TotalMinutes;
		                    string exitReason = DetermineExitReason(execution.Order.Name);
		                    
		                    NotifyTradeCompletion(currentTradeId, price, exitReason, duration);
		                    Print($"Trade completed: {currentTradeId}, Exit: {price:F2}, Reason: {exitReason}");
		                    currentTradeId = "";
		                }
		            }
		        }
		    }
		    catch (Exception ex)
		    {
		        Print($"Execution update error: {ex.Message}");
		    }
		}
		
		private string DetermineExitReason(string orderName)
		{
		    if (orderName.Contains("Stop"))
		        return "stop_loss";
		    else if (orderName.Contains("Target") || orderName.Contains("Profit"))
		        return "take_profit";
		    else if (orderName.Contains("Exit") || orderName.Contains("Reverse"))
		        return "signal_exit";
		    else
		        return "market_close";
		}
        
        #endregion
        
        #region TCP Communication
        
        private void ConnectToPython()
		{
		    try
		    {
		        if (State == State.Realtime)
		        {
		            lastConnectionAttempt = DateTime.Now;
		            connectionAttempts++;
		        }
		        
		        if (!isConnectedToFeatureServer)
		        {
		            try
		            {
		                featureClient = new TcpClient("localhost", 5556);
		                isConnectedToFeatureServer = true;
		                Print("Connected to Python feature server (port 5556)");
		            }
		            catch (Exception ex)
		            {
		                Print($"Feature server connection failed: {ex.Message}");
		                isConnectedToFeatureServer = false;
		            }
		        }
		        
		        if (!isConnectedToSignalServer)
		        {
		            try
		            {
		                signalClient = new TcpClient("localhost", 5557);
		                isConnectedToSignalServer = true;
		                Print("Connected to Python signal server (port 5557)");
		            }
		            catch (Exception ex)
		            {
		                Print($"Signal server connection failed: {ex.Message}");
		                isConnectedToSignalServer = false;
		            }
		        }
		        
		        if (isConnectedToFeatureServer && isConnectedToSignalServer && State == State.Realtime)
		        {
		            // Connection message
		            Print("Research strategy fully connected to Python ML system");
		            Print("Using: RSI + Bollinger Bands + EMA + SMA + Volume (15m/5m/1m timeframes)");
		            Print("ML Model: Enhanced Ensemble with 1-minute entry timing");
		        }
		    }
		    catch (Exception ex)
		    {
		        Print($"Python connection error: {ex.Message}");
		        isConnectedToFeatureServer = false;
		        isConnectedToSignalServer = false;
		    }
		}
        
        private void StartSignalReceiver()
        {
            if (isRunning) return; // Already started
            
            isRunning = true;
            signalThread = new Thread(ReceiveSignals)
            {
                IsBackground = true,
                Name = "ResearchSignalReceiver"
            };
            signalThread.Start();
            Print("Signal receiver thread started");
        }
        
        private void ReceiveSignals()
        {
            while (isRunning)
            {
                try
                {
                    if (signalClient?.Connected != true)
                    {
                        Thread.Sleep(1000);
                        continue;
                    }
                    
                    // Read signal header (4 bytes for message length)
                    var headerBytes = new byte[4];
                    int headerRead = 0;
                    
                    while (headerRead < 4)
                    {
                        int bytesRead = signalClient.GetStream().Read(headerBytes, headerRead, 4 - headerRead);
                        if (bytesRead == 0)
                        {
                            Print("Signal connection lost");
                            isConnectedToSignalServer = false;
                            return;
                        }
                        headerRead += bytesRead;
                    }
                    
                    int messageLength = BitConverter.ToInt32(headerBytes, 0);
                    
                    if (messageLength <= 0 || messageLength > 10000)
                    {
                        Print($"Invalid message length: {messageLength}");
                        continue;
                    }
                    
                    // Read signal data
                    var messageBytes = new byte[messageLength];
                    int totalRead = 0;
                    
                    while (totalRead < messageLength)
                    {
                        int bytesRead = signalClient.GetStream().Read(
                            messageBytes, totalRead, messageLength - totalRead);
                        if (bytesRead == 0)
                        {
                            Print("Signal data connection lost");
                            isConnectedToSignalServer = false;
                            return;
                        }
                        totalRead += bytesRead;
                    }
                    
                    // Process the complete message
                    string signalJson = Encoding.UTF8.GetString(messageBytes);
                    ProcessSignal(signalJson);
                }
                catch (Exception ex)
                {
                    Print($"Signal receive error: {ex.Message}");
                    isConnectedToSignalServer = false;
                    Thread.Sleep(2000); // Wait before retry
                }
            }
            
            Print("Signal receiver thread stopped");
        }
        
        private void SendMarketDataToPython()
        {
            if (featureClient?.Connected != true)
            {
                isConnectedToFeatureServer = false;
                return;
            }
            
            try
            {
                var jsonBuilder = new StringBuilder();
                jsonBuilder.Append("{");
                
                // Add 15-minute price data
                jsonBuilder.Append("\"price_15m\":[");
                for (int i = 0; i < prices15m.Count; i++)
                {
                    if (i > 0) jsonBuilder.Append(",");
                    jsonBuilder.Append(prices15m[i].ToString("F6"));
                }
                jsonBuilder.Append("],");
                
                // Add 15-minute volume data
                jsonBuilder.Append("\"volume_15m\":[");
                for (int i = 0; i < volumes15m.Count; i++)
                {
                    if (i > 0) jsonBuilder.Append(",");
                    jsonBuilder.Append(volumes15m[i].ToString("F2"));
                }
                jsonBuilder.Append("],");
                
                // Add 5-minute price data
                jsonBuilder.Append("\"price_5m\":[");
                for (int i = 0; i < prices5m.Count; i++)
                {
                    if (i > 0) jsonBuilder.Append(",");
                    jsonBuilder.Append(prices5m[i].ToString("F6"));
                }
                jsonBuilder.Append("],");
                
                // Add 5-minute volume data
                jsonBuilder.Append("\"volume_5m\":[");
                for (int i = 0; i < volumes5m.Count; i++)
                {
                    if (i > 0) jsonBuilder.Append(",");
                    jsonBuilder.Append(volumes5m[i].ToString("F2"));
                }
                jsonBuilder.Append("],");
                
                // ENHANCED: Add 1-minute data
                jsonBuilder.Append("\"price_1m\":[");
                for (int i = 0; i < prices1m.Count; i++)
                {
                    if (i > 0) jsonBuilder.Append(",");
                    jsonBuilder.Append(prices1m[i].ToString("F6"));
                }
                jsonBuilder.Append("],");
                
                jsonBuilder.Append("\"volume_1m\":[");
                for (int i = 0; i < volumes1m.Count; i++)
                {
                    if (i > 0) jsonBuilder.Append(",");
                    jsonBuilder.Append(volumes1m[i].ToString("F2"));
                }
                jsonBuilder.Append("],");
                
                // Add timestamp
                jsonBuilder.Append($"\"timestamp\":{DateTime.Now.Ticks}");
                jsonBuilder.Append("}");
                
                string json = jsonBuilder.ToString();
                byte[] jsonBytes = Encoding.UTF8.GetBytes(json);
                byte[] header = BitConverter.GetBytes(jsonBytes.Length);
                
                var stream = featureClient.GetStream();
                stream.Write(header, 0, 4);
                stream.Write(jsonBytes, 0, jsonBytes.Length);
                
                if (CurrentBar % 50 == 0)
                {
                    // ENHANCED: Update debug message
                    Print($"Sent 3-timeframe data: 15m={prices15m.Count}, 5m={prices5m.Count}, 1m={prices1m.Count}");
                }
            }
            catch (Exception ex)
            {
                Print($"Data send error: {ex.Message}");
                isConnectedToFeatureServer = false;
            }
        }
        
        #endregion
        
        #region Data Management
		
		private void TrimList(List<double> list, int maxCount)
		{
		    if (list.Count > maxCount)
		        list.RemoveAt(0);
		}
        
        #endregion
        
        #region Signal Processing
        
		private void ProcessSignal(string signalJson)
		{
		    try
		    {
		        var signal = ParseSignalJson(signalJson);
		        if (signal == null) return;
				
				if (!IsValidSignal(signal)) return;
		        
		        signalCount++;
		        
		        Print($"ML Signal #{signalCount}: Action={GetActionName(signal.action)}, " +
		              $"Confidence={signal.confidence:F3}, Quality={signal.quality}");
		        
		        // Python already filtered - just execute
		        ExecuteMLSignal(signal);
		    }
		    catch (Exception ex)
		    {
		        Print($"Signal processing error: {ex.Message}");
		    }
		}
        
		private SignalData ParseSignalJson(string json)
		{
		    try 
		    {
		        var signal = new SignalData();
		        
		        // Extract action
		        var actionStart = json.IndexOf("\"action\":") + 9;
		        var actionEnd = json.IndexOf(",", actionStart);
		        signal.action = int.Parse(json.Substring(actionStart, actionEnd - actionStart));
		        
		        // Extract confidence
		        var confStart = json.IndexOf("\"confidence\":") + 13;
		        var confEnd = json.IndexOf(",", confStart);
		        signal.confidence = double.Parse(json.Substring(confStart, confEnd - confStart));
		        
		        // Parsing with proper error handling
		        var qualityPattern = "\"quality\":\"";
		        var qualStart = json.IndexOf(qualityPattern);
		        if (qualStart >= 0)
		        {
		            qualStart += qualityPattern.Length;
		            var qualEnd = json.IndexOf("\"", qualStart);
		            if (qualEnd > qualStart)
		            {
		                signal.quality = json.Substring(qualStart, qualEnd - qualStart);
		            }
		            else
		            {
		                signal.quality = "parsed_error";
		            }
		        }
		        else
		        {
		            // Try alternative parsing - sometimes quality might be at the end
		            var altPattern = "\"quality\": \"";
		            qualStart = json.IndexOf(altPattern);
		            if (qualStart >= 0)
		            {
		                qualStart += altPattern.Length;
		                var qualEnd = json.IndexOf("\"", qualStart);
		                if (qualEnd > qualStart)
		                {
		                    signal.quality = json.Substring(qualStart, qualEnd - qualStart);
		                }
		                else
		                {
		                    signal.quality = "alt_parse_error";
		                }
		            }
		            else
		            {
		                signal.quality = "not_found";
		            }
		        }
		        
		        // Extract timestamp
		        var timeStart = json.IndexOf("\"timestamp\":") + 12;
		        var timeEnd = json.IndexOf("}", timeStart);
		        if (timeEnd == -1) timeEnd = json.Length; // Handle end of string
		        signal.timestamp = long.Parse(json.Substring(timeStart, timeEnd - timeStart));
		        
		        return signal;
		    }
		    catch (Exception ex) 
		    {
		        Print($"JSON parsing error: {ex.Message} - JSON: {json}");
		        return null;
		    }
		}
		
		private bool IsValidSignal(SignalData signal)
		{
		    if (signal == null) return false;
		    
		    // 1. Signal age protection (execution-layer safety)
		    var signalAge = (DateTime.Now.Ticks - signal.timestamp) / TimeSpan.TicksPerSecond;
		    if (signalAge > 30)
		    {
		        Print($"Signal too old: {signalAge}s - skipping");
		        return false;
		    }
		    
		    // 2. Position-based spacing (C# position state)
		    if (Position.MarketPosition != MarketPosition.Flat && 
		        (DateTime.Now - lastConnectionAttempt).TotalSeconds < 10)
		    {
		        Print("Signal spacing filter - preventing over-trading");
		        return false;
		    }
		    
		    return true; // Python already handled confidence/quality filtering
		}
        
		private void ExecuteMLSignal(SignalData signal)
		{
		    try
		    {
		        int positionSize = CalculatePositionSize(signal.confidence);
		        
		        if (positionSize <= 0) return;
		        
		        switch (signal.action)
		        {
		            case 1: // Buy signal
		                if (Position.MarketPosition != MarketPosition.Long)
		                {
		                    if (Position.MarketPosition == MarketPosition.Short)
		                        ExitShort("ML_Reverse");
		                    
		                    EnterLong(positionSize, "ML_Long");
		                    Print($"LONG ENTRY: size={positionSize}, confidence={signal.confidence:F3}");
		                    VisualizeMLSignal(signal.action, signal.confidence, signal.quality);
		                }
		                else if (AllowScaleIn && signal.confidence > 0.75 && Position.Quantity < MaxPositionPieces)
		                {
	                        int scaleSize = Math.Max(1, positionSize / 2);
	                        EnterLong(scaleSize, "ML_Long_Scale");
	                        Print($"SCALE IN LONG: size={scaleSize}");
		                }
		                break;
		                
		            case 2: // Sell signal
		                if (Position.MarketPosition != MarketPosition.Short)
		                {
		                    if (Position.MarketPosition == MarketPosition.Long)
		                        ExitLong("ML_Reverse");
		                    
		                    EnterShort(positionSize, "ML_Short");
		                    Print($"SHORT ENTRY: size={positionSize}, confidence={signal.confidence:F3}");
		                    VisualizeMLSignal(signal.action, signal.confidence, signal.quality);
		                }
		                else if (AllowScaleIn && signal.confidence > 0.75 && Position.Quantity < MaxPositionPieces)
		                {
	                        int scaleSize = Math.Max(1, positionSize / 2);
	                        EnterShort(scaleSize, "ML_Short_Scale");
	                        Print($"SCALE IN SHORT: size={scaleSize}");
		                }
		                break;
		                
		            case 0: // Hold signal
		                if (Position.MarketPosition != MarketPosition.Flat)
		                {
		                    // Partial exit on low confidence
		                    if (signal.confidence < 0.4 && Position.Quantity > 1)
		                    {
		                        int partialSize = Position.Quantity / 2;
		                        if (Position.MarketPosition == MarketPosition.Long)
		                            ExitLong(partialSize, "ML_Partial_Exit", "");
		                        else
		                            ExitShort(partialSize, "ML_Partial_Exit", "");
		                        Print($"PARTIAL EXIT: {partialSize} contracts");
		                    }
		                    else
		                    {
		                        // Full exit
		                        if (Position.MarketPosition == MarketPosition.Long)
		                            ExitLong("ML_Exit");
		                        else
		                            ExitShort("ML_Exit");
		                    }
		                    VisualizeMLSignal(signal.action, signal.confidence, signal.quality);
		                }
		                break;
		        }
		    }
		    catch (Exception ex)
		    {
		        Print($"Enhanced signal execution error: {ex.Message}");
		    }
		}
		
		protected override void OnOrderUpdate(Order order, double limitPrice, double stopPrice, 
                                    int quantity, int filled, double averageFillPrice, 
                                    OrderState orderState, DateTime time, ErrorCode error, string comment)
		{
		    // Set exit orders when entry order is filled
		    if (order.Name.Contains("ML_") && orderState == OrderState.Filled)
		    {
		        SetExitOrders(order.Name, averageFillPrice);
		    }
		}
		
		private void SetExitOrders(string entrySignal, double fillPrice)
		{
		    try
		    {
		        entryPrice = fillPrice;
		        hasPosition = true;
		        
		        // Dynamic exit calculation
		        double atrValue = atr14[0];
		        double baseMultiplier = 2.5; // Base ATR multiplier
		        
		        // Calculate dynamic stops/targets
		        int dynamicStop = Math.Max(30, (int)(atrValue / TickSize * baseMultiplier));
		        int dynamicTarget = (int)(dynamicStop * 2.0); // 2:1 ratio
		        
		        // Cap maximum values
		        dynamicStop = Math.Min(dynamicStop, 120);
		        dynamicTarget = Math.Min(dynamicTarget, 240);
		        
		        Print($"Dynamic exits: Stop={dynamicStop} ticks, Target={dynamicTarget} ticks (ATR={atrValue:F2})");
		        
		        if (Position.MarketPosition == MarketPosition.Long)
		        {
		            SetStopLoss(entrySignal, CalculationMode.Ticks, dynamicStop, false);
		            SetProfitTarget(entrySignal, CalculationMode.Ticks, dynamicTarget);
		        }
		        else if (Position.MarketPosition == MarketPosition.Short)
		        {
		            SetStopLoss(entrySignal, CalculationMode.Ticks, dynamicStop, false);
		            SetProfitTarget(entrySignal, CalculationMode.Ticks, dynamicTarget);
		        }
		    }
		    catch (Exception ex)
		    {
		        Print($"Dynamic exit order setup error: {ex.Message}");
		    }
		}
		
		private string GetActionName(int action)
		{
		    switch (action)
		    {
		        case 1: return "BUY";
		        case 2: return "SELL";
		        default: return "HOLD";
		    }
		}
		
		private void NotifyTradeCompletion(string tradeId, double exitPrice, string exitReason, int durationMinutes)
		{
		    try
		    {
		        Print($"NotifyTradeCompletion called: {tradeId}");  // DEBUG LINE
		        
		        if (featureClient?.Connected != true) 
		        {
		            Print($"Cannot send trade completion - featureClient not connected");  // DEBUG LINE
		            return;
		        }
		        
		        Print($"Sending trade completion via feature client");  // DEBUG LINE
		        
		        // Create simple JSON manually
		        var json = $"{{\"type\":\"trade_completion\",\"signal_id\":\"{tradeId}\",\"exit_price\":{exitPrice},\"exit_reason\":\"{exitReason}\",\"duration_minutes\":{durationMinutes},\"timestamp\":{DateTime.Now.Ticks}}}";
		        
		        Print($"JSON to send: {json}");  // DEBUG LINE
		        
		        byte[] data = Encoding.UTF8.GetBytes(json);
		        byte[] header = BitConverter.GetBytes(data.Length);
		        
		        var stream = featureClient.GetStream();
		        stream.Write(header, 0, 4);
		        stream.Write(data, 0, data.Length);
		        
		        Print($"Trade completion SUCCESSFULLY sent: {tradeId}, Exit: {exitPrice:F2}, Duration: {durationMinutes}min");
		    }
		    catch (Exception ex)
		    {
		        Print($"Trade completion ERROR: {ex.Message}");
		    }
		}
        
        #endregion
        
        #region Position Management
        
		private int CalculatePositionSize(double confidence)
		{
		    // Research-aligned tiered position sizing
		    int baseSize = 1;
		    
		    if (confidence >= 0.9)
				baseSize = 4;
			else if (confidence >= 0.8)
		        baseSize = 3;      
		    else if (confidence >= 0.7)
		        baseSize = 2;      
		    else if (confidence >= 0.6)
		        baseSize = 1;      
		    else if (confidence >= 0.5)
		        baseSize = 1;  
		    
		    return Math.Min(baseSize, MaxPositionSize);
		}
        
        #endregion
        
        #region Cleanup
        
		private void CleanupWithStats()
		{
		    try
		    {
		        Print("Research Strategy shutting down...");
		        
		        isRunning = false;
		        
		        // Stop signal receiver thread
		        if (signalThread?.IsAlive == true)
		        {
		            if (!signalThread.Join(2000))
		            {
		                Print("Signal thread did not stop gracefully");
		            }
		        }
		        
		        // Close TCP connections
		        try
		        {
		            featureClient?.Close();
		            signalClient?.Close();
		            Print("TCP connections closed");
		        }
		        catch (Exception ex)
		        {
		            Print($"Connection cleanup error: {ex.Message}");
		        }
		        
		        // Log final statistics ONLY if we actually traded
		        Print($"Research Strategy Final Stats:");
		        Print($"ML Signals Received: {signalCount}");
		        Print($"Trades Executed: {tradesExecuted}");
		        Print($"Connection Attempts: {connectionAttempts}");
		        Print("Research Strategy stopped successfully");
		    }
		    catch (Exception ex)
		    {
		        Print($"Cleanup error: {ex.Message}");
		    }
		}
		
		private void QuietCleanup()
		{
		    try
		    {
		        // Silent cleanup for initialization failures
		        isRunning = false;
		        
		        if (signalThread?.IsAlive == true)
		        {
		            signalThread.Join(1000); // Shorter timeout
		        }
		        
		        featureClient?.Close();
		        signalClient?.Close();
		        
		        // No print statements for startup failures
		    }
		    catch
		    {
		        // Silent failure
		    }
		}
        
        #endregion
        
        #region Helper Classes
        
        public class SignalData
        {
            public int action { get; set; }
            public double confidence { get; set; }
            public string quality { get; set; }
            public long timestamp { get; set; }
        }
        
        #endregion
		
		#region Chart Updates
		
		private void UpdateChartIndicators()
		{
		    if (!ShowIndicators || State != State.Realtime)
		        return;
		        
		    try
		    {
		        // Force indicator updates if needed
		        if (ema20 != null) ema20.Update();
		        if (sma50 != null) sma50.Update();
		        if (bb != null) bb.Update();
		        if (rsi14 != null) rsi14.Update();
		        
		        // Calculate volume ratio for display (we'll show this as text)
		        if (volumeIndicator != null && CurrentBar > 20)
		        {
		            double currentVol = Volume[0];
		            double avgVol = 0;
		            for (int i = 0; i < 20; i++)
		            {
		                avgVol += Volume[i];
		            }
		            avgVol /= 20;
		            
		            double volRatio = avgVol > 0 ? currentVol / avgVol : 1.0;
		            
		            // Display volume ratio as text on chart every 10 bars to avoid clutter
		            if (CurrentBar % 10 == 0)
		            {
		                Draw.TextFixed(this, $"VolRatio_{CurrentBar}", 
		                    $"Vol Ratio: {volRatio:F2}", 
		                    TextPosition.TopRight, 
		                    Brushes.White, 
		                    new SimpleFont("Arial", 10), 
		                    Brushes.Transparent, 
		                    Brushes.Transparent, 
		                    0);
		            }
		        }
		    }
		    catch (Exception ex)
		    {
		        Print($"Chart indicator update error: {ex.Message}");
		    }
		}
		
		private void VisualizeMLSignal(int action, double confidence, string quality)
		{
		    if (!ShowIndicators) return;
		    
		    try
		    {
		        string signalText = "";
		        Brush signalColor = Brushes.Gray;
		        
		        switch (action)
		        {
		            case 1: // Buy
		                signalText = $"BUY ({confidence:F2})";
		                signalColor = Brushes.Green;
		                Draw.ArrowUp(this, $"Buy_{CurrentBar}", false, 0, Low[0] - 2 * TickSize, signalColor);
		                break;
		            case 2: // Sell  
		                signalText = $"SELL ({confidence:F2})";
		                signalColor = Brushes.Red;
		                Draw.ArrowDown(this, $"Sell_{CurrentBar}", false, 0, High[0] + 2 * TickSize, signalColor);
		                break;
		            case 0: // Hold/Exit
		                if (Position.MarketPosition != MarketPosition.Flat)
		                {
		                    signalText = $"EXIT ({confidence:F2})";
		                    signalColor = Brushes.Orange;
		                    Draw.Diamond(this, $"Exit_{CurrentBar}", false, 0, Close[0], signalColor);
		                }
		                break;
		        }
		        
		        // Add text label for signal
		        if (!string.IsNullOrEmpty(signalText))
		        {
		            Draw.Text(this, $"Signal_{CurrentBar}", signalText, 0, 
		                action == 2 ? High[0] + 4 * TickSize : Low[0] - 4 * TickSize, 
		                signalColor);
		        }
		    }
		    catch (Exception ex)
		    {
		        Print($"Signal visualization error: {ex.Message}");
		    }
		}
		
		#endregion
    }
}