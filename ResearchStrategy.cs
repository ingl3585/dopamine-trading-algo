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
		                        Print($"Sending historical data for training: 15m={prices15m.Count}, 5m={prices5m.Count}");
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
		    StopLossTicks = 75;        // Simple fixed stop
		    TakeProfitTicks = 200;      // 4:1 reward-to-risk (research optimal)
		    MinConfidence = 0.5;       // Research: 60% accuracy threshold
		    MaxPositionSize = 10;       // Simple position limits
		    
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
			
			if (ShowIndicators)
			{
			    // Initialize indicators
			    ema20 = EMA(BarsArray[0], 20);
			    sma50 = SMA(BarsArray[0], 50);
			    bb = Bollinger(BarsArray[0], 2.0, 20);
			    rsi14 = RSI(BarsArray[0], 14, 3);
			    volumeIndicator = VOL(BarsArray[0]);
			    
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
			    
			    // Add indicators to chart
			    AddChartIndicator(ema20);
			    AddChartIndicator(sma50);
			    AddChartIndicator(bb);
			    
			    // RSI goes in separate panel
			    AddChartIndicator(rsi14);
			}
        }
        
		protected override void OnBarUpdate()
		{
		    try
		    {
		        // Validate we have enough data on all timeframes before processing
		        if (CurrentBars[0] < 1 || 
		            (BarsArray.Length > 1 && CurrentBars[1] < 1) || 
		            (BarsArray.Length > 2 && CurrentBars[2] < 1))
		        {
		            return; // Wait until all timeframes have data
		        }
		        
		        if (State == State.Historical)
		        {
		            // Collect data from all timeframes during historical processing
		            UpdateMarketData();
		        }
		        else if (State == State.Realtime)
		        {
		            // Only send to Python on primary timeframe updates in real-time
		            // This ensures all secondary timeframes have updated their latest values
		            if (BarsInProgress == 0)
		            {
		                UpdateMarketData();
						
						if (ShowIndicators)
		                {
		                    UpdateChartIndicators();
		                }
		                
		                if (isConnectedToFeatureServer)
		                {
		                    SendMarketDataToPython();
		                }
		            }
		        }
		    }
		    catch (Exception ex)
		    {
		        Print($"OnBarUpdate error: {ex.Message}");
		    }
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
		            
		            Print($"=== EXECUTION DEBUG ===");
		            Print($"Order Name: '{execution.Order.Name}'");
		            Print($"Order Type: {execution.Order.OrderType}");
		            Print($"Order Action: {execution.Order.OrderAction}");
		            Print($"Quantity: {quantity}");
		            Print($"Price: {price:F2}");
		            Print($"Market Position BEFORE: {Position.MarketPosition}");
		            Print($"Market Position AFTER: {marketPosition}");
		            Print($"Position Quantity: {Position.Quantity}");
		            Print($"hasPosition flag: {hasPosition}");
		            Print($"Current Trade ID: '{currentTradeId}'");
		            Print($"=======================");
		            
		            // MOVED: Create trade ID for ML entry orders (outside position logic)
		            if (execution.Order.Name.Contains("ML_") && 
		                string.IsNullOrEmpty(currentTradeId) &&
		                (execution.Order.OrderAction == OrderAction.Buy || execution.Order.OrderAction == OrderAction.SellShort))
		            {
		                lastTradeEntry = DateTime.Now;
		                tradeIdCounter++;
		                currentTradeId = $"trade_{tradeIdCounter}";
		                Print($"Trade started: {currentTradeId} at {price:F2}");
		            }
		            
					if (marketPosition == MarketPosition.Flat || 
					    (execution.Order.Name.Contains("Profit target") && Position.Quantity == 0) ||
					    (execution.Order.Name.Contains("Stop loss") && Position.Quantity == 0))
		            {
		                hasPosition = false;
		                entryPrice = 0;
		                
		                Print($"Position now FLAT - checking for trade completion notification");
		                Print($"Current trade ID: '{currentTradeId}'");
		                
		                // Only send completion notification when going flat
		                if (!string.IsNullOrEmpty(currentTradeId))
		                {
		                    int duration = (int)(DateTime.Now - lastTradeEntry).TotalMinutes;
		                    string exitReason = "unknown";
		                    
		                    Print($"Order name for exit reason: '{execution.Order.Name}'");
		                    
		                    // Determine exit reason from order name
		                    if (execution.Order.Name.Contains("Stop"))
		                        exitReason = "stop_loss";
		                    else if (execution.Order.Name.Contains("Target") || execution.Order.Name.Contains("Profit"))
		                        exitReason = "take_profit";
		                    else if (execution.Order.Name.Contains("Close") || execution.Order.Name.Contains("Reverse"))
		                        exitReason = "signal_exit";
		                    else
		                        exitReason = "market_close";
		                    
		                    Print($"Determined exit reason: {exitReason}");
		                    
		                    NotifyTradeCompletion(currentTradeId, price, exitReason, duration);
		                    
		                    Print($"Trade completion sent: {currentTradeId}, Exit: {price:F2}, Reason: {exitReason}, Duration: {duration}min");
		                    currentTradeId = "";  // Clear the trade ID
		                }
		                else
		                {
		                    Print("WARNING: No currentTradeId found for completion notification");
		                }
		                
		                Print("Position closed - ready for new signals");
		            }
		            else if (!hasPosition)  // Only when opening NEW position
		            {
		                hasPosition = true;
		                entryPrice = price;
		                Print($"NEW position opened at {price:F2}");
		            }
		            else
		            {
		                // Position is being added to (additional contracts)
		                Print($"Adding to existing position - no new trade ID");
		            }
		        }
		        else
		        {
		            Print("WARNING: execution.Order is null");
		        }
		    }
		    catch (Exception ex)
		    {
		        Print($"Execution update error: {ex.Message}");
		    }
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
		        
		        // Connect to Python feature server
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
		        
		        // Connect to Python signal server
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
		        
		        // Only print success message once
		        if (isConnectedToFeatureServer && isConnectedToSignalServer && State == State.Realtime)
		        {
		            Print("Research strategy fully connected to Python ML system");
		            Print("Using: RSI + Bollinger Bands + EMA + SMA + Volume (15m/5m timeframes)");
		            Print("ML Model: Logistic Regression");
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
                // Create simple JSON manually (avoiding System.Text.Json dependency)
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
                
                // Add timestamp
                jsonBuilder.Append($"\"timestamp\":{DateTime.Now.Ticks}");
                jsonBuilder.Append("}");
                
                string json = jsonBuilder.ToString();
                byte[] jsonBytes = Encoding.UTF8.GetBytes(json);
                byte[] header = BitConverter.GetBytes(jsonBytes.Length);
                
                var stream = featureClient.GetStream();
                stream.Write(header, 0, 4);
                stream.Write(jsonBytes, 0, jsonBytes.Length);
                
                // Debug output every 50 bars
                if (CurrentBar % 50 == 0)
                {
                    Print($"Sent market data: 15m bars={prices15m.Count}, 5m bars={prices5m.Count}");
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
        
		private void UpdateMarketData()
		{
		    try
		    {
		        // Update 15-minute data
		        if (BarsArray.Length > 1 && CurrentBars[1] >= 0)
		        {
		            double price15m = Closes[1][0];
		            double volume15m = Volumes[1][0];
		            
		            prices15m.Add(price15m);
		            volumes15m.Add(volume15m);
		            
		            if (prices15m.Count > 1000)
		            {
		                prices15m.RemoveAt(0);
		                volumes15m.RemoveAt(0);
		            }
		        }
		        
		        // Update 5-minute data
		        if (BarsArray.Length > 2 && CurrentBars[2] >= 0)
		        {
		            double price5m = Closes[2][0];
		            double volume5m = Volumes[2][0];
		            
		            prices5m.Add(price5m);
		            volumes5m.Add(volume5m);
		            
		            if (prices5m.Count > 400)
		            {
		                prices5m.RemoveAt(0);
		                volumes5m.RemoveAt(0);
		            }
		        }
		        
		        // MUCH less frequent debug output - only every 100 bars
		        if (CurrentBar % 100 == 0 && State == State.Realtime)
		        {
		            Print($"Data: 15m={prices15m.Count} bars, 5m={prices5m.Count} bars");
		        }
		    }
		    catch (Exception ex)
		    {
		        Print($"Data update error: {ex.Message}");
		    }
		}
        
        #endregion
        
        #region Signal Processing
        
        private void ProcessSignal(string signalJson)
        {
            try
            {
                // Parse JSON manually (simple parsing for our known structure)
                var signal = ParseSignalJson(signalJson);
                if (signal == null) return;
                
                signalCount++;
                
                Print($"ML Signal #{signalCount}: Action={GetActionName(signal.action)}, " +
                      $"Confidence={signal.confidence:F3}, Quality={signal.quality}");
                
                // Validate signal
                if (!IsValidSignal(signal)) return;
                
                // Execute trade based on signal
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
		    // Research principle: Quality over quantity
		    // Your research.txt: "High-accuracy signals consistently outperform high-frequency, lower-quality signals"
		    
		    // Check confidence threshold
		    if (signal.confidence < MinConfidence)
		    {
		        Print($"Signal confidence {signal.confidence:F3} below threshold {MinConfidence:F3}");
		        return false;
		    }
		    
		    // Research-aligned quality filter
		    if (signal.quality == "poor")
		    {
		        Print("Signal quality too poor - skipping (research-aligned filtering)");
		        return false;
		    }
		    
		    // Prevent over-trading (research principle)
		    if (Position.MarketPosition != MarketPosition.Flat && 
		        (DateTime.Now - lastConnectionAttempt).TotalSeconds < 10) // Increased from 5 to 10
		    {
		        Print("Signal spacing filter - preventing over-trading");
		        return false;
		    }
		    
		    // Add signal age validation (research: signals have half-life)
		    var signalAge = (DateTime.Now.Ticks - signal.timestamp) / TimeSpan.TicksPerSecond;
		    if (signalAge > 30) // Signals older than 30 seconds are stale
		    {
		        Print($"Signal too old: {signalAge}s - skipping");
		        return false;
		    }
		    
		    return true;
		}
        
        private void ExecuteMLSignal(SignalData signal)
        {
            try
            {
                // Calculate position size based on research guidelines
                int positionSize = CalculatePositionSize(signal.confidence);
                
                if (positionSize <= 0)
                {
                    Print($"Position size calculation resulted in 0 - skipping signal");
                    return;
                }
                
				switch (signal.action)
				{
				    case 1: // Buy signal
				        if (Position.MarketPosition != MarketPosition.Long)
				        {
				            if (Position.MarketPosition == MarketPosition.Short)
				            {
				                ExitShort("ML_Reverse");
				                Print("Reversing from SHORT to LONG");
				            }
				            
				            EnterLong(positionSize, "ML_Long");
				            Print($"LONG ENTRY: size={positionSize}, confidence={signal.confidence:F3}, quality={signal.quality}");

				            VisualizeMLSignal(signal.action, signal.confidence, signal.quality);
				        }
				        else
				        {
				            Print("Already LONG - ignoring buy signal");
				        }
				        break;
				        
				    case 2: // Sell signal
				        if (Position.MarketPosition != MarketPosition.Short)
				        {
				            if (Position.MarketPosition == MarketPosition.Long)
				            {
				                ExitLong("ML_Reverse");
				                Print("Reversing from LONG to SHORT");
				            }
				            
				            EnterShort(positionSize, "ML_Short");
				            Print($"SHORT ENTRY: size={positionSize}, confidence={signal.confidence:F3}, quality={signal.quality}");
				            
				            // ADD THIS LINE HERE:
				            VisualizeMLSignal(signal.action, signal.confidence, signal.quality);
				        }
				        else
				        {
				            Print("Already SHORT - ignoring sell signal");
				        }
				        break;
				        
				    case 0: // Hold signal
				        if (Position.MarketPosition != MarketPosition.Flat)
				        {
				            if (Position.MarketPosition == MarketPosition.Long)
				                ExitLong("ML_Exit");
				            else
				                ExitShort("ML_Exit");
				            
				            Print($"HOLD SIGNAL: confidence={signal.confidence:F3}, quality={signal.quality}");
				            
				            VisualizeMLSignal(signal.action, signal.confidence, signal.quality);
				        }
				        else
				        {
				            Print("Already FLAT - ignoring exit signal");
				        }
				        break;
				}
            }
            catch (Exception ex)
            {
                Print($"Signal execution error: {ex.Message}");
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
				baseSize = 8;
			else if (confidence >= 0.8)
		        baseSize = 6;      
		    else if (confidence >= 0.7)
		        baseSize = 4;      
		    else if (confidence >= 0.6)
		        baseSize = 2;      
		    else if (confidence >= 0.5)
		        baseSize = 1;  
		    
		    return Math.Min(baseSize, MaxPositionSize);
		}
        
        protected override void OnOrderUpdate(Order order, double limitPrice, double stopPrice, 
                                            int quantity, int filled, double averageFillPrice, 
                                            OrderState orderState, DateTime time, ErrorCode error, string comment)
        {
            // Set exit orders when entry order is filled
            if (order.Name.Contains("ML_") && orderState == OrderState.Filled)
            {
                SetExitOrders(order.Name);
            }
        }
        
		private void SetExitOrders(string entrySignal)
		{
		    try
		    {
		        entryPrice = Close[0];
		        hasPosition = true;
		        
		        Print($"Setting exit orders for {entrySignal}");
		        Print($"Current position: {Position.MarketPosition}, Quantity: {Position.Quantity}");
		        
		        // FIXED: Use CalculationMode.Ticks with proper stop loss setup
		        // The key is to use SetStopLoss correctly - it should CLOSE the position, not create new one
		        
		        if (Position.MarketPosition == MarketPosition.Long)
		        {
		            // For LONG positions, stop loss should be BELOW entry price
		            SetStopLoss(entrySignal, CalculationMode.Ticks, StopLossTicks, false);
		            SetProfitTarget(entrySignal, CalculationMode.Ticks, TakeProfitTicks);
		            
		            Print($"LONG exit orders set - Entry: {entryPrice:F2}");
		            Print($"Stop Loss: {StopLossTicks} ticks below entry = {entryPrice - (StopLossTicks * TickSize):F2}");
		            Print($"Take Profit: {TakeProfitTicks} ticks above entry = {entryPrice + (TakeProfitTicks * TickSize):F2}");
		        }
		        else if (Position.MarketPosition == MarketPosition.Short)
		        {
		            // For SHORT positions, stop loss should be ABOVE entry price
		            SetStopLoss(entrySignal, CalculationMode.Ticks, StopLossTicks, false);
		            SetProfitTarget(entrySignal, CalculationMode.Ticks, TakeProfitTicks);
		            
		            Print($"SHORT exit orders set - Entry: {entryPrice:F2}");
		            Print($"Stop Loss: {StopLossTicks} ticks above entry = {entryPrice + (StopLossTicks * TickSize):F2}");
		            Print($"Take Profit: {TakeProfitTicks} ticks below entry = {entryPrice - (TakeProfitTicks * TickSize):F2}");
		        }
		        
		        Print($"Exit orders configured for position size: {Position.Quantity}");
		    }
		    catch (Exception ex)
		    {
		        Print($"Exit order setup error: {ex.Message}");
		    }
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