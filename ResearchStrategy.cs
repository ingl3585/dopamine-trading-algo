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
        
        // Connection status
        private bool isConnectedToFeatureServer;
        private bool isConnectedToSignalServer;
        
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
                        Print("Research Strategy data loaded - preparing for connection");
                        break;
                        
					case State.Realtime:
					    Print("Research Strategy entering real-time mode");
					    ConnectToPython();
					    StartSignalReceiver();
					    
					    // Send initial historical data for training
					    if (isConnectedToFeatureServer)
					    {
					        Print($"Sending historical data for training: 15m={prices15m.Count}, 5m={prices5m.Count}");
					        SendMarketDataToPython();
					    }
					    break;
                        
                    case State.Terminated:
                        Cleanup();
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
            Description = "Research-aligned strategy using Python ML signals with RSI, Bollinger Bands, EMA, SMA";
            Name = "ResearchStrategy";
            Calculate = Calculate.OnBarClose;
            
            // Set default values
            RiskPercent = 0.02;
            StopLossTicks = 20;
            TakeProfitTicks = 40;
            MinConfidence = 0.6;
            MaxPositionSize = 2;
            
            // Strategy settings
            EntriesPerDirection = 1;
            EntryHandling = EntryHandling.AllEntries;
            ExitOnSessionCloseSeconds = 30;
            IsFillLimitOnTouch = false;
            MaximumBarsLookBack = MaximumBarsLookBack.TwoHundredFiftySix;
            OrderFillResolution = OrderFillResolution.Standard;
            Slippage = 0;
            StartBehavior = StartBehavior.WaitUntilFlat;
            TimeInForce = TimeInForce.Gtc;
            TraceOrders = false;
            RealtimeErrorHandling = RealtimeErrorHandling.StopCancelClose;
            StopTargetHandling = StopTargetHandling.PerEntryExecution;
            BarsRequiredToTrade = 0;
        }
        
        private void ConfigureStrategy()
        {
            // Add multi-timeframe data series
            AddDataSeries(BarsPeriodType.Minute, 15);  // BarsArray[1] - 15-minute
            AddDataSeries(BarsPeriodType.Minute, 5);   // BarsArray[2] - 5-minute
            
            Print("Research Strategy configured with 15m and 5m timeframes");
        }
        
		protected override void OnBarUpdate()
		{
		    try
		    {
		        // Only process on primary timeframe updates  
		        if (BarsInProgress != 0) return;
		        
		        if (State == State.Historical)
		        {
		            // Just collect historical data, don't send to Python
		            UpdateMarketData();
		        }
		        else if (State == State.Realtime)
		        {
		            // Send data to Python for real-time trading only
		            UpdateMarketData();
		            
		            if (isConnectedToFeatureServer)
		            {
		                SendMarketDataToPython();
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
                    Print($"Trade executed: {execution.Order.Name} - {quantity} @ {price:F2}");
                    
                    if (marketPosition == MarketPosition.Flat)
                    {
                        hasPosition = false;
                        entryPrice = 0;
                        Print("Position closed - ready for new signals");
                    }
                    else
                    {
                        hasPosition = true;
                        entryPrice = price;
                    }
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
                lastConnectionAttempt = DateTime.Now;
                connectionAttempts++;
                
                Print($"Attempting to connect to Python ML system (attempt {connectionAttempts})...");
                
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
                
                if (isConnectedToFeatureServer && isConnectedToSignalServer)
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
                // Update 15-minute data (BarsArray[1])
                if (BarsArray.Length > 1 && CurrentBars.Length > 1 && CurrentBars[1] >= 0)
                {
                    double price15m = Closes[1][0];
                    double volume15m = Volumes[1][0];
                    
                    prices15m.Add(price15m);
                    volumes15m.Add(volume15m);
                    
                    // Keep last 100 bars for performance
                    if (prices15m.Count > 100)
                    {
                        prices15m.RemoveAt(0);
                        volumes15m.RemoveAt(0);
                    }
                }
                
                // Update 5-minute data (BarsArray[2])
                if (BarsArray.Length > 2 && CurrentBars.Length > 2 && CurrentBars[2] >= 0)
                {
                    double price5m = Closes[2][0];
                    double volume5m = Volumes[2][0];
                    
                    prices5m.Add(price5m);
                    volumes5m.Add(volume5m);
                    
                    // Keep last 100 bars for performance
                    if (prices5m.Count > 100)
                    {
                        prices5m.RemoveAt(0);
                        volumes5m.RemoveAt(0);
                    }
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
		    try {
		        var signal = new SignalData();
		        
		        // Extract action (unchanged)
		        var actionStart = json.IndexOf("\"action\":") + 9;
		        var actionEnd = json.IndexOf(",", actionStart);
		        signal.action = int.Parse(json.Substring(actionStart, actionEnd - actionStart));
		        
		        // Extract confidence (unchanged)
		        var confStart = json.IndexOf("\"confidence\":") + 13;
		        var confEnd = json.IndexOf(",", confStart);
		        signal.confidence = double.Parse(json.Substring(confStart, confEnd - confStart));
		        
		        // FIX: Better quality parsing
		        var qualStart = json.IndexOf("\"quality\":\"") + 11;
		        if (qualStart > 10) // Found quoted string
		        {
		            var qualEnd = json.IndexOf("\"", qualStart);
		            signal.quality = json.Substring(qualStart, qualEnd - qualStart);
		        }
		        else
		        {
		            // Fallback: treat as string anyway
		            signal.quality = "unknown";
		        }
		        
		        // Extract timestamp (unchanged)
		        var timeStart = json.IndexOf("\"timestamp\":") + 12;
		        var timeEnd = json.IndexOf("}", timeStart);
		        signal.timestamp = long.Parse(json.Substring(timeStart, timeEnd - timeStart));
		        
		        return signal;
		    }
		    catch (Exception ex) {
		        Print($"JSON parsing error: {ex.Message} - JSON: {json}");
		        return null;
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
        
        private bool IsValidSignal(SignalData signal)
        {
            // Check confidence threshold
            if (signal.confidence < MinConfidence)
            {
                Print($"Signal confidence {signal.confidence:F3} below threshold {MinConfidence:F3}");
                return false;
            }
            
            // Check signal quality
            if (signal.quality == "poor")
            {
                Print("Signal quality too poor - skipping");
                return false;
            }
            
            // Check if we're in a valid trading state
            if (Position.MarketPosition != MarketPosition.Flat && 
                (DateTime.Now - lastConnectionAttempt).TotalSeconds < 5)
            {
                Print("Too close to connection attempt - waiting");
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
                        }
                        else
                        {
                            Print("Already SHORT - ignoring sell signal");
                        }
                        break;
                        
                    case 0: // Hold/Exit signal
                        if (Position.MarketPosition != MarketPosition.Flat)
                        {
                            if (Position.MarketPosition == MarketPosition.Long)
                                ExitLong("ML_Exit");
                            else
                                ExitShort("ML_Exit");
                            
                            Print($"EXIT SIGNAL: confidence={signal.confidence:F3}, quality={signal.quality}");
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
        
        #endregion
        
        #region Position Management
        
        private int CalculatePositionSize(double confidence)
        {
            try
            {
                // Simple position sizing based on research principles
                // Fixed percentage risk with confidence adjustment
                
                int baseSize = 1; // Base position size
                
                // Adjust for confidence (research shows simple methods work best)
                if (confidence >= 0.8)
                    baseSize = 2; // High confidence
                else if (confidence >= 0.7)
                    baseSize = 2; // Good confidence (simplified to 2)
                else
                    baseSize = 1; // Standard confidence
                
                // Cap at maximum position size
                return Math.Min(baseSize, MaxPositionSize);
            }
            catch (Exception ex)
            {
                Print($"Position size calculation error: {ex.Message}");
                return 1; // Default to 1 contract
            }
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
                
                // Set stop loss (research shows simple fixed stops work well)
                SetStopLoss(entrySignal, CalculationMode.Ticks, StopLossTicks, false);
                
                // Set take profit (2:1 reward-to-risk ratio per research)
                SetProfitTarget(entrySignal, CalculationMode.Ticks, TakeProfitTicks);
                
                Print($"Exit orders set - Entry: {entryPrice:F2}, " +
                      $"Stop: {StopLossTicks} ticks, Target: {TakeProfitTicks} ticks (2:1 R:R)");
            }
            catch (Exception ex)
            {
                Print($"Exit order setup error: {ex.Message}");
            }
        }
        
        #endregion
        
        #region Cleanup
        
        private void Cleanup()
        {
            try
            {
                Print("Research Strategy shutting down...");
                
                isRunning = false;
                
                // Stop signal receiver thread
                if (signalThread?.IsAlive == true)
                {
                    if (!signalThread.Join(2000)) // Wait up to 2 seconds
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
                
                // Log final statistics
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
    }
}