// ResearchStrategy.cs

using System;
using System.Collections.Generic;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using NinjaTrader.Cbi;
using NinjaTrader.Data;
using NinjaTrader.NinjaScript;
using NinjaTrader.NinjaScript.Strategies;

namespace NinjaTrader.NinjaScript.Strategies
{
    public class ResearchStrategy : Strategy
    {
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
        
        // Minimal position tracking for learning feedback
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
                            Print("Black Box Intelligence Strategy data loaded");
                        }
                        break;
                        
                    case State.Realtime:
                        if (!isConnectedToFeatureServer || !isConnectedToSignalServer)
                        {
                            Print("Black Box Intelligence entering real-time mode");
                            ConnectToPython();
                            StartSignalReceiver();
                            
                            if (isConnectedToFeatureServer)
                            {
                                Print($"Sending historical data: 15m={prices15m.Count}, 5m={prices5m.Count}, 1m={prices1m.Count}");
                                SendMarketDataToPython();
                            }
                        }
                        break;
                        
                    case State.Terminated:
                        if (signalCount > 0 || tradesExecuted > 0)
                        {
                            CleanupWithStats();
                        }
                        else
                        {
                            QuietCleanup();
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
            Description = "Black Box Intelligence - Pure AI Pattern Discovery";
            Name = "ResearchStrategy";
            Calculate = Calculate.OnBarClose;
            
            // Minimal NinjaTrader settings
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
            AddDataSeries(BarsPeriodType.Minute, 1);   // BarsArray[3] - 1-minute
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
                        Print($"Trade started: {currentTradeId} at {price:F2}");
                    }
                    
                    // Handle position going flat - notify intelligence for learning
                    if (marketPosition == MarketPosition.Flat)
                    {
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
            if (orderName.Contains("Exit"))
                return "intelligence_exit";
            else if (orderName.Contains("Reverse"))
                return "signal_reversal";
            else
                return "session_close";
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
                        Print("Connected to Intelligence Engine (port 5556)");
                    }
                    catch (Exception ex)
                    {
                        Print($"Intelligence connection failed: {ex.Message}");
                        isConnectedToFeatureServer = false;
                    }
                }
                
                if (!isConnectedToSignalServer)
                {
                    try
                    {
                        signalClient = new TcpClient("localhost", 5557);
                        isConnectedToSignalServer = true;
                        Print("Connected to Signal Server (port 5557)");
                    }
                    catch (Exception ex)
                    {
                        Print($"Signal connection failed: {ex.Message}");
                        isConnectedToSignalServer = false;
                    }
                }
                
                if (isConnectedToFeatureServer && isConnectedToSignalServer && State == State.Realtime)
                {
                    Print("Black Box Intelligence fully connected");
                    Print("Pure AI Pattern Discovery - No hardcoded rules");
                }
            }
            catch (Exception ex)
            {
                Print($"Connection error: {ex.Message}");
                isConnectedToFeatureServer = false;
                isConnectedToSignalServer = false;
            }
        }
        
        private void StartSignalReceiver()
        {
            if (isRunning) return;
            
            isRunning = true;
            signalThread = new Thread(ReceiveSignals)
            {
                IsBackground = true,
                Name = "IntelligenceSignalReceiver"
            };
            signalThread.Start();
            Print("Intelligence signal receiver started");
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
                    Thread.Sleep(2000);
                }
            }
            
            Print("Intelligence signal receiver stopped");
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
                
                // Add 1-minute data
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
                    Print($"Market data fed to intelligence: 15m={prices15m.Count}, 5m={prices5m.Count}, 1m={prices1m.Count}");
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
                
                Print($"Intelligence Signal #{signalCount}: Action={GetActionName(signal.action)}, " +
                      $"Confidence={signal.confidence:F3}, Quality={signal.quality}");
                
                // Pure execution - no overrides, no hardcoded rules
                ExecuteIntelligenceSignal(signal);
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
                
                // Extract quality
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
                        signal.quality = "unknown";
                    }
                }
                else
                {
                    signal.quality = "unknown";
                }
                
                // Extract timestamp
                var timeStart = json.IndexOf("\"timestamp\":") + 12;
                var timeEnd = json.IndexOf("}", timeStart);
                if (timeEnd == -1) timeEnd = json.Length;
                signal.timestamp = long.Parse(json.Substring(timeStart, timeEnd - timeStart));
                
                return signal;
            }
            catch (Exception ex) 
            {
                Print($"JSON parsing error: {ex.Message}");
                return null;
            }
        }
        
        private bool IsValidSignal(SignalData signal)
        {
            if (signal == null) return false;
            
            // Basic signal age protection only
            var signalAge = (DateTime.Now.Ticks - signal.timestamp) / TimeSpan.TicksPerSecond;
            if (signalAge > 30)
            {
                Print($"Signal too old: {signalAge}s - skipping");
                return false;
            }
            
            return true;
        }
        
        private void ExecuteIntelligenceSignal(SignalData signal)
        {
            try
            {
                switch (signal.action)
                {
                    case 1: // Buy signal - pure intelligence decision
                        if (Position.MarketPosition != MarketPosition.Long)
                        {
                            if (Position.MarketPosition == MarketPosition.Short)
                                ExitShort("ML_Exit");
                            
                            EnterLong(1, "ML_Long"); // Fixed size - let intelligence learn optimal sizing
                            Print($"INTELLIGENCE LONG: confidence={signal.confidence:F3}");
                        }
                        break;
                        
                    case 2: // Sell signal - pure intelligence decision
                        if (Position.MarketPosition != MarketPosition.Short)
                        {
                            if (Position.MarketPosition == MarketPosition.Long)
                                ExitLong("ML_Exit");
                            
                            EnterShort(1, "ML_Short"); // Fixed size - let intelligence learn optimal sizing
                            Print($"INTELLIGENCE SHORT: confidence={signal.confidence:F3}");
                        }
                        break;
                        
                    case 0: // Exit signal - intelligence says get out
                        if (Position.MarketPosition != MarketPosition.Flat)
                        {
                            if (Position.MarketPosition == MarketPosition.Long)
                                ExitLong("ML_Exit");
                            else if (Position.MarketPosition == MarketPosition.Short)
                                ExitShort("ML_Exit");
                            
                            Print($"INTELLIGENCE EXIT: confidence={signal.confidence:F3}");
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
                default: return "EXIT";
            }
        }
        
        private void NotifyTradeCompletion(string tradeId, double exitPrice, string exitReason, int durationMinutes)
        {
            try
            {
                if (featureClient?.Connected != true) 
                {
                    return;
                }
                
                // Send trade outcome to intelligence for learning
                var json = $"{{\"type\":\"trade_completion\",\"signal_id\":\"{tradeId}\",\"exit_price\":{exitPrice},\"exit_reason\":\"{exitReason}\",\"duration_minutes\":{durationMinutes},\"timestamp\":{DateTime.Now.Ticks}}}";
                
                byte[] data = Encoding.UTF8.GetBytes(json);
                byte[] header = BitConverter.GetBytes(data.Length);
                
                var stream = featureClient.GetStream();
                stream.Write(header, 0, 4);
                stream.Write(data, 0, data.Length);
                
                Print($"Trade outcome sent to intelligence: {tradeId}");
            }
            catch (Exception ex)
            {
                Print($"Trade completion notification error: {ex.Message}");
            }
        }
        
        #endregion
        
        #region Cleanup
        
        private void CleanupWithStats()
        {
            try
            {
                Print("Black Box Intelligence shutting down...");
                
                isRunning = false;
                
                if (signalThread?.IsAlive == true)
                {
                    if (!signalThread.Join(2000))
                    {
                        Print("Signal thread did not stop gracefully");
                    }
                }
                
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
                
                Print($"Intelligence Strategy Final Stats:");
                Print($"Intelligence Signals Received: {signalCount}");
                Print($"Trades Executed: {tradesExecuted}");
                Print($"Connection Attempts: {connectionAttempts}");
                Print("Black Box Intelligence stopped - All patterns preserved");
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
                isRunning = false;
                
                if (signalThread?.IsAlive == true)
                {
                    signalThread.Join(1000);
                }
                
                featureClient?.Close();
                signalClient?.Close();
            }
            catch
            {
                // Silent cleanup
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