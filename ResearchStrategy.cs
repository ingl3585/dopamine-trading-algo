// ResearchStrategy.cs - SIMPLIFIED for Black Box Trading System

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
        
        // TCP connections
        private TcpClient featureClient;
        private TcpClient signalClient;
        private Thread signalThread;
        private bool isRunning;
        
        // Market data storage
        private List<double> prices15m = new List<double>();
        private List<double> volumes15m = new List<double>();
        private List<double> prices5m = new List<double>();
        private List<double> volumes5m = new List<double>();
        private List<double> prices1m = new List<double>();
        private List<double> volumes1m = new List<double>();
        
        // Trade tracking for learning
        private string currentTradeId = "";
        private int tradeCounter = 0;
        private DateTime tradeStartTime;
        private double entryPrice = 0.0;
        private string lastToolUsed = "";
        
        // Connection status
        private bool isConnectedToFeatures;
        private bool isConnectedToSignals;
        private int totalSignalsReceived = 0;
        private int totalTradesExecuted = 0;
        
        #endregion
        
        #region Strategy Setup
        
        protected override void OnStateChange()
        {
            switch (State)
            {
                case State.SetDefaults:
                    Description = "Simplified Black Box Trading Strategy";
                    Name = "ResearchStrategy";
                    Calculate = Calculate.OnBarClose;
                    
                    EntryHandling = EntryHandling.AllEntries;
                    ExitOnSessionCloseSeconds = 30;
                    MaximumBarsLookBack = MaximumBarsLookBack.TwoHundredFiftySix;
                    OrderFillResolution = OrderFillResolution.Standard;
                    Slippage = 0;
                    StartBehavior = StartBehavior.WaitUntilFlat;
                    TimeInForce = TimeInForce.Gtc;
                    TraceOrders = false;
                    RealtimeErrorHandling = RealtimeErrorHandling.StopCancelClose;
                    StopTargetHandling = StopTargetHandling.PerEntryExecution;
                    BarsRequiredToTrade = 1;
                    EntriesPerDirection = 5;
                    break;
                    
                case State.Configure:
                    // Add multiple timeframe data
                    AddDataSeries(BarsPeriodType.Minute, 15);
                    AddDataSeries(BarsPeriodType.Minute, 5);
                    AddDataSeries(BarsPeriodType.Minute, 1);
                    break;
                    
                case State.DataLoaded:
                    Print("Black Box Strategy loaded - connecting to Python AI");
                    break;
                    
                case State.Realtime:
                    Print("Entering real-time mode");
                    ConnectToPython();
                    StartSignalReceiver();
                    break;
                    
                case State.Terminated:
                    Cleanup();
                    break;
            }
        }
        
        #endregion
        
        #region Data Collection
        
        protected override void OnBarUpdate()
        {
            // Only process in real-time with sufficient data
            if (State != State.Realtime || CurrentBars[0] < 1 || CurrentBars[1] < 1 || 
                CurrentBars[2] < 1 || CurrentBars[3] < 1)
                return;
            
            try
            {
                // Update data on new bars
                if (IsFirstTickOfBar)
                {
                    switch (BarsInProgress)
                    {
                        case 1: // 15-minute data
                            UpdateTimeframeData(prices15m, volumes15m, Closes[1][0], Volumes[1][0], 100);
                            break;
                        case 2: // 5-minute data
                            UpdateTimeframeData(prices5m, volumes5m, Closes[2][0], Volumes[2][0], 300);
                            break;
                        case 3: // 1-minute data
                            UpdateTimeframeData(prices1m, volumes1m, Closes[3][0], Volumes[3][0], 1000);
                            break;
                    }
                }
                
                // Send data to Python on primary timeframe new bar
                if (BarsInProgress == 0 && IsFirstTickOfBar && isConnectedToFeatures)
                {
                    SendMarketDataToPython();
                }
            }
            catch (Exception ex)
            {
                Print($"OnBarUpdate error: {ex.Message}");
            }
        }
        
        private void UpdateTimeframeData(List<double> prices, List<double> volumes, 
                                       double newPrice, double newVolume, int maxCount)
        {
            prices.Add(newPrice);
            volumes.Add(newVolume);
            
            // Keep only recent data
            if (prices.Count > maxCount)
            {
                prices.RemoveAt(0);
                volumes.RemoveAt(0);
            }
        }
        
        #endregion
        
        #region TCP Communication
        
        private void ConnectToPython()
        {
            try
            {
                // Connect to feature server (port 5556)
                if (!isConnectedToFeatures)
                {
                    featureClient = new TcpClient("localhost", 5556);
                    isConnectedToFeatures = true;
                    Print("Connected to Python AI on port 5556");
                }
                
                // Connect to signal server (port 5557)
                if (!isConnectedToSignals)
                {
                    signalClient = new TcpClient("localhost", 5557);
                    isConnectedToSignals = true;
                    Print("Connected to signal server on port 5557");
                }
                
                if (isConnectedToFeatures && isConnectedToSignals)
                {
                    Print("Black Box AI system fully connected");
                }
            }
            catch (Exception ex)
            {
                Print($"Connection error: {ex.Message}");
                isConnectedToFeatures = false;
                isConnectedToSignals = false;
            }
        }
        
        private void SendMarketDataToPython()
        {
            if (!isConnectedToFeatures || featureClient?.Connected != true)
                return;
            
            try
            {
                // Build simple JSON message
                var jsonBuilder = new StringBuilder();
                jsonBuilder.Append("{");
                
                // Price data
                AppendPriceData(jsonBuilder, "price_15m", prices15m);
                AppendPriceData(jsonBuilder, "volume_15m", volumes15m);
                AppendPriceData(jsonBuilder, "price_5m", prices5m);
                AppendPriceData(jsonBuilder, "volume_5m", volumes5m);
                AppendPriceData(jsonBuilder, "price_1m", prices1m);
                AppendPriceData(jsonBuilder, "volume_1m", volumes1m);
                
                // Account data for position sizing
                jsonBuilder.Append($"\"account_balance\":{Account.Get(AccountItem.CashValue, Currency.UsDollar):F2},");
                jsonBuilder.Append($"\"buying_power\":{Account.Get(AccountItem.BuyingPower, Currency.UsDollar):F2},");
                jsonBuilder.Append($"\"daily_pnl\":{Account.Get(AccountItem.RealizedProfitLoss, Currency.UsDollar):F2},");
                jsonBuilder.Append($"\"cash_value\":{Account.Get(AccountItem.CashValue, Currency.UsDollar):F2},");
                
                jsonBuilder.Append($"\"timestamp\":{DateTime.Now.Ticks}");
                jsonBuilder.Append("}");
                
                // Send to Python
                string json = jsonBuilder.ToString();
                byte[] data = Encoding.UTF8.GetBytes(json);
                byte[] header = BitConverter.GetBytes(data.Length);
                
                var stream = featureClient.GetStream();
                stream.Write(header, 0, 4);
                stream.Write(data, 0, data.Length);
            }
            catch (Exception ex)
            {
                Print($"Data send error: {ex.Message}");
                isConnectedToFeatures = false;
            }
        }
        
        private void AppendPriceData(StringBuilder json, string name, List<double> data)
        {
            json.Append($"\"{name}\":[");
            for (int i = 0; i < data.Count; i++)
            {
                if (i > 0) json.Append(",");
                json.Append(data[i].ToString("F6"));
            }
            json.Append("],");
        }
        
        private void StartSignalReceiver()
        {
            if (isRunning) return;
            
            isRunning = true;
            signalThread = new Thread(ReceiveSignals)
            {
                IsBackground = true,
                Name = "SignalReceiver"
            };
            signalThread.Start();
            Print("Signal receiver started");
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
                    
                    // Read message header
                    var headerBytes = new byte[4];
                    int headerRead = 0;
                    
                    while (headerRead < 4)
                    {
                        int bytesRead = signalClient.GetStream().Read(headerBytes, headerRead, 4 - headerRead);
                        if (bytesRead == 0)
                        {
                            Print("Signal connection lost");
                            isConnectedToSignals = false;
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
                    
                    // Read message data
                    var messageBytes = new byte[messageLength];
                    int totalRead = 0;
                    
                    while (totalRead < messageLength)
                    {
                        int bytesRead = signalClient.GetStream().Read(
                            messageBytes, totalRead, messageLength - totalRead);
                        if (bytesRead == 0)
                        {
                            Print("Signal data connection lost");
                            isConnectedToSignals = false;
                            return;
                        }
                        totalRead += bytesRead;
                    }
                    
                    // Process signal
                    string signalJson = Encoding.UTF8.GetString(messageBytes);
                    ProcessTradingSignal(signalJson);
                }
                catch (Exception ex)
                {
                    Print($"Signal receive error: {ex.Message}");
                    Thread.Sleep(2000);
                }
            }
        }
        
        #endregion
        
        #region Signal Processing
        
        private void ProcessTradingSignal(string signalJson)
        {
            try
            {
                var signal = ParseSignal(signalJson);
                if (signal == null) return;
                
                totalSignalsReceived++;
                
                string actionName = signal.action == 1 ? "BUY" : signal.action == 2 ? "SELL" : "EXIT";
                Print($"AI Signal #{totalSignalsReceived}: {actionName} " +
                      $"(conf: {signal.confidence:F3}, size: {signal.position_size:F1}, tool: {signal.tool_used})");
                
                ExecuteSignal(signal);
            }
            catch (Exception ex)
            {
                Print($"Signal processing error: {ex.Message}");
            }
        }
        
        private TradingSignal ParseSignal(string json)
        {
            try
            {
                var signal = new TradingSignal();
                
                // Simple JSON parsing for core fields
                signal.action = ExtractIntValue(json, "action");
                signal.confidence = ExtractDoubleValue(json, "confidence");
                signal.position_size = ExtractDoubleValue(json, "position_size");
                signal.use_stop = ExtractBoolValue(json, "use_stop");
                signal.stop_price = ExtractDoubleValue(json, "stop_price");
                signal.use_target = ExtractBoolValue(json, "use_target");
                signal.target_price = ExtractDoubleValue(json, "target_price");
                signal.tool_used = ExtractStringValue(json, "tool_used");
                
                return signal;
            }
            catch (Exception ex)
            {
                Print($"Signal parsing error: {ex.Message}");
                return null;
            }
        }
        
        private void ExecuteSignal(TradingSignal signal)
        {
            try
            {
                int quantity = Math.Max(1, (int)Math.Round(signal.position_size));
                string entryName = $"AI_{signal.tool_used}_{DateTime.Now:HHmmss}";
                
                // Store trade info for completion tracking
                lastToolUsed = signal.tool_used;
                
                switch (signal.action)
                {
                    case 1: // BUY
                        if (Position.MarketPosition == MarketPosition.Short)
                        {
                            ExitShort("AI_Reverse");
                        }
                        
                        EnterLong(quantity, entryName);
                        
                        // Set stops and targets if AI decided to use them
                        if (signal.use_stop && signal.stop_price > 0)
                        {
                            SetStopLoss(entryName, CalculationMode.Price, signal.stop_price, false);
                        }
                        if (signal.use_target && signal.target_price > 0)
                        {
                            SetProfitTarget(entryName, CalculationMode.Price, signal.target_price);
                        }
                        break;
                        
                    case 2: // SELL
                        if (Position.MarketPosition == MarketPosition.Long)
                        {
                            ExitLong("AI_Reverse");
                        }
                        
                        EnterShort(quantity, entryName);
                        
                        if (signal.use_stop && signal.stop_price > 0)
                        {
                            SetStopLoss(entryName, CalculationMode.Price, signal.stop_price, false);
                        }
                        if (signal.use_target && signal.target_price > 0)
                        {
                            SetProfitTarget(entryName, CalculationMode.Price, signal.target_price);
                        }
                        break;
                        
                    case 0: // EXIT
                        if (Position.MarketPosition == MarketPosition.Long)
                            ExitLong("AI_Exit");
                        else if (Position.MarketPosition == MarketPosition.Short)
                            ExitShort("AI_Exit");
                        break;
                }
            }
            catch (Exception ex)
            {
                Print($"Signal execution error: {ex.Message}");
            }
        }
        
        #endregion
        
        #region Trade Execution Tracking
        
        protected override void OnExecutionUpdate(Execution execution, string executionId, 
                                                double price, int quantity, MarketPosition marketPosition, 
                                                string orderId, DateTime time)
        {
            try
            {
                if (execution.Order?.Name?.Contains("AI_") == true)
                {
                    totalTradesExecuted++;
                    
                    // Track trade entry
                    if (execution.Order.OrderAction == OrderAction.Buy || 
                        execution.Order.OrderAction == OrderAction.SellShort)
                    {
                        if (string.IsNullOrEmpty(currentTradeId))
                        {
                            tradeCounter++;
                            currentTradeId = $"trade_{tradeCounter}";
                            tradeStartTime = DateTime.Now;
                            entryPrice = price;
                            
                            Print($"Trade started: {currentTradeId} at ${price:F2} using {lastToolUsed}");
                        }
                    }
                    
                    // Track trade completion (position goes flat)
                    if (marketPosition == MarketPosition.Flat && !string.IsNullOrEmpty(currentTradeId))
                    {
                        SendTradeCompletionToPython(price, execution.Order.Name);
                        
                        Print($"Trade completed: {currentTradeId} exit at ${price:F2}");
                        currentTradeId = "";
                    }
                }
            }
            catch (Exception ex)
            {
                Print($"Execution tracking error: {ex.Message}");
            }
        }
        
        private void SendTradeCompletionToPython(double exitPrice, string orderName)
        {
            if (!isConnectedToFeatures || featureClient?.Connected != true || string.IsNullOrEmpty(currentTradeId))
                return;
            
            try
            {
                // Determine exit reason
                string exitReason = "unknown";
                if (orderName.Contains("Stop"))
                    exitReason = "stop_hit";
                else if (orderName.Contains("Target") || orderName.Contains("Profit"))
                    exitReason = "target_hit";
                else if (orderName.Contains("Exit"))
                    exitReason = "ai_exit";
                else if (orderName.Contains("Reverse"))
                    exitReason = "signal_reversal";
                else
                    exitReason = "session_close";
                
                // Calculate P&L
                double pnl = 0.0;
                if (SystemPerformance.AllTrades.Count > 0)
                {
                    var lastTrade = SystemPerformance.AllTrades[SystemPerformance.AllTrades.Count - 1];
                    pnl = lastTrade.ProfitCurrency;
                }
                
                int durationMinutes = (int)(DateTime.Now - tradeStartTime).TotalMinutes;
                
                // Build completion message
                var completionData = new StringBuilder();
                completionData.Append("{");
                completionData.Append("\"type\":\"trade_completion\",");
                completionData.Append($"\"trade_id\":\"{currentTradeId}\",");
                completionData.Append($"\"entry_price\":{entryPrice:F2},");
                completionData.Append($"\"exit_price\":{exitPrice:F2},");
                completionData.Append($"\"final_pnl\":{pnl:F2},");
                completionData.Append($"\"exit_reason\":\"{exitReason}\",");
                completionData.Append($"\"duration_minutes\":{durationMinutes},");
                completionData.Append($"\"tool_used\":\"{lastToolUsed}\",");
                completionData.Append($"\"signal_timestamp\":{tradeStartTime.Ticks},");
                completionData.Append($"\"timestamp\":{DateTime.Now.Ticks}");
                completionData.Append("}");
                
                // Send to Python
                string json = completionData.ToString();
                byte[] data = Encoding.UTF8.GetBytes(json);
                byte[] header = BitConverter.GetBytes(data.Length);
                
                var stream = featureClient.GetStream();
                stream.Write(header, 0, 4);
                stream.Write(data, 0, data.Length);
                
                Print($"Trade completion sent to AI: {currentTradeId}, P&L: ${pnl:F2}");
            }
            catch (Exception ex)
            {
                Print($"Trade completion error: {ex.Message}");
            }
        }
        
        #endregion
        
        #region Helper Methods
        
        private int ExtractIntValue(string json, string key)
        {
            var pattern = $"\"{key}\":";
            var start = json.IndexOf(pattern);
            if (start < 0) return 0;
            
            start += pattern.Length;
            var end = json.IndexOf(",", start);
            if (end < 0) end = json.IndexOf("}", start);
            
            if (int.TryParse(json.Substring(start, end - start).Trim(), out int result))
                return result;
            
            return 0;
        }
        
        private double ExtractDoubleValue(string json, string key)
        {
            var pattern = $"\"{key}\":";
            var start = json.IndexOf(pattern);
            if (start < 0) return 0.0;
            
            start += pattern.Length;
            var end = json.IndexOf(",", start);
            if (end < 0) end = json.IndexOf("}", start);
            
            if (double.TryParse(json.Substring(start, end - start).Trim(), out double result))
                return result;
            
            return 0.0;
        }
        
        private bool ExtractBoolValue(string json, string key)
        {
            var pattern = $"\"{key}\":";
            var start = json.IndexOf(pattern);
            if (start < 0) return false;
            
            start += pattern.Length;
            var end = json.IndexOf(",", start);
            if (end < 0) end = json.IndexOf("}", start);
            
            string value = json.Substring(start, end - start).Trim().ToLower();
            return value == "true";
        }
        
        private string ExtractStringValue(string json, string key)
        {
            var pattern = $"\"{key}\":\"";
            var start = json.IndexOf(pattern);
            if (start < 0) return "";
            
            start += pattern.Length;
            var end = json.IndexOf("\"", start);
            
            if (end > start)
                return json.Substring(start, end - start);
            
            return "";
        }
        
        #endregion
        
        #region Cleanup
        
        private void Cleanup()
        {
            try
            {
                Print("Black Box Strategy shutting down...");
                
                isRunning = false;
                
                // Stop signal thread
                if (signalThread?.IsAlive == true)
                {
                    signalThread.Join(2000);
                }
                
                // Close connections
                try
                {
                    featureClient?.Close();
                    signalClient?.Close();
                }
                catch (Exception ex)
                {
                    Print($"Connection cleanup error: {ex.Message}");
                }
                
                Print($"Strategy stopped. Signals received: {totalSignalsReceived}, Trades: {totalTradesExecuted}");
            }
            catch (Exception ex)
            {
                Print($"Cleanup error: {ex.Message}");
            }
        }
        
        #endregion
        
        #region Helper Classes
        
        private class TradingSignal
        {
            public int action { get; set; }
            public double confidence { get; set; }
            public double position_size { get; set; } = 1.0;
            public bool use_stop { get; set; } = false;
            public double stop_price { get; set; } = 0.0;
            public bool use_target { get; set; } = false;
            public double target_price { get; set; } = 0.0;
            public string tool_used { get; set; } = "unknown";
        }
        
        #endregion
    }
}