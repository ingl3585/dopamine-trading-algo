// ResearchStrategy.cs - REFACTORED: Pure position management using real NinjaTrader data

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
        
        // Position tracking for AI learning feedback
        private DateTime positionEntryTime = DateTime.MinValue;
        private double positionEntryPrice = 0.0;
        private string currentTradeId = "";
        private int tradeIdCounter = 0;
        private string lastToolUsed = "";
        
        // Enhanced tracking from original version
        private int signalCount;
        private int tradesExecuted;
        private int connectionAttempts;
        private DateTime lastConnectionAttempt;
        private DateTime lastTradeEntry;
        
        // AI signal data
        private bool lastSignalUsedAIStop = false;
        private bool lastSignalUsedAITarget = false;
        private double lastAIStopPrice = 0.0;
        private double lastAITargetPrice = 0.0;
        private double lastAIPositionSize = 1.0;
        
        // Connection status
        private bool isConnectedToFeatureServer;
        private bool isConnectedToSignalServer;
        
        private string DetermineExitReason(string orderName)
        {
            if (orderName.Contains("Exit"))
                return "intelligence_exit";
            else if (orderName.Contains("Reverse"))
                return "signal_reversal";
            else if (orderName.Contains("Stop"))
                return "stop_hit";
            else if (orderName.Contains("Target"))
                return "target_hit";
            else
                return "session_close";
        }
        
        #endregion
        
        #region Strategy Lifecycle
        
        protected override void OnStateChange()
        {
            try
            {
                switch (State)
                {
                    case State.SetDefaults:
                        Description = "Pure Position Management - No Hardcoded Limits";
                        Name = "ResearchStrategy";
                        Calculate = Calculate.OnBarClose;
                        
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
                        BarsRequiredToTrade = 1;
                        break;
                        
                    case State.Configure:
                        AddDataSeries(BarsPeriodType.Minute, 15);
                        AddDataSeries(BarsPeriodType.Minute, 5);
                        AddDataSeries(BarsPeriodType.Minute, 1);
                        break;
                        
                    case State.DataLoaded:
                        if (BarsArray?.Length > 0 && CurrentBars[0] > 0)
                        {
                            Print("Pure Position Management Strategy loaded - using real account data");
                        }
                        break;
                        
                    case State.Realtime:
                        Print("Entering real-time: Pure position management active");
                        ConnectToPython();
                        StartSignalReceiver();
                        
                        if (isConnectedToFeatureServer)
                        {
                            Print($"Sending historical data: 15m={prices15m.Count}, 5m={prices5m.Count}, 1m={prices1m.Count}");
                            SendMarketDataToPython();
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
                Print($"OnStateChange error: {ex.Message}");
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
                
                if (State == State.Realtime && BarsInProgress == 0 && IsFirstTickOfBar)
                {
                    if (isConnectedToFeatureServer)
                    {
                        SendMarketDataToPython();
                        
                        // Enhanced logging every 10 bars
                        if (CurrentBar % 10 == 0)
                        {
                            Print($"Intelligence Feed: Bar {CurrentBar} | " +
                                  $"Data: 15m={prices15m.Count}, 5m={prices5m.Count}, 1m={prices1m.Count} | " +
                                  $"Connections: Feature={isConnectedToFeatureServer}, Signal={isConnectedToSignalServer}");
                        }
                    }
                    else
                    {
                        // Alert if connection lost
                        if (CurrentBar % 50 == 0)
                        {
                            Print("WARNING: Intelligence engine disconnected - attempting reconnect");
                            ConnectToPython();
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
                    
                    // Track position entry
                    if (execution.Order.Name.Contains("AI_") && 
                        string.IsNullOrEmpty(currentTradeId) &&
                        (execution.Order.OrderAction == OrderAction.Buy || execution.Order.OrderAction == OrderAction.SellShort))
                    {
                        lastTradeEntry = DateTime.Now;
                        positionEntryTime = DateTime.Now;
                        positionEntryPrice = price;
                        tradeIdCounter++;
                        currentTradeId = $"trade_{tradeIdCounter}";
                        
                        // Extract tool from order name
                        if (execution.Order.Name.Contains("_dna_"))
                            lastToolUsed = "dna";
                        else if (execution.Order.Name.Contains("_micro_"))
                            lastToolUsed = "micro";
                        else if (execution.Order.Name.Contains("_temporal_"))
                            lastToolUsed = "temporal";
                        else if (execution.Order.Name.Contains("_immune_"))
                            lastToolUsed = "immune";
                        
                        Print($"AI Trade started: {currentTradeId} using {lastToolUsed} at ${price:F2}");
                    }
                    
                    // Handle position going flat - send completion data to Python
                    if (marketPosition == MarketPosition.Flat && !string.IsNullOrEmpty(currentTradeId))
                    {
                        int duration = (int)(DateTime.Now - lastTradeEntry).TotalMinutes;
                        string exitReason = DetermineExitReason(execution.Order.Name);
                        
                        SendTradeCompletionToPython(price, execution.Order.Name);
                        Print($"AI Trade completed: {currentTradeId}, Exit: ${price:F2}, Reason: {exitReason}");
                        currentTradeId = "";
                        lastToolUsed = "";
                    }
                }
            }
            catch (Exception ex)
            {
                Print($"Execution update error: {ex.Message}");
            }
        }
        
        #endregion
        
        #region Data Management
        
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
        
        private void TrimList(List<double> list, int maxCount)
        {
            if (list.Count > maxCount)
                list.RemoveAt(0);
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
                        Print("Connected to Python AI (port 5556)");
                    }
                    catch (Exception ex)
                    {
                        Print($"AI connection failed: {ex.Message}");
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
                
                if (isConnectedToFeatureServer && isConnectedToSignalServer)
                {
                    Print("Pure AI system connected - no hardcoded limits active");
                    if (State == State.Realtime)
                    {
                        Print("Black Box Intelligence fully connected");
                        Print("Pure AI Pattern Discovery - No hardcoded rules");
                    }
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
                Name = "AISignalReceiver"
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
                    ProcessAISignal(signalJson);
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
		        
		        // Add price data
		        jsonBuilder.Append("\"price_15m\":[");
		        for (int i = 0; i < prices15m.Count; i++)
		        {
		            if (i > 0) jsonBuilder.Append(",");
		            jsonBuilder.Append(prices15m[i].ToString("F6"));
		        }
		        jsonBuilder.Append("],");
		        
		        jsonBuilder.Append("\"volume_15m\":[");
		        for (int i = 0; i < volumes15m.Count; i++)
		        {
		            if (i > 0) jsonBuilder.Append(",");
		            jsonBuilder.Append(volumes15m[i].ToString("F2"));
		        }
		        jsonBuilder.Append("],");
		        
		        jsonBuilder.Append("\"price_5m\":[");
		        for (int i = 0; i < prices5m.Count; i++)
		        {
		            if (i > 0) jsonBuilder.Append(",");
		            jsonBuilder.Append(prices5m[i].ToString("F6"));
		        }
		        jsonBuilder.Append("],");
		        
		        jsonBuilder.Append("\"volume_5m\":[");
		        for (int i = 0; i < volumes5m.Count; i++)
		        {
		            if (i > 0) jsonBuilder.Append(",");
		            jsonBuilder.Append(volumes5m[i].ToString("F2"));
		        }
		        jsonBuilder.Append("],");
		        
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
		        
		        // CRITICAL FIX: Add account data for AI position sizing as required by prompt
		        jsonBuilder.Append($"\"buying_power\":{Account.Get(AccountItem.BuyingPower, Currency.UsDollar):F2},");
		        jsonBuilder.Append($"\"account_balance\":{Account.Get(AccountItem.CashValue, Currency.UsDollar):F2},");
		        jsonBuilder.Append($"\"daily_pnl\":{Account.Get(AccountItem.RealizedProfitLoss, Currency.UsDollar):F2},");
		        jsonBuilder.Append($"\"cash_value\":{Account.Get(AccountItem.CashValue, Currency.UsDollar):F2},");
		        jsonBuilder.Append($"\"excess_liquidity\":{Account.Get(AccountItem.ExcessInitialMargin, Currency.UsDollar):F2},");
		        jsonBuilder.Append($"\"net_liquidation\":{Account.Get(AccountItem.NetLiquidation, Currency.UsDollar):F2},");
		        
		        jsonBuilder.Append($"\"timestamp\":{DateTime.Now.Ticks}");
		        jsonBuilder.Append("}");
		        
		        string json = jsonBuilder.ToString();
		        byte[] jsonBytes = Encoding.UTF8.GetBytes(json);
		        byte[] header = BitConverter.GetBytes(jsonBytes.Length);
		        
		        var stream = featureClient.GetStream();
		        stream.Write(header, 0, 4);
		        stream.Write(jsonBytes, 0, jsonBytes.Length);
		        
				if (CurrentBar % 100 == 0)  // Log every 100 bars instead of 50
				{
				    double balance = Account.Get(AccountItem.CashValue, Currency.UsDollar);
				    double buyingPower = Account.Get(AccountItem.BuyingPower, Currency.UsDollar);
				    Print($"Account data sent to AI: Balance=${balance:F0}, BP=${buyingPower:F0}");
				}
		    }
		    catch (Exception ex)
		    {
		        Print($"Data send error: {ex.Message}");
		        isConnectedToFeatureServer = false;
		    }
		}
        
		private void SendTradeCompletionToPython(double exitPrice, string orderName)
		{
		    if (featureClient?.Connected != true || string.IsNullOrEmpty(currentTradeId))
		        return;
		    
		    try
		    {
		        string exitReason = "unknown";
		        if (orderName.Contains("Stop"))
		            exitReason = "stop_hit";
		        else if (orderName.Contains("Target"))
		            exitReason = "target_hit";
		        else if (orderName.Contains("Exit"))
		            exitReason = "ai_exit";
		        else
		            exitReason = "manual_close";
		        
		        int durationMinutes = (int)(DateTime.Now - positionEntryTime).TotalMinutes;
		        
		        // Get real account data
		        double realizedPnl = SystemPerformance.AllTrades.Count > 0 ? 
		            SystemPerformance.AllTrades[SystemPerformance.AllTrades.Count - 1].ProfitCurrency : 0.0;
		        
		        var completionData = new StringBuilder();
		        completionData.Append("{");
		        completionData.Append("\"type\":\"trade_completion\",");
		        completionData.Append($"\"trade_id\":\"{currentTradeId}\",");
		        completionData.Append($"\"signal_timestamp\":{positionEntryTime.Ticks},");
		        completionData.Append($"\"entry_price\":{positionEntryPrice:F2},");
		        completionData.Append($"\"exit_price\":{exitPrice:F2},");
		        completionData.Append($"\"final_pnl\":{realizedPnl:F2},");
		        completionData.Append($"\"exit_reason\":\"{exitReason}\",");
		        completionData.Append($"\"duration_minutes\":{durationMinutes},");
		        completionData.Append($"\"tool_used\":\"{lastToolUsed}\",");
		        completionData.Append($"\"used_ai_stop\":{(lastSignalUsedAIStop ? "true" : "false")},");
		        completionData.Append($"\"used_ai_target\":{(lastSignalUsedAITarget ? "true" : "false")},");
		        
		        // Enhanced correlation data for better learning
		        completionData.Append($"\"position_size\":{lastAIPositionSize:F2},");
		        completionData.Append($"\"account_balance\":{Account.Get(AccountItem.CashValue, Currency.UsDollar):F2},");
		        completionData.Append($"\"buying_power\":{Account.Get(AccountItem.BuyingPower, Currency.UsDollar):F2},");
		        completionData.Append($"\"daily_pnl\":{Account.Get(AccountItem.RealizedProfitLoss, Currency.UsDollar):F2},");
		        
		        // Add AI learning correlation data
		        completionData.Append($"\"ai_stop_price\":{lastAIStopPrice:F2},");
		        completionData.Append($"\"ai_target_price\":{lastAITargetPrice:F2},");
		        completionData.Append($"\"entry_timestamp\":{positionEntryTime.Ticks},");
		        completionData.Append($"\"completion_timestamp\":{DateTime.Now.Ticks},");
		        
		        completionData.Append($"\"timestamp\":{DateTime.Now.Ticks}");
		        completionData.Append("}");
		        
		        string json = completionData.ToString();
		        byte[] jsonBytes = Encoding.UTF8.GetBytes(json);
		        byte[] header = BitConverter.GetBytes(jsonBytes.Length);
		        
		        var stream = featureClient.GetStream();
		        stream.Write(header, 0, 4);
		        stream.Write(jsonBytes, 0, jsonBytes.Length);
		        
		        Print($"Trade completion sent to AI: {currentTradeId}, P&L: ${realizedPnl:F2}, Tool: {lastToolUsed}");
		    }
		    catch (Exception ex)
		    {
		        Print($"Trade completion error: {ex.Message}");
		    }
		}
        
        #endregion
        
        #region Signal Processing
        
        private void ProcessAISignal(string signalJson)
        {
            var signal = ParseSignalJson(signalJson);
            if (signal == null) return;
            
            // Store AI recommendations
            lastSignalUsedAIStop = signal.use_stop;
            lastSignalUsedAITarget = signal.use_target;
            lastAIStopPrice = signal.stop_price;
            lastAITargetPrice = signal.target_price;
            lastAIPositionSize = signal.position_size;
            
            ExecuteAISignal(signal);
        }
        
        private SignalData ParseSignalJson(string json)
        {
            try 
            {
                var signal = new SignalData();
                
                // Parse action
                var actionStart = json.IndexOf("\"action\":") + 9;
                if (actionStart > 8)
                {
                    var actionEnd = json.IndexOf(",", actionStart);
                    if (actionEnd == -1) actionEnd = json.IndexOf("}", actionStart);
                    if (int.TryParse(json.Substring(actionStart, actionEnd - actionStart).Trim(), out int actionValue))
                        signal.action = actionValue;
                }
                
                // Parse confidence
                var confStart = json.IndexOf("\"confidence\":") + 13;
                if (confStart > 12)
                {
                    var confEnd = json.IndexOf(",", confStart);
                    if (confEnd == -1) confEnd = json.IndexOf("}", confStart);
                    if (double.TryParse(json.Substring(confStart, confEnd - confStart).Trim(), out double confValue))
                        signal.confidence = confValue;
                }
                
                // Parse tool_used
                var toolPattern = "\"tool_used\":\"";
                var toolStart = json.IndexOf(toolPattern);
                if (toolStart >= 0)
                {
                    toolStart += toolPattern.Length;
                    var toolEnd = json.IndexOf("\"", toolStart);
                    if (toolEnd > toolStart)
                    {
                        signal.tool_used = json.Substring(toolStart, toolEnd - toolStart);
                    }
                }
                
                // Parse use_stop
                var useStopPattern = "\"use_stop\":";
                var useStopStart = json.IndexOf(useStopPattern);
                if (useStopStart >= 0)
                {
                    useStopStart += useStopPattern.Length;
                    var useStopEnd = json.IndexOf(",", useStopStart);
                    if (useStopEnd == -1) useStopEnd = json.IndexOf("}", useStopStart);
                    
                    string useStopStr = json.Substring(useStopStart, useStopEnd - useStopStart).Trim();
                    signal.use_stop = useStopStr.ToLower() == "true";
                }
                
                // Parse stop_price
                var stopPattern = "\"stop_price\":";
                var stopStart = json.IndexOf(stopPattern);
                if (stopStart >= 0)
                {
                    stopStart += stopPattern.Length;
                    var stopEnd = json.IndexOf(",", stopStart);
                    if (stopEnd == -1) stopEnd = json.IndexOf("}", stopStart);
                    
                    if (double.TryParse(json.Substring(stopStart, stopEnd - stopStart).Trim(), out double stopValue))
                    {
                        signal.stop_price = stopValue;
                    }
                }
                
                // Parse use_target
                var useTargetPattern = "\"use_target\":";
                var useTargetStart = json.IndexOf(useTargetPattern);
                if (useTargetStart >= 0)
                {
                    useTargetStart += useTargetPattern.Length;
                    var useTargetEnd = json.IndexOf(",", useTargetStart);
                    if (useTargetEnd == -1) useTargetEnd = json.IndexOf("}", useTargetStart);
                    
                    string useTargetStr = json.Substring(useTargetStart, useTargetEnd - useTargetStart).Trim();
                    signal.use_target = useTargetStr.ToLower() == "true";
                }
                
                // Parse target_price
                var targetPattern = "\"target_price\":";
                var targetStart = json.IndexOf(targetPattern);
                if (targetStart >= 0)
                {
                    targetStart += targetPattern.Length;
                    var targetEnd = json.IndexOf(",", targetStart);
                    if (targetEnd == -1) targetEnd = json.IndexOf("}", targetStart);
                    
                    if (double.TryParse(json.Substring(targetStart, targetEnd - targetStart).Trim(), out double targetValue))
                    {
                        signal.target_price = targetValue;
                    }
                }
                
                // Parse quality
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
                }
                
                // Parse timestamp
                var timePattern = "\"timestamp\":";
                var timeStart = json.IndexOf(timePattern);
                if (timeStart >= 0)
                {
                    timeStart += timePattern.Length;
                    var timeEnd = json.IndexOf(",", timeStart);
                    if (timeEnd == -1) timeEnd = json.IndexOf("}", timeStart);
                    
                    if (long.TryParse(json.Substring(timeStart, timeEnd - timeStart).Trim(), out long timeValue))
                    {
                        signal.timestamp = timeValue;
                    }
                }
                
                // Parse position_size
                var positionSizePattern = "\"position_size\":";
                var positionSizeStart = json.IndexOf(positionSizePattern);
                if (positionSizeStart >= 0)
                {
                    positionSizeStart += positionSizePattern.Length;
                    var positionSizeEnd = json.IndexOf(",", positionSizeStart);
                    if (positionSizeEnd == -1) positionSizeEnd = json.IndexOf("}", positionSizeStart);
                    
                    if (double.TryParse(json.Substring(positionSizeStart, positionSizeEnd - positionSizeStart).Trim(), out double positionSizeValue))
                    {
                        signal.position_size = positionSizeValue;
                    }
                }
                
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
		    
		    // Use adaptive signal timeout from Python AI system
		    // Default to 30 seconds if not provided by AI
		    double signalTimeoutSeconds = signal.adaptive_timeout > 0 ? signal.adaptive_timeout : 30.0;
		    
		    // Basic signal age protection with AI-determined timeout
		    var signalAge = (DateTime.Now.Ticks - signal.timestamp) / TimeSpan.TicksPerSecond;
		    if (signalAge > signalTimeoutSeconds)
		    {
		        Print($"Signal too old: {signalAge}s > {signalTimeoutSeconds}s (AI timeout) - skipping");
		        return false;
		    }
		    
		    return true;
		}
        
        private void ExecuteAISignal(SignalData signal)
        {
            try
            {
                if (!IsValidSignal(signal)) return;
                
                signalCount++;
                
                string toolUsed = "unknown";
				if (!string.IsNullOrEmpty(signal.quality))
				{
				    // Extract tool name from quality field like "dna_tool" or "immune_tool"
				    if (signal.quality.Contains("dna"))
				        toolUsed = "dna";
				    else if (signal.quality.Contains("micro"))
				        toolUsed = "micro";
				    else if (signal.quality.Contains("temporal"))
				        toolUsed = "temporal";
				    else if (signal.quality.Contains("immune"))
				        toolUsed = "immune";
				    else
				        toolUsed = signal.quality.Split('_')[0]; // First part before underscore
				}
				// Store for execution tracking
				lastToolUsed = toolUsed;
                
                switch (signal.action)
                {
                    case 1: // Buy
                        if (Position.MarketPosition != MarketPosition.Long)
                        {
                            if (Position.MarketPosition == MarketPosition.Short)
                                ExitShort("AI_Exit");
                            
                            // Use AI calculated position size
                            int aiPositionSize = Math.Max(1, (int)Math.Round(signal.position_size));
                            
                            string longOrderName = $"AI_{toolUsed}_Long";
                            EnterLong(aiPositionSize, longOrderName);
                            
                            // Use AI's risk management decisions
                            if (signal.use_stop && signal.stop_price > 0)
                            {
                                SetStopLoss(longOrderName, CalculationMode.Price, signal.stop_price, false);
                                Print($"  AI Stop Loss: ${signal.stop_price:F2}");
                            }
                            else
                            {
                                Print("  AI chose NO stop loss");
                            }
                            
                            if (signal.use_target && signal.target_price > 0)
                            {
                                SetProfitTarget(longOrderName, CalculationMode.Price, signal.target_price);
                                Print($"  AI Take Profit: ${signal.target_price:F2}");
                            }
                            else
                            {
                                Print("  AI chose NO take profit target");
                            }
                            
                            Print($"AI Signal: BUY using {lastToolUsed} tool, AI Size: {aiPositionSize}, Confidence: {signal.confidence:F3}");
                            
                            Print($"Intelligence Signal #{signalCount}: Action=BUY, " +
                                  $"Confidence={signal.confidence:F3}, Quality={signal.quality ?? "unknown"}");
                            Print($"BLACK BOX LONG using {lastToolUsed} TOOL: conf={signal.confidence:F3}");
                        }
                        break;
                        
                    case 2: // Sell
                        if (Position.MarketPosition != MarketPosition.Short)
                        {
                            if (Position.MarketPosition == MarketPosition.Long)
                                ExitLong("AI_Exit");
                            
                            // Use AI calculated position size
                            int aiPositionSize = Math.Max(1, (int)Math.Round(signal.position_size));
                            
                            string shortOrderName = $"AI_{toolUsed}_Short";
                            EnterShort(aiPositionSize, shortOrderName);
                            
                            if (signal.use_stop && signal.stop_price > 0)
                            {
                                SetStopLoss(shortOrderName, CalculationMode.Price, signal.stop_price, false);
                                Print($"  AI Stop Loss: ${signal.stop_price:F2}");
                            }
                            else
                            {
                                Print("  AI chose NO stop loss");
                            }
                            
                            if (signal.use_target && signal.target_price > 0)
                            {
                                SetProfitTarget(shortOrderName, CalculationMode.Price, signal.target_price);
                                Print($"  AI Take Profit: ${signal.target_price:F2}");
                            }
                            else
                            {
                                Print("  AI chose NO take profit target");
                            }
                            
                            Print($"AI Signal: SELL using {toolUsed} tool, AI Size: {aiPositionSize}, Confidence: {signal.confidence:F3}");
                            
                            Print($"Intelligence Signal #{signalCount}: Action=SELL, " +
                                  $"Confidence={signal.confidence:F3}, Quality={signal.quality ?? "unknown"}");
                            Print($"BLACK BOX SHORT using {toolUsed} TOOL: conf={signal.confidence:F3}");
                        }
                        break;
                        
                    case 0: // Exit
                        if (Position.MarketPosition != MarketPosition.Flat)
                        {
                            if (Position.MarketPosition == MarketPosition.Long)
                                ExitLong($"AI_{toolUsed}_Exit");
                            else if (Position.MarketPosition == MarketPosition.Short)
                                ExitShort($"AI_{toolUsed}_Exit");
                            
                            Print($"AI Signal: EXIT using {toolUsed} tool");
                            
                            Print($"Intelligence Signal #{signalCount}: Action=EXIT, " +
                                  $"Confidence={signal.confidence:F3}, Quality={signal.quality ?? "unknown"}");
                            Print($"BLACK BOX EXIT using {toolUsed} TOOL");
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
        
        private void Cleanup()
        {
            if (signalCount > 0 || tradesExecuted > 0)
            {
                CleanupWithStats();
            }
            else
            {
                QuietCleanup();
            }
        }
        
        #endregion
        
        #region Helper Classes
        
		public class SignalData
		{
		    public int action { get; set; }
		    public double confidence { get; set; }
		    public string quality { get; set; }
		    public string tool_used { get; set; }
		    public double position_size { get; set; } = 1.0;
		    public long timestamp { get; set; }
		    public bool use_stop { get; set; } = false;
		    public double stop_price { get; set; } = 0.0;
		    public bool use_target { get; set; } = false;
		    public double target_price { get; set; } = 0.0;
		    public double adaptive_timeout { get; set; } = 30.0;  // NEW: AI-determined timeout
		}
        
        #endregion
    }
}