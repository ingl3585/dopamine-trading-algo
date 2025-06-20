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
using System.Globalization;

namespace NinjaTrader.NinjaScript.Strategies
{
    public class ResearchStrategy : Strategy
    {
        private TcpClient dataClient;
        private TcpClient signalClient;
        private Thread signalThread;
        private bool isConnected;
        private bool isRunning;
        private bool historicalDataSent = false;
        
        private List<double> prices1m = new List<double>();
        private List<double> prices5m = new List<double>();
        private List<double> prices15m = new List<double>();
        private List<double> volumes1m = new List<double>();
        private List<double> volumes5m = new List<double>();
        private List<double> volumes15m = new List<double>();
        
        // Enhanced account tracking
        private double lastAccountBalance = 0;
        private double lastBuyingPower = 0;
        private double sessionStartPnL = 0;
        private bool sessionStartSet = false;
		
		// Data sending tracking
        private DateTime lastDataSent = DateTime.MinValue;
        private int dataSendCount = 0;
        
        protected override void OnStateChange()
        {
            switch (State)
            {
				case State.SetDefaults:
				    Description = "Adaptive AI Trading Strategy with Historical Bootstrapping";
				    Name = "ResearchStrategy";
				    Calculate = Calculate.OnBarClose;
				    BarsRequiredToTrade = 1;
				    break;
				
				case State.Configure:
				    AddDataSeries(BarsPeriodType.Minute, 15);
				    AddDataSeries(BarsPeriodType.Minute, 5);
				    AddDataSeries(BarsPeriodType.Minute, 1);
				    break;
                    
                case State.Realtime:
                    ConnectToPython();
                    StartSignalReceiver();
                    InitializeSession();
                    break;
                    
                case State.Terminated:
                    Cleanup();
                    break;
            }
        }
        
        private void InitializeSession()
        {
            if (!sessionStartSet)
            {
                sessionStartPnL = Account.Get(AccountItem.RealizedProfitLoss, Currency.UsDollar);
                sessionStartSet = true;
                Print($"Session initialized - Starting P&L: {sessionStartPnL:C}");
            }
        }
        
		protected override void OnBarUpdate()
		{
		    // Only process on the primary series (BarsInProgress == 0)
		    // OnBarUpdate gets called for each timeframe: 0=primary, 1=15m, 2=5m, 3=1m
		    if (BarsInProgress != 0 || State != State.Realtime)
		        return;
		    
		    // Update price data from all available series when processing primary series
		    UpdatePriceData();
		    
		    if (!isConnected)
		        return;
		    
		    // Send historical data once
		    if (!historicalDataSent)
		    {
		        SendHistoricalData();
		        historicalDataSent = true;
		        return; // Don't send live data on the same bar as historical
		    }
		    
		    // Send live data on every bar close of primary series
		    if (HasValidData())
		    {
		        SendDataToPython();
		        dataSendCount++;
		        
		        // Debug logging every 5 sends
		        if (dataSendCount % 5 == 0)
		        {
		            Print($"Live data sent #{dataSendCount} - Primary: {Close[0]:F2}, 15m: {(BarsArray.Length > 1 && BarsArray[1].Count > 0 ? Closes[1][0].ToString("F2") : "N/A")}, Time: {Time[0]:HH:mm:ss}");
		        }
		    }
		}
		
		private bool HasValidData()
		{
		    // Check if we have valid data from primary series and at least some historical data
		    bool primaryValid = Close[0] > 0 && Volume[0] > 0;
		    bool listsHaveData = prices1m.Count > 0 && volumes1m.Count > 0;
		    
		    return primaryValid && listsHaveData;
		}

		private void SendHistoricalData()
		{
		    try
		    {
		        Print("Sending historical data to Python...");
		        
		        // Wait for all series to have some data before sending
		        if (BarsArray.Length < 3 || BarsArray[1].Count == 0 || BarsArray[2].Count == 0)
		        {
		            Print("Waiting for all timeframe data to load...");
		            return;
		        }
		        
		        // Get 10 days of data (approximately 1000+ bars for 15min)
		        int historyDays = 10;
		        int barsToSend15m = Math.Min(historyDays * 96, BarsArray[1].Count); // 96 15-min bars per day
		        int barsToSend5m = Math.Min(historyDays * 288, BarsArray[2].Count); // 288 5-min bars per day
		        int barsToSend1m = Math.Min(historyDays * 1440, BarsArray[0].Count); // 1440 1-min bars per day (using primary)
		        
		        var historicalData = new
		        {
		            type = "historical_data",
		            bars_15m = GetHistoricalBars(BarsArray[1], barsToSend15m),
		            bars_5m = GetHistoricalBars(BarsArray[2], barsToSend5m),
		            bars_1m = GetHistoricalBars(BarsArray[0], barsToSend1m), // Using primary series as 1m data
		            timestamp = DateTime.Now.Ticks
		        };
		        
		        string json = SerializeHistoricalData(historicalData);
		        SendJsonMessage(json);
		        
		        Print($"Historical data sent: {historicalData.bars_15m.Count} 15m bars, " +
		              $"{historicalData.bars_5m.Count} 5m bars, {historicalData.bars_1m.Count} 1m bars");
		              
		        historicalDataSent = true;
		    }
		    catch (Exception ex)
		    {
		        Print($"Historical data send error: {ex.Message}");
		        // Don't set historicalDataSent = true on error, so it will retry
		    }
		}
        
        private List<BarData> GetHistoricalBars(Bars bars, int count)
        {
            var barList = new List<BarData>();
            
            if (bars == null || bars.Count == 0)
                return barList;
            
            int startIndex = Math.Max(0, bars.Count - count);
            
            for (int i = startIndex; i < bars.Count; i++)
            {
                barList.Add(new BarData
                {
                    timestamp = bars.GetTime(i).Ticks,
                    open = bars.GetOpen(i),
                    high = bars.GetHigh(i),
                    low = bars.GetLow(i),
                    close = bars.GetClose(i),
                    volume = bars.GetVolume(i)
                });
            }
            
            return barList;
        }
        
        private string SerializeHistoricalData(dynamic data)
        {
            var sb = new StringBuilder();
            sb.Append("{");
            sb.Append($"\"type\":\"historical_data\",");
            sb.Append($"\"bars_15m\":{SerializeBarArray(data.bars_15m)},");
            sb.Append($"\"bars_5m\":{SerializeBarArray(data.bars_5m)},");
            sb.Append($"\"bars_1m\":{SerializeBarArray(data.bars_1m)},");
            sb.Append($"\"timestamp\":{data.timestamp}");
            sb.Append("}");
            return sb.ToString();
        }
        
        private string SerializeBarArray(List<BarData> bars)
        {
            var sb = new StringBuilder();
            sb.Append("[");
            
            for (int i = 0; i < bars.Count; i++)
            {
                if (i > 0) sb.Append(",");
                
                var bar = bars[i];
                sb.Append("{");
                sb.Append($"\"timestamp\":{bar.timestamp},");
                sb.Append($"\"open\":{bar.open.ToString(CultureInfo.InvariantCulture)},");
                sb.Append($"\"high\":{bar.high.ToString(CultureInfo.InvariantCulture)},");
                sb.Append($"\"low\":{bar.low.ToString(CultureInfo.InvariantCulture)},");
                sb.Append($"\"close\":{bar.close.ToString(CultureInfo.InvariantCulture)},");
                sb.Append($"\"volume\":{bar.volume}");
                sb.Append("}");
            }
            
            sb.Append("]");
            return sb.ToString();
        }
        
		private void UpdatePriceData()
		{
		    // Only update when processing the primary series to avoid duplicate updates
		    if (BarsInProgress != 0) 
		        return;
		    
		    // The primary series data (whatever timeframe the strategy is running on)
		    // We'll treat this as our base timeframe
		    UpdateList(prices1m, Close[0], 1000);
		    UpdateList(volumes1m, Volume[0], 1000);
		    
		    // Update from additional series if they have data
		    // BarsArray[1] = 15m series
		    if (BarsArray.Length > 1 && BarsArray[1].Count > 0)
		    {
		        UpdateList(prices15m, Closes[1][0], 100);
		        UpdateList(volumes15m, Volumes[1][0], 100);
		    }
		    
		    // BarsArray[2] = 5m series  
		    if (BarsArray.Length > 2 && BarsArray[2].Count > 0)
		    {
		        UpdateList(prices5m, Closes[2][0], 300);
		        UpdateList(volumes5m, Volumes[2][0], 300);
		    }
		    
		    // Note: BarsArray[3] would be 1m, but we're using the primary series as our minute data
		    // If you want true 1m data separate from primary, you'd access it here as Closes[3][0]
		}
		
		private void LogDataSeriesInfo()
		{
		    Print($"Data Series Info:");
		    Print($"  Primary (BarsArray[0]): {BarsArray[0].Count} bars, Current: {Close[0]:F2}");
		    
		    if (BarsArray.Length > 1)
		        Print($"  15m (BarsArray[1]): {BarsArray[1].Count} bars, Current: {(BarsArray[1].Count > 0 ? Closes[1][0].ToString("F2") : "N/A")}");
		    
		    if (BarsArray.Length > 2)
		        Print($"  5m (BarsArray[2]): {BarsArray[2].Count} bars, Current: {(BarsArray[2].Count > 0 ? Closes[2][0].ToString("F2") : "N/A")}");
		        
		    if (BarsArray.Length > 3)
		        Print($"  1m (BarsArray[3]): {BarsArray[3].Count} bars, Current: {(BarsArray[3].Count > 0 ? Closes[3][0].ToString("F2") : "N/A")}");
		        
		    Print($"  Lists - 1m: {prices1m.Count}, 5m: {prices5m.Count}, 15m: {prices15m.Count}");
		}
        
        private void UpdateList(List<double> list, double value, int maxSize)
        {
            list.Add(value);
            if (list.Count > maxSize)
                list.RemoveAt(0);
        }
        
		private void ConnectToPython()
		{
		    try
		    {
		        Print("Attempting to connect to Python AI system...");
		        
		        dataClient = new TcpClient("localhost", 5556);
		        signalClient = new TcpClient("localhost", 5557);
		        isConnected = true;
		        
		        Print("Connected to Python AI system successfully");
		        
		        // Log initial data series info for debugging
		        LogDataSeriesInfo();
		    }
		    catch (Exception ex)
		    {
		        Print($"Connection failed: {ex.Message}");
		        isConnected = false;
		    }
		}
        
		private void SendDataToPython()
		{
		    if (!isConnected || dataClient?.Connected != true)
		    {
		        Print("Cannot send data - not connected to Python");
		        return;
		    }
		        
		    try
		    {
		        var json = BuildMarketDataJson();
		        
		        if (string.IsNullOrEmpty(json))
		        {
		            Print("Cannot send data - JSON is empty");
		            return;
		        }
		        
		        SendJsonMessage(json);
		        
		        Print($"Data send #{dataSendCount} - Lists: 1m={prices1m.Count}, 5m={prices5m.Count}, 15m={prices15m.Count}");
		    }
		    catch (Exception ex)
		    {
		        Print($"Data send error: {ex.Message}");
		        isConnected = false;
		    }
		}
        
        private void SendJsonMessage(string json)
        {
            byte[] jsonBytes = Encoding.UTF8.GetBytes(json);
            byte[] header = BitConverter.GetBytes(jsonBytes.Length);
            
            var stream = dataClient.GetStream();
            stream.Write(header, 0, 4);
            stream.Write(jsonBytes, 0, jsonBytes.Length);
        }
        
		private string BuildMarketDataJson()
		{
		    try
		    {
		        var sb = new StringBuilder();
		        sb.Append("{");
		
		        sb.Append($"\"type\":\"live_data\",");
		        sb.Append($"\"price_1m\":{SerializeDoubleArray(prices1m)},");
		        sb.Append($"\"price_5m\":{SerializeDoubleArray(prices5m)},");
		        sb.Append($"\"price_15m\":{SerializeDoubleArray(prices15m)},");
		        sb.Append($"\"volume_1m\":{SerializeDoubleArray(volumes1m)},");
		        sb.Append($"\"volume_5m\":{SerializeDoubleArray(volumes5m)},");
		        sb.Append($"\"volume_15m\":{SerializeDoubleArray(volumes15m)},");
		
		        double currentBalance = Account.Get(AccountItem.CashValue, Currency.UsDollar);
		        double currentBuyingPower = Account.Get(AccountItem.BuyingPower, Currency.UsDollar);
		        double totalPnL = Account.Get(AccountItem.RealizedProfitLoss, Currency.UsDollar);
		        double dailyPnL = sessionStartSet ? (totalPnL - sessionStartPnL) : 0;
		        double netLiquidation = Account.Get(AccountItem.NetLiquidation, Currency.UsDollar);
		        double marginUsed = Account.Get(AccountItem.InitialMargin, Currency.UsDollar);
		        double availableMargin = Math.Max(0, currentBuyingPower - marginUsed);
		
		        if (currentBalance <= 0) currentBalance = 25000;
		        if (currentBuyingPower <= 0) currentBuyingPower = currentBalance;
		        if (netLiquidation <= 0) netLiquidation = currentBalance;
		
		        sb.Append($"\"account_balance\":{currentBalance.ToString(CultureInfo.InvariantCulture)},");
		        sb.Append($"\"buying_power\":{currentBuyingPower.ToString(CultureInfo.InvariantCulture)},");
		        sb.Append($"\"daily_pnl\":{dailyPnL.ToString(CultureInfo.InvariantCulture)},");
		        sb.Append($"\"net_liquidation\":{netLiquidation.ToString(CultureInfo.InvariantCulture)},");
		        sb.Append($"\"margin_used\":{marginUsed.ToString(CultureInfo.InvariantCulture)},");
		        sb.Append($"\"available_margin\":{availableMargin.ToString(CultureInfo.InvariantCulture)},");
		        sb.Append($"\"open_positions\":{Position.Quantity},");
		        sb.Append($"\"current_price\":{Close[0].ToString(CultureInfo.InvariantCulture)},");
		        long unixSeconds = new DateTimeOffset(DateTime.UtcNow).ToUnixTimeSeconds();
		        sb.Append($"\"timestamp\":{unixSeconds}");
		
		        sb.Append("}");
		        return sb.ToString();
		    }
		    catch (Exception ex)
		    {
		        Print($"Error building market data JSON: {ex.Message}");
		        return string.Empty;
		    }
		}
        
        private string SerializeDoubleArray(List<double> array)
        {
            if (array.Count == 0) return "[]";
            
            var sb = new StringBuilder();
            sb.Append("[");
            
            for (int i = 0; i < array.Count; i++)
            {
                if (i > 0) sb.Append(",");
                sb.Append(array[i].ToString(CultureInfo.InvariantCulture));
            }
            
            sb.Append("]");
            return sb.ToString();
        }
        
        private void StartSignalReceiver()
        {
            if (isRunning) return;
            
            isRunning = true;
            signalThread = new Thread(ReceiveSignals) { IsBackground = true };
            signalThread.Start();
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
                    
                    byte[] header = new byte[4];
                    int bytesRead = signalClient.GetStream().Read(header, 0, 4);
                    if (bytesRead != 4) continue;
                    
                    int messageLength = BitConverter.ToInt32(header, 0);
                    if (messageLength <= 0 || messageLength > 10000) continue;
                    
                    byte[] messageBytes = new byte[messageLength];
                    bytesRead = signalClient.GetStream().Read(messageBytes, 0, messageLength);
                    if (bytesRead != messageLength) continue;
                    
                    string json = Encoding.UTF8.GetString(messageBytes);
                    ProcessSignal(json);
                }
                catch (Exception ex)
                {
                    Print($"Signal receive error: {ex.Message}");
                    Thread.Sleep(2000);
                }
            }
        }
        
        private void ProcessSignal(string json)
        {
            try
            {
                var signal = ParseSignalJson(json);
                
                int action = signal.Item1;
                double confidence = signal.Item2;
                int size = signal.Item3;
                bool useStop = signal.Item4;
                double stopPrice = signal.Item5;
                bool useTarget = signal.Item6;
                double targetPrice = signal.Item7;
                
                string entryName = $"AI_{DateTime.Now:HHmmss}";
                
                switch (action)
                {
                    case 1: // Buy
                        if (Position.MarketPosition == MarketPosition.Short)
                            ExitShort(entryName + "_Cover");
                        
                        if (size > 0)
                            EnterLong(size, entryName);
                        break;
                        
                    case 2: // Sell
                        if (Position.MarketPosition == MarketPosition.Long)
                            ExitLong(entryName + "_Exit");
                        
                        if (size > 0)
                            EnterShort(size, entryName);
                        break;
                }
                
                if (useStop && stopPrice > 0)
                    SetStopLoss(entryName, CalculationMode.Price, stopPrice, false);
                    
                if (useTarget && targetPrice > 0)
                    SetProfitTarget(entryName, CalculationMode.Price, targetPrice);
                
                Print($"AI Signal: {(action == 1 ? "BUY" : "SELL")} {size} contracts (Conf: {confidence:P0})");
                
            }
            catch (Exception ex)
            {
                Print($"Signal processing error: {ex.Message}");
            }
        }
        
        private Tuple<int, double, int, bool, double, bool, double> ParseSignalJson(string json)
        {
            int action = 0;
            double confidence = 0.0;
            int size = 1;
            bool useStop = false;
            double stopPrice = 0.0;
            bool useTarget = false;
            double targetPrice = 0.0;
            
            try
            {
                action = ExtractIntValue(json, "action");
                confidence = ExtractDoubleValue(json, "confidence");
                size = ExtractIntValue(json, "position_size");
                useStop = ExtractBoolValue(json, "use_stop");
                stopPrice = ExtractDoubleValue(json, "stop_price");
                useTarget = ExtractBoolValue(json, "use_target");
                targetPrice = ExtractDoubleValue(json, "target_price");
            }
            catch (Exception ex)
            {
                Print($"JSON parsing error: {ex.Message}");
            }
            
            return new Tuple<int, double, int, bool, double, bool, double>(
                action, confidence, size, useStop, stopPrice, useTarget, targetPrice);
        }
        
        private int ExtractIntValue(string json, string key)
        {
            string pattern = $"\"{key}\"";
            int keyIndex = json.IndexOf(pattern);
            if (keyIndex == -1) return 0;
            
            int colonIndex = json.IndexOf(":", keyIndex);
            if (colonIndex == -1) return 0;
            
            int startIndex = colonIndex + 1;
            while (startIndex < json.Length && (json[startIndex] == ' ' || json[startIndex] == '\t'))
                startIndex++;
            
            int endIndex = startIndex;
            while (endIndex < json.Length && char.IsDigit(json[endIndex]))
                endIndex++;
            
            if (endIndex > startIndex && int.TryParse(json.Substring(startIndex, endIndex - startIndex), out int result))
                return result;
            
            return 0;
        }
        
        private double ExtractDoubleValue(string json, string key)
        {
            string pattern = $"\"{key}\"";
            int keyIndex = json.IndexOf(pattern);
            if (keyIndex == -1) return 0.0;
            
            int colonIndex = json.IndexOf(":", keyIndex);
            if (colonIndex == -1) return 0.0;
            
            int startIndex = colonIndex + 1;
            while (startIndex < json.Length && (json[startIndex] == ' ' || json[startIndex] == '\t'))
                startIndex++;
            
            int endIndex = startIndex;
            while (endIndex < json.Length && (char.IsDigit(json[endIndex]) || json[endIndex] == '.' || json[endIndex] == '-'))
                endIndex++;
            
            if (endIndex > startIndex && double.TryParse(json.Substring(startIndex, endIndex - startIndex), NumberStyles.Float, CultureInfo.InvariantCulture, out double result))
                return result;
            
            return 0.0;
        }
        
        private bool ExtractBoolValue(string json, string key)
        {
            string pattern = $"\"{key}\"";
            int keyIndex = json.IndexOf(pattern);
            if (keyIndex == -1) return false;
            
            int colonIndex = json.IndexOf(":", keyIndex);
            if (colonIndex == -1) return false;
            
            int trueIndex = json.IndexOf("true", colonIndex);
            int falseIndex = json.IndexOf("false", colonIndex);
            
            if (trueIndex != -1 && (falseIndex == -1 || trueIndex < falseIndex))
                return true;
            
            return false;
        }
        
        protected override void OnExecutionUpdate(Execution execution, string executionId, 
                                                double price, int quantity, MarketPosition marketPosition, 
                                                string orderId, DateTime time)
        {
            if (execution.Order?.Name?.Contains("AI_") == true && marketPosition == MarketPosition.Flat)
            {
                SendTradeCompletion(execution, price);
            }
        }
        
        private void SendTradeCompletion(Execution execution, double exitPrice)
        {
            if (!isConnected || dataClient?.Connected != true)
                return;
                
            try
            {
                double pnl = 0.0;
                double entryPrice = 0.0;
                int quantity = 0;
                DateTime entryTime = DateTime.Now;
                
                if (SystemPerformance.AllTrades.Count > 0)
                {
                    var lastTrade = SystemPerformance.AllTrades[SystemPerformance.AllTrades.Count - 1];
                    pnl = lastTrade.ProfitCurrency;
                    entryPrice = lastTrade.Entry.Price;
                    quantity = lastTrade.Quantity;
                    entryTime = lastTrade.Entry.Time;
                }
                
                string exitReason = execution.Order.Name.Contains("Stop") ? "stop_hit" :
                                   execution.Order.Name.Contains("Target") ? "target_hit" : "ai_exit";
                
                // Enhanced trade completion data for advanced risk analysis
                double currentBalance = Account.Get(AccountItem.CashValue, Currency.UsDollar);
                double netLiquidation = Account.Get(AccountItem.NetLiquidation, Currency.UsDollar);
                double marginUsed = Account.Get(AccountItem.InitialMargin, Currency.UsDollar);
                double totalPnL = Account.Get(AccountItem.RealizedProfitLoss, Currency.UsDollar);
                double dailyPnL = sessionStartSet ? (totalPnL - sessionStartPnL) : 0;
                
                var json = $"{{" +
                          $"\"type\":\"trade_completion\"," +
                          $"\"final_pnl\":{pnl.ToString(CultureInfo.InvariantCulture)}," +
                          $"\"exit_price\":{exitPrice.ToString(CultureInfo.InvariantCulture)}," +
                          $"\"entry_price\":{entryPrice.ToString(CultureInfo.InvariantCulture)}," +
                          $"\"quantity\":{quantity}," +
                          $"\"exit_reason\":\"{exitReason}\"," +
                          $"\"entry_time\":{entryTime.Ticks}," +
                          $"\"exit_time\":{DateTime.Now.Ticks}," +
                          $"\"account_balance\":{currentBalance.ToString(CultureInfo.InvariantCulture)}," +
                          $"\"net_liquidation\":{netLiquidation.ToString(CultureInfo.InvariantCulture)}," +
                          $"\"margin_used\":{marginUsed.ToString(CultureInfo.InvariantCulture)}," +
                          $"\"daily_pnl\":{dailyPnL.ToString(CultureInfo.InvariantCulture)}," +
                          $"\"timestamp\":{DateTime.Now.Ticks}" +
                          $"}}";
                
                SendJsonMessage(json);
                Print($"Trade completed: P&L ${pnl:F2} ({exitReason}) - Balance: ${currentBalance:F0}");
            }
            catch (Exception ex)
            {
                Print($"Completion send error: {ex.Message}");
            }
        }
        
        private void Cleanup()
        {
            isRunning = false;
            isConnected = false;
            
            try
            {
                dataClient?.Close();
                signalClient?.Close();
                signalThread?.Join(2000);
            }
            catch (Exception ex)
            {
                Print($"Cleanup error: {ex.Message}");
            }
        }
    }
    
    public class BarData
    {
        public long timestamp;
        public double open;
        public double high;
        public double low;
        public double close;
        public long volume;
    }
}