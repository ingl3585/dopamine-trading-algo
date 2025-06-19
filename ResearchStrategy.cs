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
        
        protected override void OnStateChange()
        {
            switch (State)
            {
                case State.SetDefaults:
                    Description = "Adaptive AI Trading Strategy with Historical Bootstrapping";
                    Name = "ResearchStrategy";
                    Calculate = Calculate.OnBarClose;
                    BarsRequiredToTrade = 1;
                    
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
            if (State != State.Realtime || !IsFirstTickOfBar)
                return;
                
            UpdatePriceData();
            
            if (BarsInProgress == 0 && isConnected)
            {
                // Send historical data first, then live data
                if (!historicalDataSent)
                {
                    SendHistoricalData();
                    historicalDataSent = true;
                }
                else
                {
                    SendDataToPython();
                }
            }
        }
        
        private void SendHistoricalData()
        {
            try
            {
                Print("Sending historical data to Python...");
                
                // Get 10 days of data (approximately 1000+ bars for 15min)
                int historyDays = 10;
                int barsToSend = historyDays * 96; // 96 15-min bars per day
                
                var historicalData = new
                {
                    type = "historical_data",
                    bars_15m = GetHistoricalBars(BarsArray[1], Math.Min(barsToSend, BarsArray[1].Count)),
                    bars_5m = GetHistoricalBars(BarsArray[2], Math.Min(barsToSend * 3, BarsArray[2].Count)),
                    bars_1m = GetHistoricalBars(BarsArray[3], Math.Min(barsToSend * 15, BarsArray[3].Count)),
                    timestamp = DateTime.Now.Ticks
                };
                
                string json = SerializeHistoricalData(historicalData);
                SendJsonMessage(json);
                
                Print($"Historical data sent: {historicalData.bars_15m.Count} 15m bars, " +
                      $"{historicalData.bars_5m.Count} 5m bars, {historicalData.bars_1m.Count} 1m bars");
            }
            catch (Exception ex)
            {
                Print($"Historical data send error: {ex.Message}");
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
            switch (BarsInProgress)
            {
                case 1: // 15m
                    UpdateList(prices15m, Closes[1][0], 100);
                    UpdateList(volumes15m, Volumes[1][0], 100);
                    break;
                case 2: // 5m
                    UpdateList(prices5m, Closes[2][0], 300);
                    UpdateList(volumes5m, Volumes[2][0], 300);
                    break;
                case 3: // 1m
                    UpdateList(prices1m, Closes[3][0], 1000);
                    UpdateList(volumes1m, Volumes[3][0], 1000);
                    break;
            }
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
                dataClient = new TcpClient("localhost", 5556);
                signalClient = new TcpClient("localhost", 5557);
                isConnected = true;
                Print("Connected to Python AI system");
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
                return;
                
            try
            {
                var json = BuildMarketDataJson();
                SendJsonMessage(json);
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
            var sb = new StringBuilder();
            sb.Append("{");
            
            sb.Append($"\"type\":\"live_data\",");
            sb.Append($"\"price_1m\":{SerializeDoubleArray(prices1m)},");
            sb.Append($"\"price_5m\":{SerializeDoubleArray(prices5m)},");
            sb.Append($"\"price_15m\":{SerializeDoubleArray(prices15m)},");
            sb.Append($"\"volume_1m\":{SerializeDoubleArray(volumes1m)},");
            sb.Append($"\"volume_5m\":{SerializeDoubleArray(volumes5m)},");
            sb.Append($"\"volume_15m\":{SerializeDoubleArray(volumes15m)},");
            
            // Enhanced account data
            double currentBalance = Account.Get(AccountItem.CashValue, Currency.UsDollar);
            double currentBuyingPower = Account.Get(AccountItem.BuyingPower, Currency.UsDollar);
            double totalPnL = Account.Get(AccountItem.RealizedProfitLoss, Currency.UsDollar);
            double dailyPnL = sessionStartSet ? (totalPnL - sessionStartPnL) : 0;
            double netLiquidation = Account.Get(AccountItem.NetLiquidation, Currency.UsDollar);
            double marginUsed = Account.Get(AccountItem.InitialMargin, Currency.UsDollar);
            double availableMargin = currentBuyingPower - marginUsed;
            
            sb.Append($"\"account_balance\":{currentBalance.ToString(CultureInfo.InvariantCulture)},");
            sb.Append($"\"buying_power\":{currentBuyingPower.ToString(CultureInfo.InvariantCulture)},");
            sb.Append($"\"daily_pnl\":{dailyPnL.ToString(CultureInfo.InvariantCulture)},");
            sb.Append($"\"net_liquidation\":{netLiquidation.ToString(CultureInfo.InvariantCulture)},");
            sb.Append($"\"margin_used\":{marginUsed.ToString(CultureInfo.InvariantCulture)},");
            sb.Append($"\"available_margin\":{availableMargin.ToString(CultureInfo.InvariantCulture)},");
            sb.Append($"\"open_positions\":{Position.Quantity},");
            sb.Append($"\"timestamp\":{DateTime.Now.Ticks}");
            
            sb.Append("}");
            return sb.ToString();
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
                if (SystemPerformance.AllTrades.Count > 0)
                {
                    var lastTrade = SystemPerformance.AllTrades[SystemPerformance.AllTrades.Count - 1];
                    pnl = lastTrade.ProfitCurrency;
                }
                
                string exitReason = execution.Order.Name.Contains("Stop") ? "stop_hit" : 
                                   execution.Order.Name.Contains("Target") ? "target_hit" : "ai_exit";
                
                var json = $"{{" +
                          $"\"type\":\"trade_completion\"," +
                          $"\"final_pnl\":{pnl.ToString(CultureInfo.InvariantCulture)}," +
                          $"\"exit_price\":{exitPrice.ToString(CultureInfo.InvariantCulture)}," +
                          $"\"exit_reason\":\"{exitReason}\"," +
                          $"\"timestamp\":{DateTime.Now.Ticks}" +
                          $"}}";
                
                SendJsonMessage(json);
                Print($"Trade completed: P&L ${pnl:F2} ({exitReason})");
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
            
            Print("Strategy cleanup complete");
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