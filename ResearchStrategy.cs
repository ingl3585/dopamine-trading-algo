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
        private TcpClient dataClient;
        private TcpClient signalClient;
        private Thread signalThread;
        private bool isConnected;
        private bool isRunning;
        
        private List<double> prices1m = new List<double>();
        private List<double> prices5m = new List<double>();
        private List<double> prices15m = new List<double>();
        private List<double> volumes1m = new List<double>();
        private List<double> volumes5m = new List<double>();
        private List<double> volumes15m = new List<double>();
        
        protected override void OnStateChange()
        {
            switch (State)
            {
                case State.SetDefaults:
                    Description = "AI Trading Strategy";
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
                    break;
                    
                case State.Terminated:
                    Cleanup();
                    break;
            }
        }
        
        protected override void OnBarUpdate()
        {
            if (State != State.Realtime || !IsFirstTickOfBar)
                return;
                
            UpdatePriceData();
            
            if (BarsInProgress == 0 && isConnected)
                SendDataToPython();
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
                var data = new
                {
                    price_1m = prices1m,
                    price_5m = prices5m,
                    price_15m = prices15m,
                    volume_1m = volumes1m,
                    volume_5m = volumes5m,
                    volume_15m = volumes15m,
                    account_balance = Account.Get(AccountItem.CashValue, Currency.UsDollar),
                    buying_power = Account.Get(AccountItem.BuyingPower, Currency.UsDollar),
                    daily_pnl = Account.Get(AccountItem.RealizedProfitLoss, Currency.UsDollar),
                    timestamp = DateTime.Now.Ticks
                };
                
                string json = Newtonsoft.Json.JsonConvert.SerializeObject(data);
                byte[] jsonBytes = Encoding.UTF8.GetBytes(json);
                byte[] header = BitConverter.GetBytes(jsonBytes.Length);
                
                var stream = dataClient.GetStream();
                stream.Write(header, 0, 4);
                stream.Write(jsonBytes, 0, jsonBytes.Length);
            }
            catch (Exception ex)
            {
                Print($"Data send error: {ex.Message}");
                isConnected = false;
            }
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
                dynamic signal = Newtonsoft.Json.JsonConvert.DeserializeObject(json);
                
                int action = signal.action;
                double confidence = signal.confidence;
                int size = (int)signal.position_size;
                
                string entryName = $"AI_{DateTime.Now:HHmmss}";
                
                switch (action)
                {
                    case 1: // Buy
                        if (Position.MarketPosition == MarketPosition.Short)
                            ExitShort();
                        EnterLong(size, entryName);
                        break;
                        
                    case 2: // Sell
                        if (Position.MarketPosition == MarketPosition.Long)
                            ExitLong();
                        EnterShort(size, entryName);
                        break;
                }
                
                // Set stops and targets if provided
                if (signal.use_stop && signal.stop_price > 0)
                    SetStopLoss(entryName, CalculationMode.Price, (double)signal.stop_price, false);
                    
                if (signal.use_target && signal.target_price > 0)
                    SetProfitTarget(entryName, CalculationMode.Price, (double)signal.target_price);
                
                Print($"Signal executed: {(action == 1 ? "BUY" : "SELL")} {size} contracts");
            }
            catch (Exception ex)
            {
                Print($"Signal processing error: {ex.Message}");
            }
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
                
                var completion = new
                {
                    type = "trade_completion",
                    final_pnl = pnl,
                    exit_price = exitPrice,
                    exit_reason = execution.Order.Name.Contains("Stop") ? "stop_hit" : 
                                 execution.Order.Name.Contains("Target") ? "target_hit" : "ai_exit",
                    timestamp = DateTime.Now.Ticks
                };
                
                string json = Newtonsoft.Json.JsonConvert.SerializeObject(completion);
                byte[] jsonBytes = Encoding.UTF8.GetBytes(json);
                byte[] header = BitConverter.GetBytes(jsonBytes.Length);
                
                var stream = dataClient.GetStream();
                stream.Write(header, 0, 4);
                stream.Write(jsonBytes, 0, jsonBytes.Length);
                
                Print($"Trade completion sent: P&L ${pnl:F2}");
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
}