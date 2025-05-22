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
        private ATR atr;
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
        [Range(0.5, 10)]
        [Display(Name = "ATR Stop Multiple", Description = "ATR multiplier for stop loss", Order = 2, GroupName = "Risk Management")]
        public double AtrStopMultiple { get; set; } = 2.0;
        
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
                        // Don't log data loaded - too noisy
                        break;
                        
                    case State.Historical:
                        // Don't log Historical state - too noisy
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
                        break;
                        
                    case State.Terminated:
                        // Don't log termination unless it was active
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
            Description = "Reinforcement Learning Trading Strategy";
            Calculate = Calculate.OnBarClose;
            
            // Chart configuration
            AddPlot(Brushes.Blue, "LWPE");
            IsOverlay = false;
            
            // Entry configuration
            EntriesPerDirection = 10;
            EntryHandling = EntryHandling.AllEntries;
            
            // Reset state flags for new instance
            isTerminated = false;
            socketsStarted = false;
            running = false;
            
            // Don't log initialization - too noisy
        }
        
        private void InitializeIndicators()
        {
            atr = ATR(14);
            lwpeSeries = new Series<double>(this);
            AddChartIndicator(atr);
        }
        
        private void InitializeSockets()
        {
            if (socketsStarted)
            {
                return; // No logging for already initialized
            }
            
            try
            {
                ConnectToSockets();
                StartBackgroundThreads();
                socketsStarted = true;
                Print($"RLTrader connected to Python service - Ready for signals");
            }
            catch (Exception ex)
            {
                Print($"Socket connection failed: {ex.Message}");
                socketsStarted = false;
            }
        }
        
        private void Cleanup()
        {
            // Use a local variable to ensure thread safety
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
                return; // Already cleaned up
            }
            
            // Only print shutdown for the active trading instance
            if (socketsStarted)
            {
                Print($"RLTrader shutting down");
            }
            
            running = false;
            
            // Wait for threads to complete
            if (recvThread?.IsAlive == true)
            {
                recvThread.Join(1000);
            }
            
            if (lwpeThread?.IsAlive == true)
            {
                lwpeThread.Join(1000);
            }
            
            // Close all connections
            DisposeSockets();
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
            
            // No connection logging - keep it clean
        }
        
        private void StartBackgroundThreads()
        {
            running = true;
            
            recvThread = new Thread(SignalReceiveLoop) 
            { 
                IsBackground = true, 
                Name = "SignalReceiver" 
            };
            
            lwpeThread = new Thread(LwpeReceiveLoop) 
            { 
                IsBackground = true, 
                Name = "LwpeReceiver" 
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
            UpdateLwpePlot();
            SendFeatureVector();
            
            if (!IsReadyForTrading())
                return;
                
            ProcessLatestSignal();
            SendPositionUpdate();
        }
        
        private void UpdateLwpePlot()
        {
            lock (lwpeLock)
            {
                Values[0][0] = currentLWPE;
            }
        }
        
        private bool IsReadyForTrading()
        {
            return CurrentBar >= 20 && State == State.Realtime;
        }
        
		private void ProcessLatestSignal()
		{
		    var signal = GetLatestSignal();
		    
		    if (signal == null)
		        return;
		        
		    // ADD: Check for duplicates
		    if (IsSignalAlreadyProcessed(signal))
		        return;
		        
		    // ADD: Validate timestamp  
		    if (!IsSignalTimestampValid(signal))
		        return;
		        
		    ExecuteSignal(signal);
		    UpdateLastSignalTime(signal);
		}
		
		private SignalData GetLatestSignal()
		{
		    lock (signalLock)
		    {
		        return latestSignal;
		    }
		}
		
		private bool IsSignalTimestampValid(SignalData signal)
		{
		    try
		    {
		        // Use platform time instead of DateTime.Now
		        var platformNow = NinjaTrader.Core.Globals.Now;
		        var signalTime = Time[0]; // Use current bar timestamp
		        
		        // For external signals, you could use a simple integer timestamp
		        // and convert it to platform time for comparison
		        var signalDateTime = new DateTime(1970, 1, 1, 0, 0, 0, DateTimeKind.Utc).AddSeconds(signal.Timestamp);
		        
		        // Compare against bar time rather than system time
		        var timeDiff = (Time[0] - signalDateTime).TotalSeconds;
		        
		        // More lenient validation for external signals
		        return Math.Abs(timeDiff) <= 300; // 5 minute window
		    }
		    catch (Exception ex)
		    {
		        Print($"Timestamp validation error: {ex.Message}");
		        return false;
		    }
		}
		
		private bool IsSignalAlreadyProcessed(SignalData signal)
		{
		    // Simple approach: track last processed signal ID or timestamp
		    if (signal.Timestamp <= lastProcessedTimestamp)
		        return true;
		        
		    return false;
		}
		
		private void ExecuteSignal(SignalData signal)
		{
		    // Add position size validation
		    if (signal.Size <= 0 || signal.Size > 20) // Reasonable size limits
		    {
		        Print($"Invalid signal size: {signal.Size}");
		        return;
		    }
		
		    switch (signal.Action)
		    {
		        case 1: // BUY
		            Print($"Executing LONG: size={signal.Size}, confidence={signal.Confidence:F3}");
		            EnterLong(signal.Size, "RL_LONG");  // Add signal name for tracking
		            break;
		            
		        case 2: // SELL
		            Print($"Executing SHORT: size={signal.Size}, confidence={signal.Confidence:F3}");
		            EnterShort(signal.Size, "RL_SHORT");  // Add signal name for tracking
		            break;
		            
		        default: // HOLD
		            Print($"HOLD signal: confidence={signal.Confidence:F3}");
		            break;
		    }
		}
        
		private void UpdateLastSignalTime(SignalData signal)
		{
		    // Store the signal timestamp for duplicate prevention
		    lastProcessedTimestamp = signal.Timestamp;
		    
		    // Use Time[0] for internal tracking
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
            if (tickSock?.Connected == true)
            {
                byte[] data = Encoding.UTF8.GetBytes(tickData);
                tickSock.GetStream().Write(data, 0, data.Length);
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
                Print($"Feature transmission error: {ex.Message}");
            }
        }
        
        private FeatureVector CalculateFeatures()
        {
            double volMean = SMA(Volume, 20)[0];
            double volStd = StdDev(Volume, 20)[0];
            double normalizedVolume = volStd != 0 ? (Volume[0] - volMean) / volStd : 0;
            
            double lwpeValue;
            lock (lwpeLock)
            {
                lwpeValue = currentLWPE;
            }
            
            return new FeatureVector
            {
                Close = Close[0],
                NormalizedVolume = normalizedVolume,
                ATR = atr[0],
                LWPE = lwpeValue,
                IsLive = State == State.Realtime
            };
        }
        
        private string CreateFeaturePayload(FeatureVector features)
        {
            return string.Format(CultureInfo.InvariantCulture,
                @"{{
                    ""features"":[{0:F6},{1:F6},{2:F6},{3:F6}],
                    ""live"":{4}
                }}",
                features.Close, 
                features.NormalizedVolume, 
                features.ATR, 
                features.LWPE,
                features.IsLive ? 1 : 0);
        }
        
        #endregion
        
        #region Position Management
        
        private void SendPositionUpdate()
        {
            try
            {
                string positionJson = $"{{\"position\":{Position.Quantity}}}";
                TransmitData(sendSock, positionJson);
            }
            catch (Exception ex)
            {
                Print($"Position update error: {ex.Message}");
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
                    byte[] messageBuffer = new byte[messageLength];
                    
                    if (!ReadExactBytes(stream, messageBuffer, messageLength))
                        continue;
                        
                    ProcessReceivedSignal(messageBuffer);
                }
                catch (Exception ex)
                {
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
		            SignalId = signalDict.ContainsKey("signal_id") ? Convert.ToInt32(signalDict["signal_id"]) : 0  // ADD THIS
		        };
		        
		        Print($"Signal received → action={latestSignal.Action}, " +
		              $"confidence={latestSignal.Confidence:F3}, " +
		              $"size={latestSignal.Size}, " +
		              $"timestamp={latestSignal.Timestamp}, " +
		              $"id={latestSignal.SignalId}");  // ADD ID to logging
		    }
		}
        
        private void ProcessLwpeData(byte[] buffer, int bytesRead)
        {
            string valueString = Encoding.UTF8.GetString(buffer, 0, bytesRead).Trim();
            
            if (double.TryParse(valueString, NumberStyles.Any, CultureInfo.InvariantCulture, out double lwpeValue))
            {
                lock (lwpeLock)
                {
                    currentLWPE = lwpeValue;
                }
            }
        }
        
        private void TransmitData(TcpClient client, string data)
        {
            if (client?.Connected != true)
                return;
                
            byte[] payload = Encoding.UTF8.GetBytes(data);
            byte[] header = BitConverter.GetBytes(payload.Length);
            
            var stream = client.GetStream();
            stream.Write(header, 0, 4);
            stream.Write(payload, 0, payload.Length);
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
            public double ATR { get; set; }
            public double LWPE { get; set; }
            public bool IsLive { get; set; }
        }
        
        #endregion
    }
}