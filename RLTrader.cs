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
        private ATR atr;
        private double currentLWPE = 0.5;
        private readonly object lwpeLock = new object();
        private TcpClient sendSock, recvSock, tickSock, lwpeSock;
        private Thread recvThread, lwpeThread;
        private volatile bool running;
        private SignalData latestSignal;
        private DateTime lastSignalTime = DateTime.MinValue;
        private int currentTargetPosition = 0;
		private bool socketsStarted = false;
		private readonly object signalLock = new object();
		double lwpeCopy;
		private Series<double> lwpeSeries;
		private int lastSentPos = int.MinValue;

        [NinjaScriptProperty, Range(0.001, 0.1)]
        public double RiskPercent { get; set; } = 0.01;

        [NinjaScriptProperty, Range(0.5, 10)]
        public double AtrStopMultiple { get; set; } = 2.0;

        protected override void OnStateChange()
        {
            if (State == State.SetDefaults)
            {
                Name = "RLTrader";
                Calculate = Calculate.OnBarClose;
			    AddPlot(Brushes.Blue, "LWPE");
			    IsOverlay = false;
				EntriesPerDirection = 10;
				EntryHandling = EntryHandling.AllEntries;
            }
            else if (State == State.DataLoaded)
            {
                atr = ATR(14);
				lwpeSeries = new Series<double>(this);
				
                AddChartIndicator(atr);
            }
			else if (State == State.Historical && !socketsStarted) 
			{
			    StartSockets();
			    socketsStarted = true;
			    Print("Sockets started in Historical state");
			}
			else if (State == State.Realtime && !socketsStarted)
			{
			    StartSockets();
			    socketsStarted = true;
			    Print("Sockets started in Realtime state");
			}
            else if (State == State.Terminated)
            {
                running = false;
                recvThread?.Join(500);
                lwpeThread?.Join(500);
                sendSock?.Close();
                recvSock?.Close();
                tickSock?.Close();
                lwpeSock?.Close();
            }
        }

        private void StartSockets()
        {
            try
            {
                sendSock = new TcpClient("localhost", 5556);
                recvSock = new TcpClient("localhost", 5557);
                tickSock = new TcpClient("localhost", 5558);
                lwpeSock = new TcpClient("localhost", 5559);

                running = true;
                recvThread = new Thread(RecvLoop) { IsBackground = true };
                lwpeThread = new Thread(LwpeLoop) { IsBackground = true };
                recvThread.Start();
                lwpeThread.Start();
                Print("All sockets connected");
            }
            catch (Exception ex)
            {
                Print($"Socket error: {ex.Message}");
            }
        }

        private void RecvLoop()
        {
            var stream = recvSock.GetStream();
            var ser = new JavaScriptSerializer();
            byte[] lenBuf = new byte[4];

            while (running)
            {
                try
                {
                    if (stream.Read(lenBuf, 0, 4) != 4) break;
                    int n = BitConverter.ToInt32(lenBuf, 0);
                    byte[] buf = new byte[n];
                    int read = 0;
                    while (read < n) read += stream.Read(buf, read, n - read);

                    var dict = ser.Deserialize<Dictionary<string, object>>(Encoding.UTF8.GetString(buf));
                    lock (signalLock)
					{
					    latestSignal = new SignalData
	                    {
	                        Action = Convert.ToInt32(dict["action"]),
	                        Size = Convert.ToInt32(dict["size"]),
	                        Confidence = Convert.ToDouble(dict["confidence"]),
	                        Timestamp = Convert.ToInt64(dict["timestamp"])
	                    };
						Print(string.Format("Signal received → action={0}, confidence={1:F3}, size={2}, timestamp={3}",
						    latestSignal.Action,
						    latestSignal.Confidence,
						    latestSignal.Size,
						    latestSignal.Timestamp));
                    }
                }
                catch (Exception ex)
                {
                    Print($"Recv error: {ex.Message}");
                }
            }
        }
		
		protected override void OnBarUpdate()
		{
			// Print($"[BAR] OnBarUpdate fired - CurrentBar={CurrentBar}, State={State}");
			
			SendFeatureVector();
			
		    lock (lwpeLock)
		    {
		        Values[0][0] = currentLWPE;
		    }

		    if (CurrentBar < 20 || State != State.Realtime)
		        return;
		
		    SignalData sigCopy = null;
		    lock (signalLock)
		    {
		        sigCopy = latestSignal;
		    }
		
		    if (sigCopy == null || sigCopy.Timestamp == 0)
		        return;
		
		    // Avoid duplicate execution
		    if (sigCopy.Timestamp == lastSignalTime.Ticks)
		        return;
		
		    lastSignalTime = new DateTime(sigCopy.Timestamp * TimeSpan.TicksPerSecond);
		
		    if (sigCopy.Action == 1)  // BUY
		    {
		        Print($"Long: size={sigCopy.Size} conf={sigCopy.Confidence:F2}");
		        EnterLong(sigCopy.Size);
		    }
		    else if (sigCopy.Action == 2)  // SELL
		    {
		        Print($"Short: size={sigCopy.Size} conf={sigCopy.Confidence:F2}");
		        EnterShort(sigCopy.Size);
		    }
		    else
		    {
		        Print($"Flat/No Action: conf={sigCopy.Confidence:F2}");
		    }
		}

        protected override void OnMarketData(MarketDataEventArgs e)
        {
			// Print($"[TICK] {e.MarketDataType} {e.Price} {e.Volume}");

            if (e.MarketDataType == MarketDataType.Bid ||
                e.MarketDataType == MarketDataType.Ask ||
                e.MarketDataType == MarketDataType.Last)
            {
                long unixMs = (long)(e.Time.ToUniversalTime() - new DateTime(1970, 1, 1)).TotalMilliseconds;
                string tickStr = $"{unixMs},{e.Price},{e.Volume},{e.MarketDataType}\n";
                tickSock?.GetStream().Write(Encoding.UTF8.GetBytes(tickStr), 0, tickStr.Length);
            }
        }
		
		protected override void OnPositionUpdate(Position position, double averagePrice, int quantity, MarketPosition marketPosition)
		{
		    if (quantity == lastSentPos) return;
		
		    lastSentPos = quantity;
		    try
		    {
		        string posJson = $"{{\"position\":{quantity}}}";
		        byte[] posData = Encoding.UTF8.GetBytes(posJson);
		        byte[] posHeader = BitConverter.GetBytes(posData.Length);
		        sendSock?.GetStream().Write(posHeader, 0, 4);
		        sendSock?.GetStream().Write(posData, 0, posData.Length);
		        Print($"[Position Sync] Sent updated position: {quantity}");
		    }
		    catch (Exception ex)
		    {
		        Print($"[Position Sync Error] {ex.Message}");
		    }
		}

        private void LwpeLoop()
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
                        string val = Encoding.UTF8.GetString(buffer, 0, bytesRead).Trim();
                        if (double.TryParse(val, NumberStyles.Any, CultureInfo.InvariantCulture, out double parsed))
                        {
                            lock (lwpeLock)
                            {
                                currentLWPE = parsed;
                            }
                        }
                    }
                }
                catch (Exception ex)
                {
                    Print($"LWPE error: {ex.Message}");
                    Thread.Sleep(1000);
                }
            }
        }

        private void SendFeatureVector()
        {
            if (sendSock?.Connected != true) return;

            double volMean = SMA(Volume, 20)[0];
            double volStd = StdDev(Volume, 20)[0];
            double normVol = volStd != 0 ? (Volume[0] - volMean) / volStd : 0;

            double lwpeCopy;
            lock (lwpeLock)
            {
                lwpeCopy = currentLWPE;
            }

            string payload = string.Format(CultureInfo.InvariantCulture,
                @"{{
                    ""features"":[{0:F6},{1:F6},{2:F6},{3:F6}],
                    ""live"":{4}
                }}",
                Close[0], normVol, atr[0], lwpeCopy,
                State == State.Realtime ? 1 : 0);
			
			// Print($"[feature] Close={Close[0]:F2}, normVol={normVol:F4}, ATR={atr[0]:F4}, LWPE={lwpeCopy:F4}, live={(State == State.Realtime ? 1 : 0)}");

            try
            {
                byte[] data = Encoding.UTF8.GetBytes(payload);
                byte[] header = BitConverter.GetBytes(data.Length);
                sendSock.GetStream().Write(header, 0, 4);
                sendSock.GetStream().Write(data, 0, data.Length);
            }
            catch (Exception ex)
            {
                Print($"Send error: {ex.Message}");
            }
        }

        private class SignalData
        {
            public int Action;
            public int Size;
            public double Confidence;
            public long Timestamp;
        }
    }
}
