using System;
using System.IO;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using System.ComponentModel;
using System.ComponentModel.DataAnnotations;
using NinjaTrader.Cbi;
using NinjaTrader.Data;
using NinjaTrader.NinjaScript;
using NinjaTrader.NinjaScript.Indicators;
using NinjaTrader.NinjaScript.Strategies;
using System.Collections.Generic;
using System.Web.Script.Serialization;

 

namespace NinjaTrader.NinjaScript.Strategies
{
    public class RLTrader : Strategy
    {
        // Indicators & parameters
        private EMA fastEma, slowEma;
        private RSI rsi;
        private ATR atr;

        [NinjaScriptProperty, Display(Name = "Fast EMA", Order = 1, GroupName = "Parameters")]
        [Range(1, int.MaxValue)] public int FastPeriod { get; set; } = 9;

        [NinjaScriptProperty, Display(Name = "Slow EMA", Order = 2, GroupName = "Parameters")]
        [Range(1, int.MaxValue)] public int SlowPeriod { get; set; } = 21;

        [NinjaScriptProperty, Display(Name = "RSI Period", Order = 3, GroupName = "Parameters")]
        [Range(1, int.MaxValue)] public int RsiPeriod { get; set; } = 14;

        [NinjaScriptProperty, Display(Name = "Risk %", Order = 4, GroupName = "Risk")]
        [Range(0.001, 0.1)] public double RiskPercent { get; set; } = 0.01;

        [NinjaScriptProperty, Display(Name = "ATR × Stop", Order = 5, GroupName = "Risk")]
        [Range(0.5, 10)] public double AtrStopMultiple { get; set; } = 2.0;

        // Sockets
        private TcpClient sendSock; // Python (features)
        private TcpClient recvSock; // Python (signals)
        private Thread    recvThread;
        private volatile bool  running;

        private readonly object sigLock = new object();
        private SignalData latestSignal;
        private DateTime   lastSignalTime = DateTime.MinValue;
		private bool socketsStarted = false;

        // Small POCO for incoming signal
        private class SignalData
        {
            public int    Action;
            public int    Size;
            public double Confidence;
            public long   Timestamp;
        }

        private int BarsRequiredToTrade
            => Math.Max(FastPeriod, Math.Max(SlowPeriod, Math.Max(RsiPeriod, 14)));

        protected override void OnStateChange()
        {
            if (State == State.SetDefaults)
            {
                Name          = "RLTrader";
                Description   = "Reinforcement Learning Trader";
                Calculate     = Calculate.OnBarClose;
                StartBehavior = StartBehavior.WaitUntilFlat;
                IsOverlay     = false;
            }
            else if (State == State.DataLoaded)
            {
                fastEma = EMA(FastPeriod);
                slowEma = EMA(SlowPeriod);
                rsi     = RSI(RsiPeriod, 3);
                atr     = ATR(14);
				
				AddChartIndicator(fastEma); 
			    AddChartIndicator(slowEma);
			    AddChartIndicator(rsi);
			    AddChartIndicator(atr); 
            }
			else if (State == State.Historical && !socketsStarted) 
			{
			    StartSockets();
			    socketsStarted = true;
			}
			else if (State == State.Realtime && !socketsStarted)
			{
			    StartSockets();
			    socketsStarted = true;
			}
            else if (State == State.Terminated)
            {
                running = false;
                recvThread?.Join(500);
                sendSock?.Close();
                recvSock?.Close();
            }
        }

        private void StartSockets()
        {
            try
            {
                sendSock = new TcpClient("localhost", 5556);
                recvSock = new TcpClient("localhost", 5557);
                running  = true;

                recvThread = new Thread(RecvLoop) { IsBackground = true };
                recvThread.Start();

                Print("Sockets connected");
            }
            catch (Exception ex)
            {
                Print("Socket error – " + ex.Message);
                sendSock = recvSock = null; // Strategy will still run, just no ML
            }
        }

		private void RecvLoop()
		{
		    var stream = recvSock.GetStream();
		    var lenBuf = new byte[4];
		    var ser    = new JavaScriptSerializer();
		
		    while (running)
		    {
		        try
		        {
		            if (stream.Read(lenBuf, 0, 4) != 4)
		                break;
		
		            int n = BitConverter.ToInt32(lenBuf, 0);
		            var buf = new byte[n];
		            int got = 0;
		            while (got < n) got += stream.Read(buf, got, n - got);
		
		            var dict = ser.Deserialize<Dictionary<string, object>>(
		                           Encoding.UTF8.GetString(buf));
		
		            var sig = new SignalData
		            {
		                Action     = Convert.ToInt32 (dict["action"]),
		                Size       = Convert.ToInt32 (dict["size"]),
		                Confidence = Convert.ToDouble(dict["confidence"]),
		                Timestamp  = Convert.ToInt64 (dict["timestamp"])
		            };
		
		            lock (sigLock) latestSignal = sig;
		        }
		        catch (Exception ex)
		        {
		            Print("Recv error: " + ex.Message);
		            break;
		        }
		    }
		}

		protected override void OnBarUpdate()
		{
		    if (CurrentBar < BarsRequiredToTrade)
		        return;
		
		    ActOnSignal();
		    SendFeatureVector();
		}

		private void SendFeatureVector()
		{
		    if (sendSock == null || !sendSock.Connected) return;
		
		    string payload =
		        $"{{\"features\":[{Close[0]:F6},{fastEma[0]:F6},{slowEma[0]:F6}," +
		        $"{rsi[0]:F6},{atr[0]:F6},{Volume[0]:F0}],\"live\":{(State == State.Realtime ? 1 : 0)}}}";
		
		    try
		    {
		        var ns  = sendSock.GetStream();
		        byte[] b = Encoding.UTF8.GetBytes(payload);
		        ns.Write(BitConverter.GetBytes(b.Length), 0, 4);
		        ns.Write(b, 0, b.Length);
		    }
		    catch { }
		}

		private void ActOnSignal()
		{
		    SignalData sig;
		    lock (sigLock) sig = latestSignal;
		    if (sig == null) return;
		
		    if (DateTimeOffset.FromUnixTimeSeconds(sig.Timestamp).UtcDateTime <= lastSignalTime)
		        return;
		    lastSignalTime = DateTimeOffset.FromUnixTimeSeconds(sig.Timestamp).UtcDateTime;
		
		    int baseQty = Math.Max(1, sig.Size);
		    int tgtQty  = sig.Action == 0 ?  baseQty :
		                  sig.Action == 2 ? -baseQty : 0;
		
		    int diff = tgtQty - Position.Quantity;
		    if (diff == 0) return;
		
		    if (Position.Quantity != 0 && Math.Sign(diff) != Math.Sign(Position.Quantity))
		    {
		        ExitLong(); ExitShort();
		        diff = tgtQty;
		    }
		
		    if (diff > 0)
		    {
		        EnterLong (diff,  "RL_Add");
		    }
		    else if (diff < 0)
		    {
		        EnterShort(-diff, "RL_Add");
		    }
		
		    Print($"Target={tgtQty}  Curr={Position.Quantity}  Adj={diff}  conf={sig.Confidence:P0}");
		}
    }
}