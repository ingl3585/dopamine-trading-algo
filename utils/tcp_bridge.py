# utils/tcp_bridge.py

import socket
import struct
import json
import threading
import logging

log = logging.getLogger(__name__)

class TCPBridge:
    def __init__(self, host: str, feat_port: int, sig_port: int):
        self.on_features = lambda *args: None
        self._current_position = 0 

        self._feat_srv = socket.socket()
        self._feat_srv.bind((host, feat_port))
        self._feat_srv.listen(1)

        self._sig_srv = socket.socket()
        self._sig_srv.bind((host, sig_port))
        self._sig_srv.listen(1)

        log.info("Waiting for NinjaTrader")
        self.fsock, _ = self._feat_srv.accept()
        self.ssock, _ = self._sig_srv.accept()
        log.info("NinjaTrader connected")

        threading.Thread(target=self._reader, daemon=True).start()

    def _reader(self):
        stream = self.fsock
        while True:
            try:
                hdr = stream.recv(4, socket.MSG_WAITALL)
                if not hdr:
                    break
                n = struct.unpack('<I', hdr)[0]
                data = stream.recv(n, socket.MSG_WAITALL)
                if len(data) != n:
                    continue
                msg = json.loads(data.decode())
                if "position" in msg:
                    self._current_position = msg["position"]
                elif "features" in msg:
                    feat = msg["features"]
                    live = msg.get("live", 0)
                    self.on_features(feat, live)
            except Exception as e:
                log.warning("Recv error: %s", e)
                break

    def send_signal(self, sig: dict):
        try:
            blob = json.dumps(sig, separators=(',', ':')).encode()
            self.ssock.sendall(struct.pack('<I', len(blob)) + blob)
        except Exception as e:
            log.warning("Send error: %s", e)

    def close(self):
        for s in (self.fsock, self.ssock, self._feat_srv, self._sig_srv):
            try:
                s.close()
            except:
                pass