# rl_agent.py

import threading, time, random
from collections import deque
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F

from policy_network import PolicyRNN


class _Replay:
    """Simple uniform replay (swap for PER later)."""
    def __init__(self, cap: int = 500_000):
        self.buf = deque(maxlen=cap)

    def push(self, trans: Tuple[np.ndarray, int, float, np.ndarray, bool]):
        self.buf.append(trans)

    def sample(self, n: int = 64):
        batch = random.sample(self.buf, n)
        return map(np.array, zip(*batch))

    def __len__(self): return len(self.buf)


class RLTradingAgent:
    """
    • call `act(obs)` every bar/tick to get (action, stopATR, tpATR)
    • call `store(s, a, r, s2, done)` AFTER a trade closes
    Starts a background learner thread automatically.
    """
    def __init__(self, obs_size: int = 15):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = PolicyRNN(obs_size).to(self.device)
        self.target = PolicyRNN(obs_size).to(self.device)
        self.target.load_state_dict(self.policy.state_dict())
        self.opt = torch.optim.Adam(self.policy.parameters(), lr=3e-4)

        self.replay = _Replay()
        self.gamma = 0.99
        self.sync_every = 1_000
        self.step = 0

        threading.Thread(target=self._learn_forever, daemon=True).start()

    # ---------- live decision ------------------------------------------------
    def act(self, obs: np.ndarray):
        """obs shape (obs_size,) → returns (action_idx, stopATR, tpATR)."""
        with torch.no_grad():
            x = torch.tensor(obs, dtype=torch.float32, device=self.device)
            x = x.unsqueeze(0).unsqueeze(0)      # (1,1,obs)
            logits, stop, tp = self.policy(x)
            probs = torch.softmax(logits, -1).cpu().numpy()[0]
            action = int(np.random.choice(3, p=probs))
            return action, float(stop), float(tp)

    # ---------- training signal ---------------------------------------------
    def store(self, s, a, r, s2, done):
        """Push transition AFTER you know reward & done."""
        self.replay.push((s, a, r, s2, done))

    # ---------- background learner loop -------------------------------------
    def _learn_forever(self):
        while True:
            if len(self.replay) < 10_000:
                time.sleep(0.2)
                continue

            s, a, r, s2, d = self.replay.sample(64)
            s = torch.tensor(s, dtype=torch.float32, device=self.device)
            s2 = torch.tensor(s2, dtype=torch.float32, device=self.device)
            a = torch.tensor(a, dtype=torch.int64, device=self.device)
            r = torch.tensor(r, dtype=torch.float32, device=self.device)
            d = torch.tensor(d, dtype=torch.float32, device=self.device)

            q, _, _ = self.policy(s.unsqueeze(1))
            q = q.squeeze(1).gather(1, a.view(-1, 1)).squeeze(1)

            with torch.no_grad():
                q_next, _, _ = self.target(s2.unsqueeze(1))
                q_next = q_next.squeeze(1).max(1)[0]
                target = r + (1 - d) * self.gamma * q_next

            loss = F.mse_loss(q, target)
            self.opt.zero_grad(); loss.backward(); self.opt.step()

            self.step += 1
            if self.step % self.sync_every == 0:
                self.target.load_state_dict(self.policy.state_dict())
