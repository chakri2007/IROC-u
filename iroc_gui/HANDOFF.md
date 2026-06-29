# HANDOFF — GCS override commands (`/gcs/command`)

For the **flight-code team**. The base-station GUI now has a manual **OVERRIDES**
panel (START + emergency commands). When the operator confirms a command, the
GUI backend publishes a single fixed string on a ROS topic. **You own the other
half:** a subscriber in the flight workspace that turns each string into a flight
action using your existing MAVROS / offboard logic.

The GUI side never touches flight code. This topic is the entire integration
surface. Build the subscriber below into your own package — do not edit anything
under `iroc_gui/`.

---

## 1. The contract

| | |
|---|---|
| **Topic** | `/gcs/command` |
| **Type** | `std_msgs/String` |
| **QoS** | **RELIABLE**, **VOLATILE** (not latched), **depth 10** |
| **Payload** | exactly one of the vocabulary strings below — nothing else is ever sent |
| **Direction** | GUI → drone |

> QoS must match exactly. RELIABLE so a safety command can't be silently
> dropped; VOLATILE (not `TRANSIENT_LOCAL`) so a node restart never replays a
> stale ABORT. If your subscriber uses a different reliability, ROS may refuse
> the match and you'll receive **nothing**.

### Vocabulary

| Command | Operator intent | Enabled in GUI when | **You implement** |
|---|---|---|---|
| `START` | Begin autonomous mission: undock → launch → survey | DISARMED **and** on-ground | arm + offboard + takeoff + start survey |
| `ABORT` | Emergency: controlled descent + disarm **in place** | armed / in-air | vertical land at current X/Y, then disarm (**not** a motor-kill) |
| `HOLD` | Halt motion, hold/loiter current position | in-air | set HOLD/LOITER, freeze position |
| `RTL` | Return to launch (base), then loiter | in-air | `AUTO.RTL` or your return-to-base logic, hold at base |
| `ABORT_DOCK` | Break off a docking/precision-landing attempt | in-air | abort descent, climb to safe alt, hold |
| `RECALL` | End sortie gracefully: return to base **and** land | in-air | return to base, then dock/land normally |

**Confirm these mappings back to us** — especially `ABORT` (land-in-place vs.
RTL-then-land) and the `RTL` / `RECALL` distinction. The GUI's enable/disable
rules are built to match this table; if you change the semantics, tell us so the
gating stays honest.

---

## 2. Reference subscriber (copy into your workspace)

Drop this into your flight package (e.g. `iroc_ros_ws/.../gcs_command_bridge.py`),
add it to your `setup.py` entry points, and fill in the six handlers with your
real flight logic. It already matches the QoS and ignores anything off-vocab.

```python
#!/usr/bin/env python3
"""gcs_command_bridge.py — flight-side receiver for GUI override commands."""
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from std_msgs.msg import String

VOCAB = ("START", "ABORT", "HOLD", "RTL", "ABORT_DOCK", "RECALL")


class GcsCommandBridge(Node):
    def __init__(self):
        super().__init__("gcs_command_bridge")
        qos = QoSProfile(depth=10,
                         reliability=ReliabilityPolicy.RELIABLE,
                         durability=DurabilityPolicy.VOLATILE)
        self.create_subscription(String, "/gcs/command", self.on_command, qos)
        self.ack_pub = self.create_publisher(String, "/gcs/command_ack", qos)
        self._last = None
        self.get_logger().info("GCS command bridge up — listening on /gcs/command")

    def on_command(self, msg):
        cmd = msg.data.strip().upper()
        if cmd not in VOCAB:
            self.get_logger().warn(f"ignoring unknown command: {msg.data!r}")
            return
        # ABORT must always win, even mid-handler. Debounce non-emergency repeats:
        if cmd == self._last and cmd != "ABORT":
            return
        self._last = cmd
        self.get_logger().info(f"GCS command: {cmd}")

        handler = {
            "START":      self.do_start,
            "ABORT":      self.do_abort,
            "HOLD":       self.do_hold,
            "RTL":        self.do_rtl,
            "ABORT_DOCK": self.do_abort_dock,
            "RECALL":     self.do_recall,
        }[cmd]
        try:
            handler()
            self.ack(cmd, "ACCEPTED")
        except Exception as e:
            self.get_logger().error(f"{cmd} failed: {e}")
            self.ack(cmd, f"REJECTED:{e}")

    def ack(self, cmd, result):
        m = String(); m.data = f"{cmd}:{result}"
        self.ack_pub.publish(m)

    # ── Fill these in with YOUR flight logic ───────────────────────────────
    def do_start(self):      raise NotImplementedError  # arm + offboard + takeoff + survey
    def do_abort(self):      raise NotImplementedError  # controlled descent + disarm in place
    def do_hold(self):       raise NotImplementedError  # HOLD/LOITER, hold position
    def do_rtl(self):        raise NotImplementedError  # AUTO.RTL / return logic, then loiter
    def do_abort_dock(self): raise NotImplementedError  # break off docking, climb, hold
    def do_recall(self):     raise NotImplementedError  # return to base + land


def main():
    rclpy.init()
    node = GcsCommandBridge()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
```

---

## 3. Your safety responsibilities

The GUI guarantees only that it **published** a valid vocab string. Everything
about *acting safely* is yours:

- **ABORT pre-empts everything.** It must interrupt any in-progress handler.
- **Ignore impossible commands** in your own state machine too (e.g. `START`
  while already armed). The backend also gates these, but never rely on it —
  treat every command as advisory and re-check against real flight state.
- **Debounce** repeats (the reference does this for non-emergency commands).
- **Fail safe:** if a handler can't run, hold — don't do something worse.

---

## 4. The ack channel (recommended — big trust win)

If you publish `/gcs/command_ack` (`std_msgs/String`, same QoS) the GUI flips
from *"ABORT sent — awaiting drone ack"* to *"drone ack: ABORT:ACCEPTED"*. On a
scored, recorded console that's the difference between "we hope it heard us" and
visible proof the drone obeyed. The reference node already does this. Format is
free-form; we display whatever string you send. Suggested: `"<CMD>:ACCEPTED"` /
`"<CMD>:REJECTED:<reason>"`.

---

## 5. Acceptance test (no GUI needed)

```bash
# Terminal 1 — watch what the GUI sends
ros2 topic echo /gcs/command

# Terminal 2 — simulate the GUI (or click the buttons in the real GUI)
ros2 topic pub --once /gcs/command std_msgs/String "{data: HOLD}"

# Terminal 3 — confirm your bridge acks
ros2 topic echo /gcs/command_ack
```

Then run your `gcs_command_bridge` and repeat — you should see the log line and
the ack. Check the QoS matched:

```bash
ros2 topic info /gcs/command --verbose   # publisher (GUI) vs your subscriber
```

If the subscriber shows 0 matched publishers while the GUI is running, it's
almost always a QoS mismatch — set RELIABLE / VOLATILE / depth 10.

---

## 6. ⚠️ Rules check before you wire the in-air overrides

IRoC-U forbids manual *intervention*. An operator-triggered **START** and an
emergency **ABORT** are standard safety controls and almost certainly allowed,
but the mid-flight overrides (`HOLD` / `RTL` / `RECALL` / `ABORT_DOCK`) may count
as intervention and affect scoring. **Confirm against rulebook V3.0 which of
these are competition-legal before enabling them in a scored run.** It's cheap
to leave a handler as a safe no-op until that's settled.
