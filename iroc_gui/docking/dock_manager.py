#!/usr/bin/env python3
"""
dock_manager.py — Anveshan docking orchestrator (runs on the GUI machine).
==========================================================================
One ROS 2 node that replaces tcp_comm.py + tcp_comm_correct.py + drone_status.py.
It is the GUI-side half of the docking system: it talks to the docking-station
RPi (`station_gcs.py`, TCP :55555) and runs the dock/charge state machine.

WHAT IT DOES
  • Opens a TCP client to the station (auto-reconnects).
  • On startup it WAITS `settle_delay_sec` (let every process come up), then —
    if `auto_dock` — docks the drone that's already sitting on the pad.
  • After charging, holds. When the drone FLIES and LANDS again (detected from
    /mavros/extended_state), it auto re-docks + recharges.
  • Accepts manual commands from the GUI on /dock/command.
  • Streams everything (TCP traffic + the charger's live PID output) onto
    /dock/log so the GUI's "terminal" panel can show it, and publishes the
    high-level state on /dock/state (latched).

WIRE PROTOCOL (matches station_gcs.py)
  Frame = 4-byte big-endian length + UTF-8 payload.
  We send: START_DOCKING | UNDOCK | IDLE | START_CHARGING <mah> | STOP_CHARGING | EMERGENCY
  Station replies: "RECEIVED", then "DONE:<text>" / "ERROR:<text>",
                   async "DONE:Charging stopped" when charging finishes,
                   and (new) "LOG:<line>" frames carrying the live PID output.

This is a plain rclpy script (not a colcon package) so it can be launched the
same way as the backend: `python3 dock_manager.py --ros-args -p ...`.
"""

import socket
import threading
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy, QoSHistoryPolicy
from std_msgs.msg import String

# ExtendedState gives us landed_state (ON_GROUND / IN_AIR / TAKEOFF / LANDING).
try:
    from mavros_msgs.msg import ExtendedState
    MAVROS_AVAILABLE = True
except ImportError:
    ExtendedState = None
    MAVROS_AVAILABLE = False


# Station command vocabulary (what station_gcs.py understands).
CMD_DOCK     = "START_DOCKING"
CMD_UNDOCK   = "UNDOCK"
CMD_IDLE     = "IDLE"
CMD_CHARGE   = "START_CHARGING"   # + " <mah>"
CMD_STOP     = "STOP_CHARGING"
CMD_EMERG    = "EMERGENCY"


class DockManager(Node):

    def __init__(self):
        super().__init__("dock_manager")

        # ---- Parameters (set from the launch file later) ----
        self.declare_parameter("gcs_host",           "192.168.0.114")
        self.declare_parameter("gcs_port",           55555)
        self.declare_parameter("settle_delay_sec",   30.0)
        self.declare_parameter("default_charge_mah", 260)
        self.declare_parameter("auto_dock",          True)
        self.declare_parameter("reconnect_sec",      3.0)
        self.declare_parameter("extended_state_topic", "/mavros/extended_state")

        self.host          = self.get_parameter("gcs_host").value
        self.port          = int(self.get_parameter("gcs_port").value)
        self.settle_delay  = float(self.get_parameter("settle_delay_sec").value)
        self.charge_mah    = int(self.get_parameter("default_charge_mah").value)
        self.auto_dock     = bool(self.get_parameter("auto_dock").value)
        self.reconnect_sec = float(self.get_parameter("reconnect_sec").value)

        # ---- State ----
        self.state        = "BOOT"
        self._was_in_air  = False
        self._sock        = None
        self._send_lock   = threading.Lock()
        self._running     = True

        # ---- Publishers ----
        # /dock/state latched so a late-joining backend/GUI gets the current state.
        latched = QoSProfile(depth=1, history=QoSHistoryPolicy.KEEP_LAST,
                             durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
                             reliability=QoSReliabilityPolicy.RELIABLE)
        self.state_pub = self.create_publisher(String, "/dock/state", latched)
        self.log_pub   = self.create_publisher(String, "/dock/log", 50)

        # ---- Subscribers ----
        # Manual commands from the GUI (DOCK / UNDOCK / CHARGE[:mah] / STOP_CHARGING / EMERGENCY / IDLE)
        self.create_subscription(String, "/dock/command", self.on_command, 10)

        if MAVROS_AVAILABLE:
            self.create_subscription(
                ExtendedState,
                self.get_parameter("extended_state_topic").value,
                self.on_extended_state, 10)
        else:
            self._log("mavros_msgs not available — auto re-dock on landing disabled.")

        # ---- Kick off TCP + settle timer ----
        threading.Thread(target=self._tcp_loop, daemon=True).start()
        self._set_state("SETTLING", f"waiting {self.settle_delay:.0f}s for startup to settle")
        threading.Timer(self.settle_delay, self._after_settle).start()

        self.get_logger().info(
            f"DockManager up. station={self.host}:{self.port} "
            f"settle={self.settle_delay:.0f}s charge={self.charge_mah}mAh auto={self.auto_dock}")

    # ======================================================================
    # STATE + LOG helpers
    # ======================================================================

    def _set_state(self, state: str, detail: str = ""):
        self.state = state
        msg = String(); msg.data = state
        self.state_pub.publish(msg)
        line = f"[STATE] {state}" + (f" — {detail}" if detail else "")
        self._log(line)

    def _log(self, text: str):
        """Emit a line to the GUI terminal (/dock/log) and the node log."""
        m = String(); m.data = text
        self.log_pub.publish(m)
        self.get_logger().info(text)

    # ======================================================================
    # TCP client (connect / reconnect / receive)
    # ======================================================================

    def _tcp_loop(self):
        """Keep a live connection to the station; reconnect on drop."""
        while self._running:
            try:
                self._log(f"[TCP] connecting to {self.host}:{self.port} …")
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(10)
                s.connect((self.host, self.port))
                s.settimeout(None)
                self._sock = s
                self._log("[TCP] connected to station")
                self._listen(s)
            except Exception as e:
                self._log(f"[TCP] connection error: {e}")
            finally:
                self._sock = None
            if not self._running:
                break
            self._log(f"[TCP] reconnecting in {self.reconnect_sec:.0f}s …")
            time.sleep(self.reconnect_sec)

    def _recvall(self, sock, n):
        """Read exactly n bytes or return None on EOF."""
        buf = b""
        while len(buf) < n:
            chunk = sock.recv(n - len(buf))
            if not chunk:
                return None
            buf += chunk
        return buf

    def _listen(self, sock):
        while self._running:
            header = self._recvall(sock, 4)
            if header is None:
                self._log("[TCP] station closed the connection")
                break
            length = int.from_bytes(header, "big")
            payload = self._recvall(sock, length)
            if payload is None:
                break
            self._handle_station_message(payload.decode("utf-8", "replace").strip())

    def _send(self, command: str):
        sock = self._sock
        if sock is None:
            self._log(f"[TCP] NOT sent (no connection): {command}")
            return False
        try:
            data = command.encode("utf-8")
            with self._send_lock:
                sock.sendall(len(data).to_bytes(4, "big"))
                sock.sendall(data)
            self._log(f"[TX] {command}")
            return True
        except Exception as e:
            self._log(f"[TCP] send error: {e}")
            return False

    # ======================================================================
    # Station → us  (RECEIVED / DONE / ERROR / LOG)
    # ======================================================================

    def _handle_station_message(self, msg: str):
        low = msg.lower()

        # Live charger/PID output — relay straight to the GUI terminal.
        if msg.startswith("LOG:"):
            self._log(f"[PID] {msg[4:].strip()}")
            return

        if msg == "RECEIVED":
            self._log("[RX] RECEIVED (station ack)")
            return

        if msg.startswith("ERROR:"):
            self._log(f"[RX] {msg}")
            self._set_state("ERROR", msg[6:].strip())
            return

        # Everything else is a DONE:/status line — log it and advance the machine.
        self._log(f"[RX] {msg}")

        if "docking successful" in low:
            self._set_state("DOCKED")
            if self.auto_dock:
                self.start_charging(self.charge_mah)
        elif "charging started" in low or "charging with" in low:
            self._set_state("CHARGING")
        elif "charging stopped" in low or "charging complete" in low:
            self._set_state("CHARGED")
        elif "undocking successful" in low:
            self._set_state("UNDOCKED")
            self._send(CMD_IDLE)
            self._set_state("IDLE")
        elif "emergency_stop" in low:
            self._set_state("ERROR", "emergency stop performed")

    # ======================================================================
    # Actions / state machine
    # ======================================================================

    def _after_settle(self):
        if self.auto_dock and self.state in ("SETTLING", "BOOT"):
            self._log("[SM] settle complete → docking the drone on the pad")
            self.dock()
        else:
            self._set_state("IDLE", "settle complete (auto_dock off)")

    def dock(self):
        self._set_state("DOCKING")
        self._send(CMD_DOCK)

    def undock(self):
        self._set_state("UNDOCKING")
        self._send(CMD_UNDOCK)

    def start_charging(self, mah: int = None):
        mah = int(mah) if mah else self.charge_mah
        self._set_state("CHARGING", f"target {mah} mAh")
        self._send(f"{CMD_CHARGE} {mah}")

    def stop_charging(self):
        self._send(CMD_STOP)

    def emergency(self):
        self._log("[SM] EMERGENCY — stopping docking motors")
        self._send(CMD_EMERG)

    def idle(self):
        self._send(CMD_IDLE)
        self._set_state("IDLE")

    # ======================================================================
    # Inputs: GUI commands + flight landed-state
    # ======================================================================

    def on_command(self, msg: String):
        raw = msg.data.strip()
        cmd = raw.upper()
        self._log(f"[GUI] command: {raw}")
        if cmd == "DOCK":
            self.dock()
        elif cmd == "UNDOCK":
            self.undock()
        elif cmd.startswith("CHARGE"):
            # "CHARGE" or "CHARGE:450"
            mah = self.charge_mah
            if ":" in cmd:
                try: mah = int(cmd.split(":", 1)[1])
                except ValueError: pass
            self.start_charging(mah)
        elif cmd == "STOP_CHARGING":
            self.stop_charging()
        elif cmd == "EMERGENCY":
            self.emergency()
        elif cmd == "IDLE":
            self.idle()
        else:
            self._log(f"[GUI] ignoring unknown command: {raw}")

    def on_extended_state(self, msg):
        ls = msg.landed_state
        in_air = ls in (ExtendedState.LANDED_STATE_IN_AIR,
                        ExtendedState.LANDED_STATE_TAKEOFF)
        on_ground = (ls == ExtendedState.LANDED_STATE_ON_GROUND)

        if in_air:
            if not self._was_in_air:
                self._set_state("FLYING")
            self._was_in_air = True
        elif on_ground and self._was_in_air:
            # Transition in-air → on-ground = the drone just landed after a flight.
            self._was_in_air = False
            self._log("[SM] landing detected (extended_state ON_GROUND after IN_AIR)")
            if self.auto_dock:
                self.dock()

    def destroy_node(self):
        self._running = False
        try:
            if self._sock:
                self._sock.close()
        except Exception:
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = DockManager()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
