#!/usr/bin/env python3
"""
station_gcs.py — Docking-station TCP server (runs on the docking-station RPi).
=============================================================================
Cleaned-up successor to the hand-written gcs.py. Same wire protocol, same
Arduino actions, plus the piece that was missing: it now STREAMS the live
charger / PID output back over TCP as "LOG:<line>" frames, so the GUI's docking
terminal can show what the charging loop is doing.

WIRE PROTOCOL  (matches dock_manager.py)
  Frame = 4-byte big-endian length + UTF-8 payload.
  Drone/GUI → us : START_DOCKING | UNDOCK | IDLE | START_CHARGING <mah> | STOP_CHARGING | EMERGENCY
  Us → drone/GUI : "RECEIVED"  (immediate ack)
                   "DONE:<text>" / "ERROR:<text>"  (final result)
                   "DONE:Charging stopped"          (async, when charging finishes)
                   "LOG:<line>"                     (async, live charger/PID output)  ← NEW

HARDWARE
  Arduino Mega on SERIAL_PORT @ BAUD:  'f' = dock,  's' = emergency stop,  'b' = undock.
  Charger driven by CHARGER_SCRIPT (battery_charging_to_particular_mah.py --target-mah N).
"""

import socket
import threading
import subprocess
import sys
import time

import serial

# ---- Config (edit for the station) ----
HOST = "0.0.0.0"
PORT = 55555
SERIAL_PORT = "/dev/ttyACM0"
BAUD = 9600
CHARGER_SCRIPT = "battery_charging_to_particular_mah.py"
DEFAULT_CHARGE_MAH = 260            # pinned default (was inconsistent 260/450 before)

charger_process = None
current_client = None
client_lock = threading.Lock()      # serialise all sends to the one connected client


# ====================== TCP send helper ======================
def send_frame(text: str):
    """Send one length-prefixed UTF-8 frame to the connected client (thread-safe)."""
    with client_lock:
        sock = current_client
        if sock is None:
            return
        try:
            data = text.encode("utf-8")
            sock.sendall(len(data).to_bytes(4, "big"))
            sock.sendall(data)
        except Exception as e:
            print(f"[TCP] send error: {e}")


# ====================== Arduino actions ======================
def _arduino(byte_cmd: bytes, settle: float = 0.0):
    ser = serial.Serial(SERIAL_PORT, BAUD, timeout=1)
    time.sleep(2)            # let the board reset after opening the port
    ser.write(byte_cmd)
    ser.close()
    if settle:
        time.sleep(settle)


def start_docking():
    print("[ACTION] Docking…")
    try:
        _arduino(b"f", settle=15)
        print("[ACTION] Docking complete")
        return "Docking successful"
    except Exception as e:
        return f"Docking failed: {e}"


def stop_docking():
    print("[ACTION] Emergency stop…")
    try:
        _arduino(b"s")
        print("[ACTION] Emergency stop complete")
        return "Emergency_stop successful"
    except Exception as e:
        return f"Emergency_stop failed: {e}"


def start_undocking():
    print("[ACTION] Undocking…")
    try:
        _arduino(b"b", settle=15)
        print("[ACTION] Undocking complete")
        return "Undocking successful"
    except Exception as e:
        return f"Undocking failed: {e}"


def go_idle():
    print("[ACTION] Idle")
    return "Idle mode activated"


# ====================== Charging ======================
def start_charging(target_mah: int = DEFAULT_CHARGE_MAH):
    global charger_process
    print(f"[CHARGER] Starting, target {target_mah} mAh")
    try:
        if charger_process and charger_process.poll() is None:
            charger_process.terminate()
            time.sleep(1.5)

        # -u (unbuffered) so the charger's print()s stream line-by-line over TCP
        # in real time. This is the ONLY accommodation needed — the charger script
        # itself is untouched (it's a reverse-engineered, known-good binary path).
        charger_process = subprocess.Popen(
            ["sudo", sys.executable, "-u", CHARGER_SCRIPT, "--target-mah", str(target_mah)],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, universal_newlines=True,
        )
        # Stream its output over TCP (the live PID lines) + watch for completion.
        threading.Thread(target=stream_charger_output, args=(charger_process,), daemon=True).start()
        threading.Thread(target=monitor_charger,        args=(charger_process,), daemon=True).start()
        return f"Charging started with {target_mah} mAh"
    except Exception as e:
        return f"Failed to start charger: {e}"


def stream_charger_output(process):
    """Relay each charger line to the GUI terminal as a LOG: frame (the PID stream)."""
    try:
        for line in iter(process.stdout.readline, ""):
            line = line.strip()
            if line:
                print(f"[CHARGER] {line}")
                send_frame(f"LOG:{line}")
    except Exception:
        pass


def monitor_charger(process):
    process.wait()
    print("[CHARGER] Charging completed")
    send_frame("DONE:Charging stopped")


def stop_charging():
    global charger_process
    if charger_process and charger_process.poll() is None:
        try:
            charger_process.terminate()
            charger_process.wait(timeout=6)
            return "Charging stopped"
        except Exception:
            charger_process.kill()
            return "Charging force stopped"
    return "No charger is running"


# ====================== Command routing ======================
COMMAND_HANDLERS = {
    "START_DOCKING": start_docking,
    "UNDOCK":        start_undocking,
    "IDLE":          go_idle,
    "STOP_CHARGING": stop_charging,
    "EMERGENCY":     stop_docking,
}


def _recvall(sock, n):
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return buf


def handle_client(client_socket, addr):
    global current_client
    print(f"[TCP] client connected: {addr}")
    with client_lock:
        current_client = client_socket
    try:
        while True:
            header = _recvall(client_socket, 4)
            if header is None:
                break
            length = int.from_bytes(header, "big")
            payload = _recvall(client_socket, length)
            if payload is None:
                break

            command_raw = payload.decode("utf-8", "replace").strip().upper()
            print(f"[TCP] received: {command_raw}")
            parts = command_raw.split()
            command = parts[0] if parts else ""
            param = parts[1] if len(parts) > 1 else None

            # 1) immediate ack
            send_frame("RECEIVED")

            # 2) execute
            if command == "START_CHARGING":
                mah = int(param) if (param and param.isdigit()) else DEFAULT_CHARGE_MAH
                reply = f"DONE:{start_charging(mah)}"
            elif command in COMMAND_HANDLERS:
                try:
                    reply = f"DONE:{COMMAND_HANDLERS[command]()}"
                except Exception as e:
                    reply = f"ERROR:{e}"
            else:
                reply = "DONE:UNKNOWN_COMMAND"

            # 3) final result
            send_frame(reply)

    except Exception as e:
        print(f"[TCP] client {addr} error: {e}")
    finally:
        with client_lock:
            if current_client is client_socket:
                current_client = None
        client_socket.close()
        print(f"[TCP] client {addr} disconnected")


def main():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((HOST, PORT))
    server.listen(5)
    print(f"[GCS] docking-station server on {HOST}:{PORT} — waiting for the drone/GUI…")
    try:
        while True:
            client_socket, addr = server.accept()
            threading.Thread(target=handle_client, args=(client_socket, addr), daemon=True).start()
    except KeyboardInterrupt:
        print("\n[GCS] shutting down")
    finally:
        server.close()


if __name__ == "__main__":
    main()
