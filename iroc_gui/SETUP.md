# Anveshan GCS — Setup Guide

This is the **ground-control station (GUI)** for the Anveshan drone. This guide gets
it running on two machines. No prior experience needed — copy the commands top to bottom.

---

## The idea in one picture

You have **two machines**, each with one job:

| Machine | What runs there | Why |
|---|---|---|
| **Jetson** (on the drone) | The whole GUI *engine* — backend + video + ROS | ROS lives here; keep it local so nothing dies over Wi-Fi |
| **Laptop** (companion) | Just a **web browser** (real flights) — OR the full thing in **mock mode** (testing, no drone) | It's a window into the Jetson |

**Tip:** set it up on the **laptop first** (Part A). It proves the whole GUI works in
5 minutes with fake data and zero risk to the drone. Then deploy to the Jetson (Part B).

---

## Part A — Laptop (test with mock data, no drone needed)

Open a terminal and run these once:

```bash
# 0. One-time prerequisites (Ubuntu/Debian). Safe to re-run.
sudo apt update
sudo apt install -y git python3 python3-venv python3-pip

# 1. Create the workspace
mkdir -p ~/IROC_Integrated
cd ~/IROC_Integrated

# 2. Get the GUI code (branch: gui) and copy ONLY the GUI folder in.
#    (The flight-side folders are deliberately left out.)
git clone -b gui https://github.com/chakri2007/IROC-u.git /tmp/iroc-src
cp -r /tmp/iroc-src/iroc_gui/. ~/IROC_Integrated/
rm -rf /tmp/iroc-src

# 3. Make a fresh virtual environment
python3 -m venv .venv
source .venv/bin/activate            # your prompt now shows (.venv)

# 4. Install the GUI's Python packages
pip install --upgrade pip
pip install -r requirements.txt
```

Now run it — **two terminals**, both starting with `cd ~/IROC_Integrated`:

```bash
# ── Terminal 1: the backend (engine) ──
cd ~/IROC_Integrated
source .venv/bin/activate
python3 backend/backend.py
#   Expect:  [WARN] rclpy not found – running with mock data
#   That WARN is CORRECT on a laptop — it means "no drone, showing test data."
#   Backend is live at http://localhost:5000  (API docs at /docs)
```

```bash
# ── Terminal 2: serve the web page ──
cd ~/IROC_Integrated
python3 -m http.server 8080 -d frontend
```

Open a browser on the laptop:

```
http://localhost:8080
```

You'll see moving mock telemetry, the instrument frames on the right, and
**Initiate Setup** at the bottom-right under them. On a laptop, "Initiate Setup" won't
launch real ROS nodes (there's no ROS here) — that's for the Jetson.

Stop either terminal with **Ctrl + C**.

---

## Part B — Jetson (real deployment on the drone)

Same workspace, one difference: the Jetson **has** ROS, so the virtual environment must
be allowed to see the system's `rclpy`. The key flag is **`--system-site-packages`**.

```bash
# On the Jetson:
mkdir -p ~/IROC_Integrated && cd ~/IROC_Integrated
git clone -b gui https://github.com/chakri2007/IROC-u.git /tmp/iroc-src
cp -r /tmp/iroc-src/iroc_gui/. ~/IROC_Integrated/
rm -rf /tmp/iroc-src

# venv that can see system ROS packages  ← the important part
python3 -m venv .venv --system-site-packages

# Source ROS FIRST, then activate the venv
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=1
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Run it (two terminals). Each terminal starts with this same "environment" line:

```bash
source /opt/ros/humble/setup.bash && export ROS_DOMAIN_ID=1 && source ~/IROC_Integrated/.venv/bin/activate
```

```bash
# Terminal 1 — backend (should NOT print the mock WARN now)
cd ~/IROC_Integrated && python3 backend/backend.py

# Terminal 2 — web page
cd ~/IROC_Integrated && python3 -m http.server 8080 -d frontend
```

Then, from the **laptop's browser** (same Wi-Fi as the Jetson):

```
http://<JETSON-IP>:8080
```

Find `<JETSON-IP>` by running `hostname -I` on the Jetson (use the first number,
e.g. `192.168.0.42`). The page auto-connects to the backend on that same Jetson IP.

---

## Updating to the latest GUI later

When the GUI team pushes changes, refresh your copy:

```bash
cd ~/IROC_Integrated
git clone -b gui https://github.com/chakri2007/IROC-u.git /tmp/iroc-src
cp -r /tmp/iroc-src/iroc_gui/. ~/IROC_Integrated/     # overwrites code, keeps your .venv + config.json
rm -rf /tmp/iroc-src
```

Then restart the two terminals. Your local `config.json` (your saved settings) is not
overwritten.

---

## Troubleshooting

- **Backend prints `[WARN] rclpy not found`**
  - On a **laptop**: correct — it's running mock data.
  - On the **Jetson**: the venv didn't pick up ROS. Fallback that always works: skip the
    venv, just `source /opt/ros/humble/setup.bash` then `python3 backend/backend.py`.
- **Browser shows the page but no data / "connecting"**: the backend (Terminal 1) isn't
  running, or you opened the wrong IP. The page always talks to port **5000** on the same
  host that served it.
- **`git clone` says "already exists"**: delete `/tmp/iroc-src` first (`rm -rf /tmp/iroc-src`).
- **Instrument frames look squished / cramped**: you're on an old version. Update (see above).
- **Two things use port 8080 / 5000**: something's already running. Close old terminals or
  reboot.

---

## What the buttons do

- **Initiate Setup** (bottom-right, under the instruments): brings up the Jetson software
  stack in order (SLAM, MAVROS, camera, video, rosbag, semantic). Jetson only.
- **CONFIG** (top bar): all editable settings — IPs, camera, paths, thresholds.
- **DOCK** (top bar): docking / charging station manager.
- **OVERRIDES** (top bar): START / ABORT / HOLD.
  - ⚠️ These are **display-only until the flight team adds `gcs_command_bridge`** (see
    `HANDOFF.md`). Until then, do not rely on ABORT from the GUI.
