#!/usr/bin/env bash
# TubeMap installer for the Raspberry Pi (Ubuntu 24.04 / ROS Jazzy image or Raspberry Pi OS Bookworm).
# Run as your normal user from the extracted TubeMap folder:   bash install_pi.sh
# Options:  --keep-old   leave the old tube-positioning service running (camera conflict!)
#           --port N     dashboard port (default 8088)
#           --astra      also build the Orbbec SDK v1 for an Astra+ camera (adds 5-15 minutes)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERSION="$(sed -n 's/^VERSION = "\(.*\)"/\1/p' "$HERE/tubemap/__init__.py")"
PORT=8088; KEEP_OLD=0; ASTRA=0
while [[ $# -gt 0 ]]; do case "$1" in --keep-old) KEEP_OLD=1;; --port) PORT="$2"; shift;; --astra) ASTRA=1;; *) echo "unknown option $1"; exit 2;; esac; shift; done
[[ $EUID -eq 0 ]] && { echo "Run as your normal user (not sudo). The script asks for sudo when needed."; exit 1; }

ROOT="$HOME/tubemap"; APP="$ROOT/app-$VERSION"; VENV="$ROOT/venv"
echo "== TubeMap $VERSION -> $ROOT"
mkdir -p "$ROOT"
rm -rf "$APP.tmp"; mkdir -p "$APP.tmp"
cp -r "$HERE/tubemap" "$HERE/agent" "$HERE/scripts" "$HERE/README.md" "$APP.tmp/"
rm -rf "$APP"; mv "$APP.tmp" "$APP"
ln -sfn "$APP" "$ROOT/current"

echo "== Python environment"
command -v python3 >/dev/null || { echo "python3 missing"; exit 1; }
python3 -c "import venv, ensurepip" 2>/dev/null || { sudo apt-get update && sudo apt-get install -y python3-venv; }
[[ -x "$VENV/bin/python" ]] || python3 -m venv --system-site-packages "$VENV"
PY="$VENV/bin/python"
"$PY" -m pip install -q --upgrade pip wheel || true
"$PY" -c "import numpy, cv2" 2>/dev/null || "$PY" -m pip install -q "numpy" "opencv-python-headless"
if ! "$PY" -c "import depthai" 2>/dev/null; then
  if ! "$PY" -m pip install -q "depthai>=2.24,<3"; then
    OLD="$HOME/projects/tube-positioning/oak_venv"
    SITE="$(ls -d "$OLD"/lib/python3*/site-packages 2>/dev/null | head -1 || true)"
    if [[ -n "$SITE" ]]; then
      echo "   pip could not install depthai; reusing the one in $OLD"
      "$PY" - "$SITE" <<'PYEOF'
import sys, site, pathlib
p = pathlib.Path(site.getsitepackages()[0]) / "zz_oak_venv.pth"
p.write_text(sys.argv[1] + "\n")
PYEOF
    else
      echo "   WARNING: depthai not available. OAK cameras will show 'not installed' until it is."
    fi
  fi
fi
"$PY" -c "import depthai; print('   depthai', depthai.__version__)" 2>/dev/null || true
"$PY" -c "import sklearn" 2>/dev/null || "$PY" -m pip install -q "scikit-learn>=1.3" || echo "   scikit-learn not installed: training uses the simpler linear model"
if [[ "${TUBEMAP_ORBBEC:-0}" == "1" ]]; then "$PY" -m pip install -q pyorbbecsdk2 || echo "   pyorbbecsdk not installed"; fi

echo "== USB permissions for OAK (Movidius 03e7)"
RULE=/etc/udev/rules.d/80-movidius.rules
if ! grep -qs 03e7 "$RULE"; then
  echo 'SUBSYSTEM=="usb", ATTRS{idVendor}=="03e7", MODE="0666"' | sudo tee "$RULE" >/dev/null
  sudo udevadm control --reload-rules && sudo udevadm trigger || true
fi

if [[ $ASTRA -eq 1 ]]; then
  echo "== Orbbec SDK v1 for Astra+"
  TUBEMAP_VENV="$VENV" bash "$ROOT/current/scripts/install_orbbec_astra.sh" || echo "   Astra+ SDK build failed; TubeMap still installs. Re-run scripts/install_orbbec_astra.sh later."
fi

echo "== Self-test"
( cd "$ROOT/current" && "$PY" -m tubemap selftest ) || { echo "Self-test failed; not switching services."; exit 1; }

if [[ -d "$HOME/.tube_positioning/image_lab/annotations" && ! -f "$HOME/.tubemap/.imagelab_imported" ]]; then
  echo "== Importing existing Image Lab labels (originals are left untouched)"
  ( cd "$ROOT/current" && "$PY" -m tubemap import "$HOME/.tube_positioning/image_lab" ) && mkdir -p "$HOME/.tubemap" && touch "$HOME/.tubemap/.imagelab_imported" || echo "   import skipped"
fi

echo "== Service"
UNIT=/etc/systemd/system/tubemap.service
sudo tee "$UNIT" >/dev/null <<UNITEOF
[Unit]
Description=TubeMap tubesheet mapping dashboard
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$ROOT/current
Environment=PYTHONUNBUFFERED=1
ExecStart=$PY -m tubemap serve --port $PORT
Restart=always
RestartSec=3
TimeoutStopSec=10

[Install]
WantedBy=multi-user.target
UNITEOF
sudo systemctl daemon-reload
if [[ $KEEP_OLD -eq 0 ]] && systemctl list-unit-files | grep -q '^tube-positioning.service'; then
  echo "   stopping the old tube-positioning service so TubeMap can use the camera"
  echo "   (switch back any time: bash $ROOT/current/scripts/use_old_workspace.sh)"
  sudo systemctl disable --now tube-positioning.service || true
fi
sudo systemctl enable tubemap.service
sudo systemctl restart tubemap.service

echo -n "== Waiting for dashboard"
for i in $(seq 1 40); do
  if curl -fs "http://localhost:$PORT/api/status" >/dev/null 2>&1; then echo " ready"; break; fi
  echo -n "."; sleep 1
done
IP="$(hostname -I 2>/dev/null | awk '{print $1}')"
echo
echo "TubeMap $VERSION is running:  http://$(hostname).local:$PORT   or   http://$IP:$PORT"
echo "Logs: journalctl -u tubemap -f     Data: ~/.tubemap"
