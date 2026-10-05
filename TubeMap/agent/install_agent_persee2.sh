#!/usr/bin/env bash
# Run ON the Orbbec Persee 2 (Ubuntu image) from the copied agent folder:  bash install_agent_persee2.sh
# Installs the TubeMap camera agent as a service on port 8095.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYBIN="${PYBIN:-python3}"
echo "Python: $($PYBIN --version)"
$PYBIN -c "import numpy, cv2" 2>/dev/null || sudo apt-get install -y python3-numpy python3-opencv
if ! $PYBIN -c "import pyorbbecsdk" 2>/dev/null; then
  echo "pyorbbecsdk is not importable with $PYBIN."
  echo "Install it following agent/README.md (Persee 2 section), then re-run this script."
  echo "You can still test the agent with:  $PYBIN $HERE/tubemap_agent.py --simulate"
  exit 1
fi
sudo tee /etc/systemd/system/tubemap-agent.service >/dev/null <<UNITEOF
[Unit]
Description=TubeMap camera agent
After=network-online.target
[Service]
User=$USER
WorkingDirectory=$HERE
ExecStart=$(command -v $PYBIN) $HERE/tubemap_agent.py --port 8095
Restart=always
RestartSec=3
[Install]
WantedBy=multi-user.target
UNITEOF
sudo systemctl daemon-reload
sudo systemctl enable --now tubemap-agent.service
echo "Agent running on port 8095. In the TubeMap dashboard: Devices > Add device > Network camera agent > $(hostname -I | awk '{print $1}')"
