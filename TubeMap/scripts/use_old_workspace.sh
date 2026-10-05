#!/usr/bin/env bash
# Stop TubeMap and start the previous TubeWorkspace (tube-positioning) appliance again.
set -e
sudo systemctl disable --now tubemap.service || true
sudo systemctl enable --now tube-positioning.service
echo "Old TubeWorkspace service running (port 8080). Switch back: bash ~/tubemap/current/scripts/use_tubemap.sh"
