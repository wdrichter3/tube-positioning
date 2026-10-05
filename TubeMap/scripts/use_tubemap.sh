#!/usr/bin/env bash
# Stop the previous TubeWorkspace appliance and run TubeMap (only one can use the OAK at a time).
set -e
sudo systemctl disable --now tube-positioning.service 2>/dev/null || true
sudo systemctl enable --now tubemap.service
echo "TubeMap running: http://$(hostname).local:8088"
