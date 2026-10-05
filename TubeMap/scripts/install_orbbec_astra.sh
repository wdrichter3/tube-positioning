#!/usr/bin/env bash
# Build and install the Orbbec SDK v1 Python wrapper (needed for Astra+ / Astra Pro Plus) into TubeMap's
# Python environment, and install Orbbec's USB rules. Run as your normal user on the Pi:
#     bash ~/tubemap/current/scripts/install_orbbec_astra.sh
# Takes 5-15 minutes on a Pi 5. Orbbec publishes no ARM64 wheel for SDK v1, so it is compiled here.
# Note: SDK v1 replaces pyorbbecsdk2 in this environment (Gemini/Femto cameras need v2 instead).
set -euo pipefail
VENV="${TUBEMAP_VENV:-$HOME/tubemap/venv}"
PY="$VENV/bin/python"
SRC="$HOME/tubemap/src/pyorbbecsdk-v1"
[[ -x "$PY" ]] || { echo "TubeMap environment not found at $VENV. Run install_pi.sh first."; exit 1; }
[[ $EUID -eq 0 ]] && { echo "Run as your normal user (not sudo)."; exit 1; }

echo "== Build tools"
sudo apt-get update -qq
sudo apt-get install -y -qq git cmake build-essential python3-dev libusb-1.0-0-dev >/dev/null

echo "== Source (Orbbec pyorbbecsdk, main branch = SDK v1)"
if [[ -d "$SRC/.git" ]]; then git -C "$SRC" pull --ff-only -q || true
else mkdir -p "$(dirname "$SRC")"; git clone -q --depth 1 -b main https://github.com/orbbec/pyorbbecsdk.git "$SRC"; fi

echo "== Build (pybind11 2.12+ so it works with NumPy 1.x and 2.x)"
"$PY" -m pip install -q "pybind11>=2.12"
"$PY" -m pip uninstall -y -q pyorbbecsdk2 pyorbbecsdk 2>/dev/null || true
rm -rf "$SRC/build" "$SRC/install"; mkdir -p "$SRC/build"
( cd "$SRC/build"
  cmake -DPython3_EXECUTABLE="$PY" -Dpybind11_DIR="$("$PY" -m pybind11 --cmakedir)" .. >/dev/null
  make -j"$(nproc)" >/dev/null
  make install >/dev/null )
LIB="$SRC/install/lib"
ls "$LIB"/pyorbbecsdk*.so >/dev/null || { echo "Build produced no module in $LIB"; exit 1; }
SITE="$("$PY" -c 'import site; print(site.getsitepackages()[0])')"
echo "$LIB" > "$SITE/orbbec_sdk_v1.pth"

echo "== USB rules"
if [[ -f "$SRC/scripts/install_udev_rules.sh" ]]; then
  ( cd "$SRC" && sudo bash ./scripts/install_udev_rules.sh ) || true
elif [[ -f "$SRC/scripts/99-obsensor-libusb.rules" ]]; then
  sudo cp "$SRC/scripts/99-obsensor-libusb.rules" /etc/udev/rules.d/
fi
sudo udevadm control --reload-rules && sudo udevadm trigger || true

echo "== Check"
"$PY" - <<'PYEOF'
import pyorbbecsdk as ob
gen = 2 if hasattr(ob.Pipeline, "get_d2c_depth_profile_list") else 1
print("pyorbbecsdk imported, SDK v%d" % gen)
PYEOF
sudo systemctl restart tubemap.service 2>/dev/null || true
echo
echo "Done. Unplug and replug the Astra+ (USB 3 port, blue), then in TubeMap: Devices > Add device >"
echo "Orbbec (USB) > Discover > Test > role 'Depth checks (companion)' > Save."
