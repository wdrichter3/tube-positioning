# TubeMap camera agent (Orbbec Persee 2)

The Persee 2 is a small computer (Amlogic A311D, Ubuntu 18.04 or Android) with an Orbbec Gemini 2 L depth camera inside. The Pi can't reach that camera over USB, so the agent runs on the Persee and serves the camera over Ethernet. In the TubeMap dashboard, add it as **Devices › Add device › Network camera agent**.

## Files

- `tubemap_agent.py`: the agent. It needs Python 3.6+, NumPy and OpenCV.
- `orbbec_capture.py`: colour and aligned depth capture through `pyorbbecsdk` (SDK v1 or v2 API).
- `install_agent_persee2.sh`: installs the agent as a service on port 8095.

## Setup

1. Put the Persee 2 on the same network as the Pi, using wired Ethernet. Use the Ubuntu image, not Android.
2. Copy this `agent` folder to the Persee:
   ```
   scp -r agent <user>@<persee-ip>:~/tubemap-agent
   ```
3. On the Persee, check the agent without the camera first:
   ```
   cd ~/tubemap-agent
   python3 tubemap_agent.py --simulate
   ```
   In the dashboard, add a Network camera agent with the Persee's IP and press **Test**. You should see a dot pattern.
4. Make `pyorbbecsdk` importable on the Persee:
   - Check with `python3 -c "import pyorbbecsdk"`. Orbbec's Persee image may already include the SDK and its Python examples.
   - If it's missing, install Orbbec's Python wrapper for Linux ARM64. The wheels need Python 3.8 or newer. On Ubuntu 18.04:
     ```
     sudo apt install python3.8 python3.8-venv
     python3.8 -m pip install pyorbbecsdk2 numpy opencv-python-headless
     ```
     Then run the agent with `python3.8`. If there's no wheel for the Persee, build the wrapper from the pyorbbecsdk GitHub repository following Orbbec's instructions.
5. Run it for real:
   ```
   python3 tubemap_agent.py --width 1280 --height 800 --fps 15
   ```
   Or install it as a service:
   ```
   PYBIN=python3.8 bash install_agent_persee2.sh
   ```

## Protocol

- `GET /info`: device, intrinsics and agent version.
- `GET /frame.jpg?max=960`: latest colour frame, for live preview.
- `GET /capture`: the next colour and depth pair, lossless, as an `.npz` file (`color`, `depth` in mm, `meta` JSON).

## Notes

- The Gemini 2 L's colour camera is 1280×800 with a wide ~93° field of view, well below the OAK-D Pro's 12 MP. Use the Persee for the coarse obstacle and overview model, and the OAK for tube detection.
- The Orbbec capture code follows the documented pyorbbecsdk API but hasn't been run on a Persee 2 yet. If `/capture` returns an error, `journalctl -u tubemap-agent` or the console output shows which call failed.
