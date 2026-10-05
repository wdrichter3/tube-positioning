"""python -m tubemap [serve|selftest] ..."""
from __future__ import annotations

import argparse
import os
import signal
import sys
import threading
import webbrowser

from . import VERSION


def cmd_serve(a):
    from . import config
    from .server import serve
    if a.data:
        os.environ["TUBEMAP_DATA"] = a.data
    try:
        app, srv = serve(a.host, a.port, start_devices=not a.no_devices)
    except OSError as exc:
        port = a.port or config.load_settings().get("port", 8088)
        print(f"Port {port} is already in use ({exc}). TubeMap is probably already running in another window.")
        if a.open:
            webbrowser.open(f"http://localhost:{port}")
        return
    port = srv.server_address[1]
    print(f"TubeMap {VERSION}  data: {config.data_dir()}")
    print(f"Dashboard: http://localhost:{port}  (or http://<this-computer>:{port} on the network)")
    if a.open:
        threading.Timer(1.0, lambda: webbrowser.open(f"http://localhost:{port}")).start()

    def stop(*_):
        threading.Thread(target=srv.shutdown, daemon=True).start()
    signal.signal(signal.SIGTERM, stop)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        app.devices.shutdown()
        srv.server_close()


def cmd_selftest(a):
    import time
    from .vision import mapping, synth
    ok = True
    for pat, pitch, tid in (("TRIANGULAR", 1.0, 0.62), ("SQUARE", 1.25, 0.834)):
        t = time.time()
        sc = synth.render(pattern=pat, pitch_in=pitch, tube_id_in=tid, tilt_deg=15)
        g = {"mode": "known", "pattern": pat, "pitch_mm": sc["pitch_mm"], "tube_id_mm": sc["tube_id_mm"]}
        m = mapping.build_map(sc["color"], sc["depth"], sc["intrinsics"], g)
        truth = sum(1 for x in sc["truth"] if x["state"] == "OPEN")
        got = m["counts"].get("OPEN", 0)
        good = m["ready"] and got == truth
        ok &= good
        print(f"{pat:10s} open {got}/{truth}  ready={m['ready']}  {time.time() - t:.1f}s  {'PASS' if good else 'FAIL'}")
    try:
        from .devices.oak import OakDriver, build_pipeline
        import depthai as dai
        build_pipeline(dai, OakDriver.normalize({}))[0].serializeToJson()
        print("DepthAI pipeline builds: PASS", getattr(dai, "__version__", ""))
    except ImportError:
        print("DepthAI not installed: OAK driver unavailable")
    except Exception as exc:
        ok = False
        print("DepthAI pipeline: FAIL", exc)
    sys.exit(0 if ok else 1)


def cmd_import(a):
    from . import dataset
    if a.data:
        os.environ["TUBEMAP_DATA"] = a.data
    from pathlib import Path
    p = Path(a.path).expanduser()
    r = dataset.import_imagelab_dir(p) if p.is_dir() else dataset.import_zip(p.read_bytes())
    print(f"Imported {r['imported']} capture(s) ({r['format']}), skipped {r['skipped']}.")


def cmd_doctor(a):
    """Environment and camera report; also written to <data>/doctor.txt to send along with questions."""
    import json
    import platform
    import socket
    from . import config
    lines = []

    def out(k, v=""):
        line = f"{k:<26}{v}"
        print(line, flush=True)
        lines.append(line)

    out("TubeMap", VERSION)
    out("Python", f"{platform.python_version()} ({sys.executable})")
    out("System", f"{platform.system()} {platform.release()} {platform.machine()}")
    for mod in ("numpy", "cv2"):
        try:
            m = __import__(mod)
            out(mod, m.__version__)
        except Exception as exc:
            out(mod, f"MISSING ({exc})")
    try:
        import depthai as dai
        out("depthai", dai.__version__)
        devs = dai.Device.getAllAvailableDevices()
        out("OAK cameras found", len(devs))
        for d in devs:
            out("  OAK", f"{d.getMxId()} state={d.state.name} protocol={d.protocol.name}")
    except ImportError:
        out("depthai", "not installed (OAK unavailable)")
    except Exception as exc:
        out("OAK search", f"error: {exc}")
    try:
        from .devices.orbbec_capture import sdk_generation
        gen = sdk_generation()
        out("pyorbbecsdk", "not installed" if gen is None else f"SDK v{gen}" + (" (Astra+ OK)" if gen == 1 else " (no Astra+ support)"))
        if gen:
            from .devices.orbbec import probe_devices
            try:
                found = probe_devices()
                out("Orbbec cameras found", len(found))
                for d in found:
                    out("  Orbbec", f"{d.get('name')} serial={d.get('serial')}")
            except Exception as exc:
                out("Orbbec search", f"error: {exc}")
    except Exception as exc:
        out("pyorbbecsdk", f"error: {exc}")
    out("Data folder", str(config.data_dir()))
    for d in config.load_devices():
        out("  configured", f"{d['name']} [{d['driver']}, role {d['role']}, {'on' if d.get('enabled') else 'off'}]")
    port = int(config.load_settings().get("port", 8088))
    with socket.socket() as s:
        busy = s.connect_ex(("127.0.0.1", port)) == 0
    out("Port", f"{port} {'IN USE (TubeMap already running?)' if busy else 'free'}")
    try:
        (config.data_dir() / "doctor.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"\nSaved to {config.data_dir() / 'doctor.txt'}")
    except Exception:
        pass


def main(argv=None):
    ap = argparse.ArgumentParser(prog="tubemap", description=f"TubeMap {VERSION}")
    sub = ap.add_subparsers(dest="cmd")
    s = sub.add_parser("serve", help="run the dashboard and device service")
    s.add_argument("--host", default="0.0.0.0")
    s.add_argument("--port", type=int, default=None)
    s.add_argument("--data", default=None, help="data folder (default ~/.tubemap)")
    s.add_argument("--no-devices", action="store_true", help="don't start cameras (labeling only)")
    s.add_argument("--open", action="store_true", help="open a browser")
    sub.add_parser("selftest", help="check the mapping pipeline on simulated sheets")
    sub.add_parser("doctor", help="check Python packages and list connected cameras")
    im = sub.add_parser("import", help="import an Image Lab folder/ZIP or a TubeMap dataset ZIP")
    im.add_argument("path")
    im.add_argument("--data", default=None)
    a = ap.parse_args(argv)
    if a.cmd == "selftest":
        cmd_selftest(a)
    elif a.cmd == "import":
        cmd_import(a)
    elif a.cmd == "doctor":
        cmd_doctor(a)
    else:
        if a.cmd is None:
            a = ap.parse_args(["serve"] + (argv or sys.argv[1:]))
        cmd_serve(a)


if __name__ == "__main__":
    main()
