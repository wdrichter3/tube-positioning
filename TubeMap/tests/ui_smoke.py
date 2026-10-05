"""Headless browser walk-through of every tab (python3 tests/ui_smoke.py [url] [outdir])."""
import sys, time
from playwright.sync_api import sync_playwright

URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8099"
OUT = sys.argv[2] if len(sys.argv) > 2 else "/tmp"
errors = []
with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_page(viewport={"width": 1500, "height": 900})
    pg.on("pageerror", lambda e: errors.append(str(e)))
    pg.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
    pg.goto(URL); pg.wait_for_timeout(2500)
    pg.screenshot(path=f"{OUT}/ui_live.png")
    pg.click("#btn-capture"); pg.wait_for_timeout(2500)
    pg.click("#tabs button[data-tab=map]"); pg.wait_for_timeout(1500)
    pg.fill("#g-pitch", "1.0"); pg.fill("#g-id", "0.62")
    pg.click("#map-build"); pg.wait_for_selector(".banner", timeout=30000); pg.wait_for_timeout(500)
    # select a tube: click the canvas centre
    box = pg.locator("#map-canvas").bounding_box()
    pg.mouse.click(box["x"] + box["width"] * 0.5, box["y"] + box["height"] * 0.45); pg.wait_for_timeout(300)
    pg.screenshot(path=f"{OUT}/ui_map.png")
    print("map banner:", pg.inner_text(".banner"))
    print("selected:", pg.inner_text("#sel-site")[:80].replace("\n", " "))
    pg.fill("#anc-x", "100"); pg.fill("#anc-y", "200"); pg.click("#anc-add"); pg.wait_for_timeout(800)
    pg.click("#map-to-label"); pg.wait_for_timeout(3500)
    pg.screenshot(path=f"{OUT}/ui_label.png")
    print("label counts:", pg.inner_text("#lab-counts").replace("\n", " "))
    lb = pg.locator("#lab-canvas").bounding_box()
    pg.keyboard.press("2")
    pg.keyboard.down("Shift"); pg.mouse.click(lb["x"] + lb["width"] * 0.5, lb["y"] + lb["height"] * 0.5); pg.keyboard.up("Shift")
    pg.click("#lab-review"); pg.wait_for_timeout(1000)
    print("label msg:", pg.inner_text("#lab-msg"))
    pg.click("#tabs button[data-tab=devices]"); pg.wait_for_timeout(1500)
    pg.click("#dev-add"); pg.wait_for_timeout(300)
    pg.select_option("#dev-driver", "agent"); pg.wait_for_timeout(300)
    pg.screenshot(path=f"{OUT}/ui_devices.png")
    pg.select_option("#dev-driver", "simulator"); pg.fill("#dev-name", "Second sim"); pg.click("#dev-save"); pg.wait_for_timeout(2500)
    print("devices:", pg.inner_text("#dev-list")[:300].replace("\n", " | "))
    pg.click("#tabs button[data-tab=model]"); pg.wait_for_timeout(1500)
    pg.screenshot(path=f"{OUT}/ui_model.png")
    pg.click("#tabs button[data-tab=settings]"); pg.wait_for_timeout(1000)
    pg.screenshot(path=f"{OUT}/ui_settings.png")
    pg.click("#tabs button[data-tab=live]"); pg.wait_for_timeout(1500)
    print("live devs:", pg.inner_text("#live-devs").replace("\n", " | ")[:200])
    b.close()
print("JS errors:", errors or "none")
