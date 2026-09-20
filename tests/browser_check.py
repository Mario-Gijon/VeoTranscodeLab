"""Local Chromium integration and CPU/GLSL parity check. No Veo network access.

Run from the repository root: .venv/bin/python tests/browser_check.py
Starts temporary servers and isolated match storage; never touches historical output.
"""
from __future__ import annotations

import base64
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from urllib.request import urlopen

import cv2
import numpy as np
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis.reproduce_veo_panorama import render
from analysis.veo_metadata import load_veo_metadata
from dev_app.backend.app.view_geometry import view_context


def ready(url, process):
    for _ in range(100):
        if process.poll() is not None:
            raise RuntimeError("Test server exited before startup")
        try:
            with urlopen(url, timeout=1) as response:
                if response.status == 200:
                    return
        except OSError:
            time.sleep(.1)
    raise RuntimeError("Test server startup timed out")


def main():
    with tempfile.TemporaryDirectory(prefix="veo-browser-check-") as temporary:
        work = Path(temporary)
        env = {**os.environ, "VEO_RUNTIME_ROOT": str(work / "runtime"), "VITE_API_BASE_URL": "http://127.0.0.1:8005", "VEO_FRONTEND_ORIGINS": "http://127.0.0.1:5175"}
        with (work / 'servers.log').open('w') as log:
            backend = subprocess.Popen([sys.executable, '-m', 'uvicorn', 'dev_app.backend.app.main:app', '--host', '127.0.0.1', '--port', '8005'], cwd=ROOT, env=env, stdout=log, stderr=log)
            frontend = subprocess.Popen(['node', 'node_modules/vite/bin/vite.js', '--host', '127.0.0.1', '--port', '5175', '--strictPort'], cwd=ROOT / 'dev_app/frontend', env=env, stdout=log, stderr=log)
            try:
                ready('http://127.0.0.1:8005/api/health', backend)
                with urlopen('http://127.0.0.1:8005/openapi.json') as response:
                    specification = json.load(response)
                    assert '/api/view-point' in specification['paths']
                    assert '/api/veo/inspect' in specification['paths']
                    print('FastAPI startup and OpenAPI passed', flush=True)
                ready('http://127.0.0.1:5175', frontend)
                with sync_playwright() as playwright:
                    browser = playwright.chromium.launch(headless=True, args=['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader'])
                    page = browser.new_page(viewport={"width": 2200, "height": 1200})
                    errors = []
                    page.on('pageerror', lambda error: errors.append(str(error)))
                    page.goto('http://127.0.0.1:5175')
                    page.get_by_role('heading', name='Matches', exact=True).wait_for()
                    for index, calibration in enumerate(sorted((ROOT / 'data').glob('*.veo'))):
                        image_path = ROOT / 'samples' / ('stacked_frame.jpg' if index == 0 else 'match2_stacked_frame.jpg')
                        image = cv2.imread(str(image_path))
                        png = base64.b64encode(cv2.imencode('.png', image)[1]).decode()
                        context = view_context(str(calibration))
                        actual = page.evaluate('''async ({context, png}) => {
                          const {createPanoramaGL} = await import('/src/components/panoramaGL.ts');
                          const canvas = document.createElement('canvas'); canvas.width=640; canvas.height=360;
                          const renderer = createPanoramaGL(canvas, context);
                          const image = new Image(); image.src='data:image/png;base64,'+png; await image.decode();
                          renderer.draw(image); const result=canvas.toDataURL('image/png'); renderer.dispose(); return result;
                        }''', {'context': context, 'png': png})
                        actual_image = cv2.imdecode(np.frombuffer(base64.b64decode(actual.split(',')[1]), dtype=np.uint8), cv2.IMREAD_COLOR)
                        expected = render(image, load_veo_metadata(calibration), 640, 360)['panorama']
                        difference = np.abs(expected.astype(float) - actual_image.astype(float))
                        mean, p99 = float(difference.mean()), float(np.percentile(difference, 99))
                        print(json.dumps({'match': calibration.name, 'mean_rgb_error': mean, 'p99_rgb_error': p99}), flush=True)
                        assert mean < 2 and p99 < 10, "Captured shader / CPU render parity failed"
                    # Generate a tiny-duration local fixture with browser-compatible codec.
                    clip = work / 'dual.mp4'
                    subprocess.run(['ffmpeg', '-v', 'error', '-loop', '1', '-i', str(ROOT / 'samples/stacked_frame.jpg'), '-t', '3', '-r', '5', '-c:v', 'libx264', '-preset', 'ultrafast', '-pix_fmt', 'yuv420p', str(clip)], check=True)
                    request = page.request
                    calibrations = request.get('http://127.0.0.1:8005/api/calibrations').json()
                    match = request.post('http://127.0.0.1:8005/api/matches', data={'name': 'Browser fixture', 'calibration_id': calibrations[0]['id']}).json()
                    for role in ['dual_lens', 'tracking']:
                        response = request.post(f"http://127.0.0.1:8005/api/matches/{match['id']}/media/path", data={'path': str(clip), 'role': role})
                        assert response.status == 201
                    page.reload()
                    captures = Path(os.environ['VEO_UI_CAPTURE_DIR']) if os.environ.get('VEO_UI_CAPTURE_DIR') else None
                    if captures:
                        captures.mkdir(parents=True, exist_ok=True)
                        page.get_by_role('button', name='Open match', exact=True).wait_for()
                        page.screenshot(path=str(captures / 'library.png'))
                    page.get_by_role('button', name='Open match', exact=True).click()
                    page.wait_for_function("document.querySelector('video')?.readyState >= 2")
                    page.get_by_label('Time (s)').fill('1.2')
                    page.wait_for_function("Math.abs(document.querySelector('video').currentTime-1.2)<0.02 && !document.querySelector('video').seeking")
                    page.get_by_role('button', name='Dual lens', exact=True).click()
                    page.get_by_role('button', name='RIGHT', exact=True).click()
                    page.locator('canvas').first.click(position={'x': 400, 'y': 250})
                    page.get_by_role('button', name='Measurements · 1 points').wait_for()
                    page.get_by_role('button', name='Tracking', exact=True).click()
                    page.wait_for_function("Math.abs(document.querySelector('video').currentTime-1.2)<0.02")
                    assert page.get_by_text('Tracking supports playback.', exact=False).count() == 1
                    page.get_by_role('button', name='Panorama', exact=True).click()
                    page.wait_for_function("document.querySelector('video')?.readyState >= 2 && !document.querySelector('video').seeking")
                    page.locator('canvas').first.click(position={'x': 500, 'y': 320})
                    page.get_by_role('button', name='Measurements · 2 points').wait_for()
                    page.get_by_role('button', name='Play', exact=True).click()
                    page.wait_for_function("document.querySelector('video').currentTime > 1.5")
                    page.get_by_role('button', name='Pause', exact=True).click()
                    if captures:
                        page.screenshot(path=str(captures / 'workspace.png'))
                        page.get_by_role('button', name='Manage files', exact=True).click()
                        page.get_by_role('dialog').wait_for()
                        page.screenshot(path=str(captures / 'manage-files.png'), animations='disabled')
                        page.get_by_role('button', name='Done', exact=True).click()
                    page.set_viewport_size({'width': 900, 'height': 1000})
                    assert page.locator('svg[aria-label="Pitch reference"]').count() == 1
                    assert not errors, errors
                    print('Browser workflow: library, roles, shared seek, raw/panorama clicks, playback, responsive pitch passed', flush=True)
                    browser.close()
            except Exception:
                print((work / 'servers.log').read_text(), file=sys.stderr)
                raise
            finally:
                frontend.terminate(); backend.terminate()
                frontend.wait(timeout=10); backend.wait(timeout=10)


if __name__ == '__main__':
    main()
