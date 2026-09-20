import { test } from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { dragCamera, initialCamera, interactiveGroundPoint, interactiveRay, interactiveShader, isDrag, MAX_FOV, MIN_FOV, PITCH_LIMIT, safeCamera, wheelCamera } from '../src/components/interactiveCamera.ts'
import { appendMeasurement, canSelect, matchTimestamp, selectedPitchPoints, VIEW_MODES } from '../src/components/workspaceGeometry.ts'
import { cameraDirectionAtTime, parseCameraDirections, projectTrackingGroundPoint } from '../src/components/trackingOverlay.ts'
import { projectGroundPoint, interactiveTransform } from '../src/components/interactiveCamera.ts'
import { nativeVideoPixel } from '../src/components/viewCoordinates.ts'

const identity = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
const world = [...identity]; world[13] = 10
const context = { uniforms: { camera2world: { value: world }, camera2presentation: { value: identity },
  va_left: { value: -.8 }, va_right: { value: .8 }, va_bottom: { value: .2 }, va_top: { value: .8 } }, field_length: 105, field_width: 68 }
const camera = { longitude: 0, latitude: Math.PI / 4, fov: Math.PI / 3 }
const close = (a, b) => assert.ok(Math.abs(a - b) < 1e-10, `${a} != ${b}`)

test('four independent modes retain the media clock, not stale React time', () => {
  assert.deepEqual(VIEW_MODES, ['tracking', 'interactive', 'panorama', 'dual_lens'])
  for (const _mode of VIEW_MODES) close(matchTimestamp({ currentTime: 71.23456 }, 70), 71.23456)
  close(matchTimestamp(null, 71.23456), 71.23456)
})
test('camera initializes from presentation bounds, clamps tilt/zoom, wraps yaw and prevents NaN', () => {
  assert.deepEqual(initialCamera(context), { longitude: 0, latitude: .5, fov: Math.PI / 3 })
  const extreme = safeCamera({ longitude: 1e20, latitude: 1e20, fov: 1e20 })
  assert.ok(Math.abs(extreme.longitude) <= Math.PI)
  assert.equal(extreme.latitude, PITCH_LIMIT); assert.equal(extreme.fov, MAX_FOV)
  assert.ok(Object.values(safeCamera({ longitude: NaN, latitude: Infinity, fov: NaN })).every(Number.isFinite))
})
test('horizontal drag rotates, vertical drag clamps, small movement remains a click', () => {
  const horizontal = dragCamera(camera, 100, 0, 800, 450)
  assert.ok(horizontal.longitude < camera.longitude); assert.equal(horizontal.latitude, camera.latitude)
  assert.equal(dragCamera(camera, 0, 1e8, 800, 450).latitude, PITCH_LIMIT)
  assert.equal(dragCamera(camera, 0, -1e8, 800, 450).latitude, -PITCH_LIMIT)
  assert.equal(isDrag(3, 3), false); assert.equal(isDrag(6, 0), true); assert.equal(isDrag(0, -6), true)
})
test('wheel/trackpad directly changes FOV with finite limits', () => {
  assert.ok(wheelCamera(camera, -100).fov < camera.fov)
  assert.ok(wheelCamera(camera, 100).fov > camera.fov)
  let narrow = camera, wide = camera
  for (let i = 0; i < 100; i++) { narrow = wheelCamera(narrow, -1000); wide = wheelCamera(wide, 1000) }
  assert.equal(narrow.fov, MIN_FOV); assert.equal(wide.fov, MAX_FOV)
})
test('pixel -> perspective ray -> world -> ground -> canonical X/Z', () => {
  const ray = interactiveRay(context, camera, 50, 50, 101, 101)
  assert.deepEqual(ray.origin, [0, 10, 0])
  close(ray.direction[0], 0); close(ray.direction[1], -Math.SQRT1_2); close(ray.direction[2], Math.SQRT1_2)
  const point = interactiveGroundPoint(context, camera, 50, 50, 101, 101)
  close(point.world_x, 0); close(point.world_z, 10); assert.equal(point.status, 'inside_field')
})
test('current longitude, latitude, FOV, aspect and calibration all affect projection', () => {
  const point = (cam, x = 50) => interactiveGroundPoint(context, cam, x, 50, 101, 101)
  const yawed = point({ ...camera, longitude: .4 })
  close(yawed.world_x, 10 * Math.sin(.4)); close(yawed.world_z, 10 * Math.cos(.4))
  close(point({ ...camera, latitude: Math.PI / 3 }).world_z, 10 / Math.sqrt(3))
  assert.notEqual(point(camera, 70).world_x, point({ ...camera, fov: .6 }, 70).world_x)
  assert.notEqual(interactiveRay(context, camera, 70, 50, 101, 101).direction[0], interactiveRay(context, camera, 140, 50, 201, 101).direction[0])
  const rotated = { ...context, uniforms: { ...context.uniforms, camera2presentation: { value: [0, 0, -1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1] } } }
  close(interactiveGroundPoint(rotated, camera, 50, 50, 101, 101).world_x, 10)
})
test('sky, parallel, behind-camera, out-of-field and nonfinite states do not produce points', () => {
  for (const cam of [{ ...camera, latitude: -.5 }, { ...camera, latitude: 0 }, { ...camera, latitude: .001 }, { ...camera, fov: 0 }, { ...camera, fov: NaN }, { ...camera, longitude: Infinity }]) {
    assert.equal(interactiveGroundPoint(context, cam, 50, 50, 101, 101), null)
  }
  assert.equal(interactiveGroundPoint(context, camera, -1, 50, 101, 101), null)
})
test('responsive/DPR pixel-center mapping preserves Interactive world coordinates', () => {
  const a = interactiveGroundPoint(context, camera, 65, 55, 101, 101)
  const b = interactiveGroundPoint(context, camera, 196, 166, 303, 303)
  close(a.world_x, b.world_x); close(a.world_z, b.world_z)
  for (const scale of [.4, 1, 2]) {
    const bounds = { left: 20, top: 30, width: 303 * scale, height: 303 * scale }
    assert.deepEqual(nativeVideoPixel(20 + 196.5 * scale, 30 + 166.5 * scale, bounds, 303, 303), { x: 196, y: 166 })
  }
})
test('both new views replace the shared marker and reject invalid Tracking results', () => {
  const point = interactiveGroundPoint(context, camera, 50, 50, 101, 101)
  let history = appendMeasurement([], { ...point, view: 'interactive', timestamp_s: 1.234 })
  history = appendMeasurement(history, { ...point, measured_x: 4, world_x: 4, view: 'tracking', timestamp_s: 5.678 })
  assert.equal(history.length, 2); assert.equal(selectedPitchPoints(history).length, 1)
  assert.equal(selectedPitchPoints(history)[0].view, 'tracking')
  for (const _view of VIEW_MODES) assert.equal(selectedPitchPoints(history)[0].measured_x, 4)
  for (const invalid of [{ status: 'camera_state_unavailable', measured_x: null }, { status: 'outside_field' }, { measured_z: NaN }]) {
    assert.equal(appendMeasurement(history, { ...point, ...invalid, view: 'tracking' }), history)
  }
  for (const view of ['panorama', 'dual_lens']) {
    history = appendMeasurement(history, { ...point, view })
    assert.equal(selectedPitchPoints(history)[0].view, view)
  }
})
test('Interactive shader retains captured Panorama branch and all lens projection code', () => {
  const source = readFileSync(new URL('../../../reference/veo_panoramic.frag.glsl', import.meta.url), 'utf8')
  const extended = interactiveShader(source)
  const start = source.indexOf('  // *** Calculation for PANORAMA ONLY *** //')
  const end = source.indexOf('  // *** END *** //') + '  // *** END *** //'.length
  assert.ok(extended.includes(source.slice(start, end)))
  assert.ok(extended.endsWith(source.slice(source.indexOf('  blend_target_point = inverse'))))
  assert.match(extended, /interactive_enabled > 0.5 && abs\(worldspace_projection_dir.y\) < 0.000001/)
  assert.match(extended, /interactive_to_world \* vec4\(view_ray, 0.0\)/)
})
test('shared Select gate preserves pitch history while disabling visual selection in every view', () => {
  const point = interactiveGroundPoint(context, camera, 50, 50, 101, 101)
  const history = appendMeasurement([], { ...point, view: 'panorama' })
  for (const view of VIEW_MODES) {
    assert.equal(canSelect(false, true), false, view)
    assert.equal(canSelect(true, false), false, view)
    assert.equal(selectedPitchPoints(history)[0].world_z, point.world_z, view)
  }
  assert.equal(canSelect(true, true), true)
})
test('Interactive world anchor forward-projection follows camera movement, zoom and visibility', () => {
  const stable = { measured_x: 0, measured_z: 10 }
  const initial = projectGroundPoint(interactiveTransform(context, camera), camera.fov, 16 / 9, stable.measured_x, stable.measured_z)
  close(initial.u, .5); close(initial.v, .5)
  const panned = projectGroundPoint(interactiveTransform(context, { ...camera, longitude: .2 }), camera.fov, 16 / 9, stable.measured_x, stable.measured_z)
  assert.ok(panned.u < .5)
  const zoomed = projectGroundPoint(interactiveTransform(context, camera), .5, 16 / 9, 5, 10)
  const wide = projectGroundPoint(interactiveTransform(context, camera), 1.2, 16 / 9, 5, 10)
  assert.ok(zoomed.u > wide.u)
  assert.equal(projectGroundPoint(interactiveTransform(context, { ...camera, longitude: Math.PI }), camera.fov, 16 / 9, 0, 10), null)
})
test('Tracking forward projection uses current .det row and hides outside the viewport', () => {
  const samples = parseCameraDirections('0,0.7853981633974483,1,0\n0.4,0.7853981633974483,1,0')
  assert.deepEqual(cameraDirectionAtTime(samples, 0, 30), samples[0]); assert.deepEqual(cameraDirectionAtTime(samples, 2 / 30, 30), samples[1])
  const trackingContext = { ...context, tracking_geometry: { available: true, reason: '', logic2camera: identity, fps: 30, aspect: 16 / 9 } }
  const first = projectTrackingGroundPoint(trackingContext, samples, 0, 0, 10)
  const second = projectTrackingGroundPoint(trackingContext, samples, 2 / 30, 0, 10)
  close(first.u, .5); assert.notEqual(first.u, second.u)
  assert.equal(projectTrackingGroundPoint(trackingContext, samples, 0, 1000, 10), null)
})
