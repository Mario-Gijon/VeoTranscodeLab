import { test } from 'node:test'
import assert from 'node:assert/strict'
import { naturalPixel, stackedPixel } from '../src/components/viewCoordinates.ts'

test('responsive clicks select the same natural pixel center', () => {
  for (const scale of [.25, .5, 1, 1.7]) {
    const bounds = { left: 20, top: 50, width: 3840 * scale, height: 2160 * scale }
    assert.deepEqual(naturalPixel(20 + 2345.5 * scale, 50 + 1456.5 * scale, bounds, 3840, 2160), {x: 2345, y: 1456})
  }
})
test('edges clamp and crop clicks use stacked coordinates', () => {
  assert.deepEqual(naturalPixel(100, 100, {left: 0, top: 0, width: 100, height: 100}, 2048, 2048), {x: 2047, y: 2047})
  assert.deepEqual(stackedPixel(100, 500, 'left'), {x: 100, y: 500})
  assert.deepEqual(stackedPixel(100, 500, 'right'), {x: 100, y: 1524})
})
