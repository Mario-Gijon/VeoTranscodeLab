import { multiplyMatrix, projectGroundPoint } from './interactiveCamera.ts'
import type { ViewContext } from '../types'

type Sample = [number, number, number, number]
const rotateY = (angle: number) => { const c = Math.cos(angle), s = Math.sin(angle); return [c, 0, -s, 0, 0, 1, 0, 0, s, 0, c, 0, 0, 0, 0, 1] }
const rotateX = (angle: number) => { const c = Math.cos(angle), s = Math.sin(angle); return [1, 0, 0, 0, 0, c, s, 0, 0, -s, c, 0, 0, 0, 0, 1] }

export function parseCameraDirections(text: string): Sample[] {
  return text.split(/\r?\n/).flatMap(line => {
    const row = line.split(',').map(Number)
    return row.length === 4 && row.every(Number.isFinite) ? [row as Sample] : []
  })
}

export function cameraDirectionAtTime(samples: Sample[], time: number, fps: number): Sample | null {
  const index = Math.max(0, Math.floor(time * fps + .5) - 1)
  return Number.isFinite(time) && time >= 0 && index < samples.length ? samples[index] : null
}

/** Exact forward inverse of tracking_ground_v3; samples are loaded once per asset. */
export function projectTrackingGroundPoint(context: ViewContext, samples: Sample[], time: number, x: number, z: number) {
  const geometry = context.tracking_geometry
  const sample = cameraDirectionAtTime(samples, time, geometry.fps ?? 30000 / 1001)
  if (!sample || !geometry.logic2camera) return null
  const [yaw, pitch, fov] = sample
  const transform = multiplyMatrix(multiplyMatrix(context.uniforms.camera2world.value as number[], geometry.logic2camera), multiplyMatrix(rotateY(yaw), rotateX(pitch)))
  return projectGroundPoint(transform, fov, geometry.aspect ?? 16 / 9, x, z)
}
