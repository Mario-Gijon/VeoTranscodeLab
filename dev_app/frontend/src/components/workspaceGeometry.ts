import type { GroundPoint } from '../types'

export const VIEW_MODES = ['tracking', 'interactive', 'panorama', 'dual_lens'] as const
export type ViewMode = typeof VIEW_MODES[number]

/** The media element is the Workspace clock, not React's timeupdate cadence. */
export function matchTimestamp(video: { currentTime: number } | null, fallback: number) {
  return video && Number.isFinite(video.currentTime) ? video.currentTime : fallback
}

export function validPitchPoint(point: GroundPoint) {
  return point.measured_x !== null && point.measured_z !== null
    && Number.isFinite(point.measured_x) && Number.isFinite(point.measured_z)
    && (point.status === 'inside_field' || point.status === 'outside_field')
}

/** Keep validation history, but show only the latest valid canonical selection. */
export function selectedPitchPoints(points: GroundPoint[]) {
  return points.filter(validPitchPoint).slice(-1)
}

export function appendMeasurement(points: GroundPoint[], point: GroundPoint) {
  if ((point.view === 'tracking' || point.view === 'interactive')
    && (!validPitchPoint(point) || point.status !== 'inside_field')) return points
  return [...points, { ...point, point_index: points.length + 1 }]
}
