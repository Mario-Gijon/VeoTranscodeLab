import type { Calibration, GroundPoint, MatchWorkspace, MediaItem, RenderedPanorama } from '../types'

const baseUrl = import.meta.env.VITE_API_BASE_URL ?? 'http://127.0.0.1:8000'
const url = (path: string) => `${baseUrl}${path}`

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(url(path), init)
  if (!response.ok) {
    const body = await response.json().catch(() => ({})) as { detail?: { message?: string } | string }
    const detail = typeof body.detail === 'object' ? body.detail?.message : body.detail
    throw new Error(detail || `Request failed (${response.status}).`)
  }
  return response.json() as Promise<T>
}

export const api = {
  url,
  health: () => request<{ status: string }>('/api/health'),
  calibrations: () => request<Calibration[]>('/api/calibrations'),
  matches: () => request<MatchWorkspace[]>('/api/matches'),
  match: (id: string) => request<MatchWorkspace>(`/api/matches/${id}`),
  createMatch: (name: string, description: string, calibrationId: string | null) => request<MatchWorkspace>('/api/matches', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ name, description, calibration_id: calibrationId }) }),
  importCalibrationPath: (matchId: string, path: string) => request<MatchWorkspace>(`/api/matches/${matchId}/calibration/path`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ path }) }),
  uploadCalibration: (matchId: string, file: File) => { const body = new FormData(); body.append('file', file); return request<MatchWorkspace>(`/api/matches/${matchId}/calibration/upload`, { method: 'POST', body }) },
  importMediaPath: (matchId: string, path: string) => request<MediaItem>(`/api/matches/${matchId}/media/path`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ path }) }),
  uploadMedia: (matchId: string, file: File) => { const body = new FormData(); body.append('file', file); return request<MediaItem>(`/api/matches/${matchId}/media/upload`, { method: 'POST', body }) },
  extractFrame: (mediaId: string, timestamp: number) => request<MediaItem>('/api/media/frame', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ media_id: mediaId, timestamp_s: timestamp }) }),
  render: (matchId: string, mediaId: string, width: number, height: number) => request<RenderedPanorama>('/api/panorama/render', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ match_id: matchId, media_id: mediaId, width, height }) }),
  groundPoint: (renderId: string, x: number, y: number, landmarkId: string | null) => request<GroundPoint>('/api/ground-point', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ render_id: renderId, x, y, landmark_id: landmarkId }) }),
}
