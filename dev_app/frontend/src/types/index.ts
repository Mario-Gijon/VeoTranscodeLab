export type MediaKind = 'image' | 'frame' | 'video' | 'clip'

export interface ViewContext {
  id: string; field_length: number; field_width: number; landmarks: Landmark[]
  uniforms: Record<string, { matrix?: number; value: number | number[] }>
  vertex_shader: string; fragment_shader: string
  tracking_geometry: { available: boolean; reason: string }
}

export interface Calibration {
  version?: string | number | null
  id: string
  name: string
  path: string
  source: string
  field_length: number
  field_width: number
}

export interface MediaItem {
  role?: 'tracking' | 'dual_lens' | 'clip' | 'image' | 'unknown'
  size_bytes?: number
  id: string
  name: string
  path: string
  source: string
  kind: MediaKind
  width: number
  height: number
  fps: number | null
  duration_s: number | null
  frame_count?: number
  timestamp_s?: number
  extracted_from?: string
  url: string
}

export interface MatchWorkspace {
  metadata?: Record<string, string | null>
  id: string
  name: string
  description: string
  created_at: string
  updated_at: string
  calibration: Calibration | null
  media: MediaItem[]
  media_count?: number
}

export interface Landmark { id: string; name: string; x: number; z: number }

export interface RenderedPanorama {
  timestamp_s?: number
  id: string
  url: string
  width: number
  height: number
  field_length: number
  field_width: number
  landmarks: Landmark[]
}

export interface GroundPoint {
  view?: string
  timestamp_s?: number
  media_id?: string
  view_width?: number
  pixel_x: number
  pixel_y: number
  u: number
  v: number
  world_x: number | null
  world_z: number | null
  distance: number | null
  inside_field: boolean | null
  status: 'inside_field' | 'outside_field' | 'no_ground_intersection'
  landmark: string | null
  landmark_name: string | null
  expected_x: number | null
  expected_z: number | null
  measured_x: number | null
  measured_z: number | null
  delta_x: number | null
  delta_z: number | null
  field_error_m: number | null
  point_index?: number
}

export interface VeoCandidate { calibration_version?: string | null; identifier: string; safe_url: string; filename: string; content_type: string | null; role: string; confidence: number; reasons: string[]; width: number | null; height: number | null; fps: number | null; duration_s: number | null; field_length: number | null; field_width: number | null }
export interface VeoInspection { id: string; match_url: string; metadata: Record<string, string | null>; candidates: VeoCandidate[]; diagnostic_path: string | null }
export interface VeoSession { error?: string | null; playwright_available: boolean; profile_exists: boolean; connect_browser_running: boolean }
export interface VeoJob { progress?: number; completed_assets?: number; identifier: string; status: string; current_asset: string | null; bytes_downloaded: number; bytes_total: number | null; error: string | null; match_id: string | null }
