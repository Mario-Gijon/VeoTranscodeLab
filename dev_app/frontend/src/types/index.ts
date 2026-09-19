export type MediaKind = 'image' | 'frame' | 'video' | 'clip'

export interface Calibration {
  id: string
  name: string
  path: string
  source: string
  field_length: number
  field_width: number
}

export interface MediaItem {
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
  id: string
  url: string
  width: number
  height: number
  field_length: number
  field_width: number
  landmarks: Landmark[]
}

export interface GroundPoint {
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
