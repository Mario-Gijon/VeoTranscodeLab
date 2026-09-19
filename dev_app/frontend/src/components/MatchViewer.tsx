import { Alert, Box, Typography } from '@mui/material'
import { useEffect, useRef, useState } from 'react'
import type { RefObject } from 'react'
import type { GroundPoint, MediaItem, ViewContext } from '../types'
import { api } from '../api/client'
import { createPanoramaGL } from './panoramaGL'
import { naturalPixel, stackedPixel } from './viewCoordinates'

export type ViewMode = 'tracking' | 'panorama' | 'dual_lens'
interface Props {
  media: MediaItem; mode: ViewMode; crop: string; context: ViewContext | null
  width: number; videoRef: RefObject<HTMLVideoElement | null>; initialTime: number
  onTime: (value: number) => void; onPlayback: (value: boolean) => void
  points: GroundPoint[]; onClick: (x: number, y: number) => void
}

export default function MatchViewer({ media, mode, crop, context, width, videoRef, initialTime, onTime, onPlayback, points, onClick }: Props) {
  const canvas = useRef<HTMLCanvasElement>(null)
  const magnifier = useRef<HTMLCanvasElement>(null)
  const surface = useRef<HTMLDivElement>(null)
  const zoom = useRef(4)
  const pointer = useRef<{x: number; y: number; cx: number; cy: number} | null>(null)
  const [error, setError] = useState('')
  const [hover, setHover] = useState(false)
  const outputWidth = mode === 'panorama' ? width : 2048
  const outputHeight = mode === 'panorama' ? width * 9 / 16 : crop === 'stacked' ? 2048 : 1024
  const pausedAt = useRef(initialTime)
  const sourceUrl = api.url(media.url)
  useEffect(() => {
    const element = surface.current
    if (!element || mode === 'tracking') return
    const wheel = (event: WheelEvent) => { event.preventDefault(); zoom.current = Math.max(2, Math.min(10, zoom.current + (event.deltaY < 0 ? 1 : -1))) }
    element.addEventListener('wheel', wheel, { passive: false })
    return () => element.removeEventListener('wheel', wheel)
  }, [mode])
  useEffect(() => {
    setError('')
    if (mode === 'tracking') return
    const destination = canvas.current
    if (!destination) return
    let gpu: ReturnType<typeof createPanoramaGL> | undefined
    let animation = 0
    let lastFrame = -1
    try {
      if (mode === 'panorama') {
        if (!context) return
        gpu = createPanoramaGL(destination, context)
      }
      const draw = () => {
        try {
        const video = videoRef.current
        if (video && video.readyState >= 2) {
          if (lastFrame !== video.currentTime) {
            if (gpu) gpu.draw(video)
            else destination.getContext('2d')?.drawImage(video, 0, crop === 'right' ? 1024 : 0, 2048, outputHeight, 0, 0, outputWidth, outputHeight)
            lastFrame = video.currentTime
          }
          const cursor = pointer.current
          const glass = magnifier.current
          const ctx = glass?.getContext('2d')
          if (cursor && ctx && glass) {
            ctx.fillStyle = '#080a0c'; ctx.fillRect(0, 0, 180, 180)
            ctx.imageSmoothingEnabled = false
            ctx.drawImage(destination, 90 - (cursor.x + .5) * zoom.current, 90 - (cursor.y + .5) * zoom.current, outputWidth * zoom.current, outputHeight * zoom.current)
            ctx.strokeStyle = 'white'; ctx.lineWidth = 1
            ctx.beginPath(); ctx.moveTo(90, 0); ctx.lineTo(90, 180); ctx.moveTo(0, 90); ctx.lineTo(180, 90); ctx.stroke()
            ctx.fillStyle = '#101216'; ctx.fillRect(0, 158, 180, 22)
            ctx.fillStyle = 'white'; ctx.font = '12px sans-serif'; ctx.fillText(`${zoom.current}× · ${cursor.x}, ${cursor.y}`, 6, 173)
          }
        }
        animation = requestAnimationFrame(draw)
        } catch (reason) {
          videoRef.current?.pause()
          setError(reason instanceof Error ? reason.message : 'Cannot draw this video frame')
        }
      }
      animation = requestAnimationFrame(draw)
    } catch (reason) { setError(reason instanceof Error ? reason.message : 'Cannot initialize viewer') }
    return () => { cancelAnimationFrame(animation); gpu?.dispose() }
  }, [mode, crop, context, outputWidth, outputHeight, videoRef, sourceUrl])

  return <Box>
    {error && <Alert severity="error">{error} CPU frame rendering remains available.</Alert>}
    <video ref={videoRef} crossOrigin="anonymous" src={sourceUrl} preload="metadata"
      style={{ display: mode === 'tracking' ? 'block' : 'none', width: '100%', maxHeight: '65vh', background: '#080a0c' }}
      onLoadedMetadata={(event) => { event.currentTarget.currentTime = Math.min(pausedAt.current, Math.max(0, event.currentTarget.duration - .001)) }}
      onTimeUpdate={(event) => onTime(event.currentTarget.currentTime)}
      onPlay={() => onPlayback(true)} onPause={() => onPlayback(false)} onEnded={() => onPlayback(false)}
      onError={() => setError('The browser cannot play this source codec. The original file is retained; try extracting a frame.')} />
    {mode !== 'tracking' && <Box ref={surface} sx={{ position: 'relative', width: '100%', cursor: 'crosshair', lineHeight: 0 }}
      onPointerMove={(event) => {
        const bounds = event.currentTarget.getBoundingClientRect()
        pointer.current = { ...naturalPixel(event.clientX, event.clientY, bounds, outputWidth, outputHeight), cx: event.clientX - bounds.left, cy: event.clientY - bounds.top }
        if (magnifier.current) {
          magnifier.current.style.left = `${Math.max(0, Math.min(bounds.width - 180, pointer.current.cx + 20))}px`
          magnifier.current.style.top = `${Math.max(0, Math.min(bounds.height - 180, pointer.current.cy + 20))}px`
        }
        setHover(true)
      }} onPointerLeave={() => { pointer.current = null; setHover(false) }}
      onClick={(event) => {
        if (videoRef.current?.seeking || !videoRef.current || videoRef.current.readyState < 2 || error) return
        videoRef.current.pause()
        const pixel = naturalPixel(event.clientX, event.clientY, event.currentTarget.getBoundingClientRect(), outputWidth, outputHeight)
        const source = mode === 'dual_lens' ? stackedPixel(pixel.x, pixel.y, crop) : pixel
        onClick(source.x, source.y)
      }}>
      <canvas key={mode} ref={canvas} width={outputWidth} height={outputHeight} style={{ width: '100%', display: 'block', background: '#080a0c' }} />
      <svg viewBox={`0 0 ${outputWidth} ${outputHeight}`} style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', pointerEvents: 'none' }}>
        {points.map((point) => {
          const y = point.pixel_y - (mode === 'dual_lens' && crop === 'right' ? 1024 : 0)
          if (y < 0 || y >= outputHeight) return null
          return <g key={point.point_index}><circle cx={point.pixel_x + .5} cy={y + .5} r={outputWidth / 140} fill="#ffd258" stroke="#000" /><text x={point.pixel_x + .5} y={y + .5} textAnchor="middle" dominantBaseline="central" fontSize={outputWidth / 90} fill="#000">{point.point_index}</text></g>
        })}
      </svg>
      <canvas ref={magnifier} width={180} height={180} style={{ display: hover ? 'block' : 'none', position: 'absolute', pointerEvents: 'none', border: '1px solid #eee', width: 180, height: 180 }} />
    </Box>}
    <Typography variant="caption" color="text.secondary">{mode === 'tracking' ? 'Tracking playback — field mapping requires the missing time-varying virtual-camera transform.' : 'Click pauses playback to measure the displayed pixel. Wheel: magnifier 2×–10×.'}</Typography>
  </Box>
}
