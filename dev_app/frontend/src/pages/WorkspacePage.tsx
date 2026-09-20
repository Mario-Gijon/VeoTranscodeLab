import { useEffect, useRef, useState } from 'react'
import { Accordion, AccordionDetails, AccordionSummary, Alert, Box, Button, Divider, FormControl, InputLabel, LinearProgress, MenuItem, Select, Slider, Stack, TextField, ToggleButton, ToggleButtonGroup, Typography } from '@mui/material'
import { api } from '../api/client'
import ManageFilesDialog from '../components/ManageFilesDialog'
import MatchViewer from '../components/MatchViewer'
import type { ViewerPick, ViewMode } from '../components/MatchViewer'
import { interactiveGroundPoint } from '../components/interactiveCamera'
import type { InteractiveCamera } from '../components/interactiveCamera'
import { appendMeasurement, matchTimestamp, selectedPitchPoints, VIEW_MODES } from '../components/workspaceGeometry'
import PanoramaViewer from '../components/PanoramaViewer'
import PitchSvg from '../components/PitchSvg'
import ValidationTable from '../components/ValidationTable'
import type { Calibration, GroundPoint, MatchWorkspace, MediaItem, RenderedPanorama, ViewContext } from '../types'

interface Props { matchId: string; calibrations: Calibration[]; onBack: () => void }

export default function WorkspacePage({ matchId, calibrations, onBack }: Props) {
  const [match, setMatch] = useState<MatchWorkspace | null>(null)
  const [context, setContext] = useState<ViewContext | null>(null)
  const [mode, setMode] = useState<ViewMode>('panorama')
  const [crop, setCrop] = useState('stacked')
  const [sourceId, setSourceId] = useState('')
  const [time, setTime] = useState(0)
  const [playing, setPlaying] = useState(false)
  const [width, setWidth] = useState(1920)
  const [stillImage, setStillImage] = useState<MediaItem | null>(null)
  const [frameRender, setFrameRender] = useState<RenderedPanorama | null>(null)
  const [points, setPoints] = useState<GroundPoint[]>([])
  const [landmark, setLandmark] = useState('')
  const [filesOpen, setFilesOpen] = useState(false)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const video = useRef<HTMLVideoElement>(null)
  const interactiveCamera = useRef<InteractiveCamera | null>(null)
  const measurementSequence = useRef(0)
  const generation = useRef(0)
  const selectedCalibrationPath = match?.calibration?.path
  const loadedMatchId = match?.id
  const directionsId = match?.media.find((item) => item.role === 'camera_directions')?.id
  const reload = async () => { const result = await api.match(matchId); setMatch(result); return result }
  useEffect(() => {
    let active = true
    api.match(matchId).then((result) => { if (active) { setMatch(result); if (!result.media.some((item) => item.role === 'dual_lens') && result.media.some((item) => item.role === 'tracking')) setMode('tracking') } }).catch((e: Error) => setError(e.message))
    return () => { active = false }
  }, [matchId])
  useEffect(() => {
    let active = true
    setContext(null); setFrameRender(null); setPoints([]); interactiveCamera.current = null; generation.current++
    if (selectedCalibrationPath && loadedMatchId) api.viewContext(loadedMatchId).then((result) => { if (active) setContext(result) }).catch((e: Error) => { if (active) setError(e.message) })
    return () => { active = false }
  }, [loadedMatchId, selectedCalibrationPath, directionsId])
  const run = async (work: () => Promise<unknown>) => {
    setBusy(true); setError('')
    try { await work() } catch (reason) { setError(reason instanceof Error ? reason.message : 'Operation failed') }
    finally { setBusy(false) }
  }
  const roleMedia = match?.media.filter((item) => mode === 'tracking' ? item.role === 'tracking' || item.role === 'clip' : item.role === 'dual_lens') ?? []
  const source = roleMedia.find((item) => item.id === sourceId) ?? roleMedia[0]
  const duration = source?.duration_s ?? 0
  const playbackDisabled = !source || !!frameRender || !!stillImage
  const seek = (value: number) => {
    const next = Math.max(0, Math.min(value, Math.max(0, duration - .001)))
    if (video.current) video.current.currentTime = next
    setTime(next)
  }
  const switchView = (value: ViewMode) => {
    setTime(matchTimestamp(video.current, time))
    generation.current++; setMode(value); setFrameRender(null); setStillImage(null)
    if ((value === 'tracking') !== (mode === 'tracking')) { video.current?.pause(); setPlaying(false) }
  }
  const renderImage = async (item: MediaItem) => {
    video.current?.pause(); setPlaying(false); generation.current++
    const rendered = await api.render(matchId, item.id, width, width * 9 / 16)
    setFrameRender({ ...rendered, timestamp_s: item.timestamp_s }); setStillImage(null); setMode('panorama'); setPoints([])
  }
  const measure = async (x: number, y: number, pick?: ViewerPick) => {
    if (!context) return
    const currentGeneration = generation.current
    const sequence = ++measurementSequence.current
    const timestamp = pick?.timestamp ?? (frameRender ? frameRender.timestamp_s ?? time : matchTimestamp(video.current, time))
    try {
      const result = mode === 'interactive'
        ? (pick?.camera ? interactiveGroundPoint(context, pick.camera, x, y, pick.width, pick.height) : null)
        : mode === 'tracking'
        ? (pick ? await api.trackingPoint(matchId, timestamp, x, y, pick.width, pick.height) : null)
        : frameRender
        ? await api.groundPoint(frameRender.id, x, y, landmark || null)
        : await api.viewPoint(context.id, mode, x, y, mode === 'dual_lens' ? 2048 : width, mode === 'dual_lens' ? 2048 : width * 9 / 16, landmark || null)
      if (!result || generation.current !== currentGeneration || sequence !== measurementSequence.current) return
      setPoints((items) => appendMeasurement(items, { ...result, timestamp_s: timestamp, view: mode, view_width: width, media_id: frameRender?.id ?? source?.id }))
    } catch (e) { if (mode !== 'tracking' && mode !== 'interactive') setError(e instanceof Error ? e.message : 'Measurement failed') }
  }
  if (!match) return <Typography sx={{ p: 3 }}>{error || 'Loading match…'}</Typography>
  const displayPoints = points.filter((point) => point.view === mode && point.media_id === (frameRender?.id ?? source?.id) && (mode === 'tracking' || point.view_width === width))
  return <Box sx={{ p: { xs: 1, md: 2 } }}>
    <Stack direction="row" alignItems="center" gap={2} mb={1}><Button onClick={onBack}>← Library</Button><Typography variant="h6">{match.name}</Typography><Box flex={1} /><Button variant="outlined" size="small" onClick={() => setFilesOpen(true)}>Manage files</Button></Stack>
    {error && <Alert severity="error" onClose={() => setError('')}>{error}</Alert>}{busy && <LinearProgress />}
    {filesOpen && <ManageFilesDialog match={match} calibrations={calibrations} onClose={() => setFilesOpen(false)} onUpdated={(result) => { video.current?.pause(); setPlaying(false); setFrameRender(null); setStillImage(null); setMatch(result); generation.current++ }} onImage={(item) => { video.current?.pause(); setPlaying(false); setStillImage(item); setFrameRender(null); setFilesOpen(false) }} onRender={(item) => { setFilesOpen(false); void run(() => renderImage(item)) }} />}
    <Stack direction="row" alignItems="center" flexWrap="wrap" gap={1} mb={1}>
      <ToggleButtonGroup size="small" exclusive value={mode} onChange={(_, value: ViewMode | null) => value && switchView(value)}>{VIEW_MODES.map((view) => <ToggleButton key={view} value={view}>{view === 'dual_lens' ? 'Dual lens' : view[0].toUpperCase() + view.slice(1)}</ToggleButton>)}</ToggleButtonGroup>
      {roleMedia.length > 1 && <Select size="small" value={source?.id ?? ''} displayEmpty onChange={(e) => { video.current?.pause(); setPlaying(false); setSourceId(e.target.value); setFrameRender(null); setStillImage(null); generation.current++ }}><MenuItem value="">No source assigned</MenuItem>{roleMedia.map((item) => <MenuItem key={item.id} value={item.id}>{item.name}</MenuItem>)}</Select>}
      {mode === 'dual_lens' && <ToggleButtonGroup size="small" exclusive value={crop} onChange={(_, value: string | null) => value && setCrop(value)}><ToggleButton value="stacked">Stacked</ToggleButton><ToggleButton value="left">LEFT</ToggleButton><ToggleButton value="right">RIGHT</ToggleButton></ToggleButtonGroup>}
      {mode === 'panorama' && <Select size="small" value={width} onChange={(e) => { setWidth(Number(e.target.value)); setFrameRender(null); generation.current++ }}><MenuItem value={1920}>1920×1080</MenuItem><MenuItem value={3840}>3840×2160</MenuItem></Select>}
    </Stack>
    <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', lg: 'minmax(0, 7fr) minmax(280px, 3fr)' }, gap: 2 }}>
      <Box minWidth={0}>
        {stillImage ? <Box><img src={api.url(stillImage.url)} alt={stillImage.name} style={{ width: '100%', maxHeight: '65vh', objectFit: 'contain' }} /><Typography variant="caption">{stillImage.name} · Image preview. Use Render frame for a calibrated stacked frame.</Typography><Button onClick={() => setStillImage(null)}>Return to video</Button></Box> : frameRender ? <PanoramaViewer panorama={frameRender} points={displayPoints} onPixelClick={(x, y) => void measure(x, y)} /> : source && (!['panorama', 'interactive'].includes(mode) || context) ? <MatchViewer key={source.id} media={source} mode={mode} crop={crop} context={context} width={width} initialTime={time} videoRef={video} cameraRef={interactiveCamera} onTime={setTime} onPlayback={setPlaying} points={displayPoints} trackingAvailable={!!context?.tracking_geometry.available} onClick={(x, y, pick) => void measure(x, y, pick)} /> : <Alert severity="info">{['panorama', 'interactive'].includes(mode) && !context ? 'Add a calibration to enable this view.' : 'Add the source video in Manage files to use this view.'}</Alert>}
        <Stack direction="row" alignItems="center" gap={1} mt={1} flexWrap="wrap">
          <Button disabled={playbackDisabled} onClick={() => { if (video.current?.paused) void video.current.play().catch((e: Error) => setError(e.message)); else video.current?.pause() }}>{playing ? 'Pause' : 'Play'}</Button>
          <Button disabled={playbackDisabled} onClick={() => { video.current?.pause(); seek(time - 1 / (source?.fps || 30)) }}>−1 frame</Button><Button disabled={playbackDisabled} onClick={() => { video.current?.pause(); seek(time + 1 / (source?.fps || 30)) }}>+1 frame</Button>
          <TextField label="Time (s)" size="small" type="number" value={Number(time.toFixed(3))} disabled={playbackDisabled} onChange={(e) => seek(Number(e.target.value))} sx={{ width: 120 }} /><Typography variant="caption">/ {duration.toFixed(1)} s</Typography>
          <Button disabled={busy || playbackDisabled} onClick={() => void run(async () => { video.current?.pause(); const frame = await api.extractFrame(source!.id, video.current?.currentTime ?? time); await reload(); if (context && source?.role === 'dual_lens') await renderImage(frame) })}>Extract frame / CPU render</Button>
          {frameRender && <Button onClick={() => { setFrameRender(null); generation.current++ }}>Return to video</Button>}
        </Stack><Slider aria-label="Match timeline" value={Math.min(time, duration)} max={duration || 1} step={source?.fps ? 1 / source.fps : .04} disabled={playbackDisabled} onChange={(_, value) => seek(value as number)} />
      </Box>
      <Box minWidth={0}><Typography variant="subtitle1">Pitch</Typography>{context ? <PitchSvg length={context.field_length} width={context.field_width} points={selectedPitchPoints(points)} /> : <Alert severity="info">Pitch dimensions become available after selecting a calibration.</Alert>}
        {mode === 'tracking' || mode === 'interactive' ? <Typography variant="caption">{mode === 'tracking' && !context?.tracking_geometry.available ? 'Pitch selection is unavailable for this video.' : 'Click the ground to move the shared pitch marker.'}</Typography> : <Stack spacing={1} mt={1}><FormControl size="small"><InputLabel shrink>Click mode / landmark</InputLabel><Select displayEmpty renderValue={(value) => value ? context?.landmarks.find((item) => item.id === value)?.name : 'Free point'} label="Click mode / landmark" value={landmark} onChange={(e) => setLandmark(e.target.value)}><MenuItem value="">Free point</MenuItem>{context?.landmarks.map((item) => <MenuItem key={item.id} value={item.id}>{item.name} · {item.x.toFixed(2)}, {item.z.toFixed(2)} m</MenuItem>)}</Select></FormControl><Typography variant="caption">Validation measures calibration error; it does not correct the geometry.</Typography></Stack>}
        <Button onClick={() => { setPoints([]); generation.current++ }}>Clear points</Button>
      </Box>
    </Box>{points.length > 0 && <><Divider sx={{ my: 1 }} /><Accordion disableGutters><AccordionSummary>Measurements · {points.length} points</AccordionSummary><AccordionDetails><ValidationTable points={points} /></AccordionDetails></Accordion></>}
  </Box>
}
