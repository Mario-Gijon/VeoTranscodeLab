import { useEffect, useRef, useState } from 'react'
import { Accordion, AccordionDetails, AccordionSummary, Alert, Box, Button, Divider, FormControl, InputLabel, LinearProgress, MenuItem, Select, Slider, Stack, TextField, ToggleButton, ToggleButtonGroup, Typography } from '@mui/material'
import { api } from '../api/client'
import MatchViewer from '../components/MatchViewer'
import type { ViewMode } from '../components/MatchViewer'
import PanoramaViewer from '../components/PanoramaViewer'
import PitchSvg from '../components/PitchSvg'
import ValidationTable from '../components/ValidationTable'
import type { Calibration, GroundPoint, MatchWorkspace, MediaItem, RenderedPanorama, ViewContext } from '../types'

interface Props { matchId: string; calibrations: Calibration[]; onBack: () => void }
const isVideo = (item: MediaItem) => ['video', 'clip'].includes(item.kind)

export default function WorkspacePage({ matchId, calibrations, onBack }: Props) {
  const [match, setMatch] = useState<MatchWorkspace | null>(null)
  const [context, setContext] = useState<ViewContext | null>(null)
  const [mode, setMode] = useState<ViewMode>('panorama')
  const [crop, setCrop] = useState('left')
  const [sourceId, setSourceId] = useState('')
  const [time, setTime] = useState(0)
  const [playing, setPlaying] = useState(false)
  const [width, setWidth] = useState(1920)
  const [stillImage, setStillImage] = useState<MediaItem | null>(null)
  const [frameRender, setFrameRender] = useState<RenderedPanorama | null>(null)
  const [points, setPoints] = useState<GroundPoint[]>([])
  const [landmark, setLandmark] = useState('')
  const [path, setPath] = useState('')
  const [calibrationPath, setCalibrationPath] = useState('')
  const [role, setRole] = useState('dual_lens')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const video = useRef<HTMLVideoElement>(null)
  const generation = useRef(0)
  const selectedCalibrationPath = match?.calibration?.path
  const loadedMatchId = match?.id
  const reload = async () => { const result = await api.match(matchId); setMatch(result); return result }
  useEffect(() => {
    let active = true
    api.match(matchId).then((result) => { if (active) setMatch(result) }).catch((e: Error) => setError(e.message))
    return () => { active = false }
  }, [matchId])
  useEffect(() => {
    let active = true
    setContext(null); setFrameRender(null); setPoints([]); generation.current++
    if (selectedCalibrationPath && loadedMatchId) api.viewContext(loadedMatchId).then((result) => { if (active) setContext(result) }).catch((e: Error) => { if (active) setError(e.message) })
    return () => { active = false }
  }, [loadedMatchId, selectedCalibrationPath])
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
    generation.current++; setMode(value); setFrameRender(null); setStillImage(null)
    if ((value === 'tracking') !== (mode === 'tracking')) { video.current?.pause(); setPlaying(false) }
  }
  const addMedia = async (item: MediaItem) => {
    video.current?.pause(); setPlaying(false); setFrameRender(null); generation.current++
    if (isVideo(item)) await api.setRole(item.id, role)
    await reload()
    if (role === 'tracking' || role === 'clip') setMode('tracking')
    setSourceId(item.id)
    setStillImage(isVideo(item) ? null : item)
  }
  const renderImage = async (item: MediaItem) => {
    video.current?.pause(); setPlaying(false); generation.current++
    const rendered = await api.render(matchId, item.id, width, width * 9 / 16)
    setFrameRender({ ...rendered, timestamp_s: item.timestamp_s }); setStillImage(null); setMode('panorama'); setPoints([])
  }
  const measure = async (x: number, y: number) => {
    if (!context || mode === 'tracking') return
    const currentGeneration = generation.current
    const timestamp = frameRender ? frameRender.timestamp_s : video.current?.currentTime ?? time
    try {
      const result = frameRender
        ? await api.groundPoint(frameRender.id, x, y, landmark || null)
        : await api.viewPoint(context.id, mode, x, y, mode === 'dual_lens' ? 2048 : width, mode === 'dual_lens' ? 2048 : width * 9 / 16, landmark || null)
      if (generation.current !== currentGeneration) return
      setPoints((items) => [...items, { ...result, point_index: items.length + 1, timestamp_s: timestamp, view: mode, view_width: width, media_id: frameRender?.id ?? source?.id }])
    } catch (e) { setError(e instanceof Error ? e.message : 'Measurement failed') }
  }
  if (!match) return <Typography sx={{ p: 3 }}>{error || 'Loading match…'}</Typography>
  const displayPoints = points.filter((point) => point.view === mode && point.media_id === (frameRender?.id ?? source?.id) && point.view_width === width)
  const activeCalibrations = [...calibrations]
  if (match.calibration && !activeCalibrations.some((item) => item.path === match.calibration?.path)) activeCalibrations.push(match.calibration)
  return <Box sx={{ p: { xs: 1, md: 2 } }}>
    <Stack direction="row" alignItems="center" gap={2} mb={1}><Button onClick={onBack}>← Library</Button><Typography variant="h6">{match.name}</Typography><Typography variant="body2" color="text.secondary">{match.description}</Typography></Stack>
    {error && <Alert severity="error" onClose={() => setError('')}>{error}</Alert>}{busy && <LinearProgress />}
    <Accordion defaultExpanded={match.media.length === 0} disableGutters sx={{ mb: 1 }}><AccordionSummary>Assets & calibration · {match.media.length} media · {match.calibration ? `${match.calibration.field_length} × ${match.calibration.field_width} m` : 'No calibration'}</AccordionSummary><AccordionDetails>
      <Stack direction={{ xs: 'column', md: 'row' }} gap={1} flexWrap="wrap" mb={1}>
        <Select size="small" value={role} onChange={(e) => setRole(e.target.value)} aria-label="Imported media role"><MenuItem value="dual_lens">Dual-lens source</MenuItem><MenuItem value="tracking">Tracking video</MenuItem><MenuItem value="clip">Clip</MenuItem><MenuItem value="unknown">Other / image</MenuItem></Select>
        <Button component="label" disabled={busy}>Upload media<input hidden type="file" accept="image/*,video/*" onChange={(e) => { const file = e.target.files?.[0]; if (file) void run(async () => addMedia(await api.uploadMedia(matchId, file))); e.target.value = '' }} /></Button>
        <TextField size="small" label="Local media path" value={path} onChange={(e) => setPath(e.target.value)} /><Button disabled={busy || !path} onClick={() => void run(async () => { await addMedia(await api.importMediaPath(matchId, path)); setPath('') })}>Add path</Button>
        <FormControl size="small" sx={{ minWidth: 220, maxWidth: 420 }}><InputLabel>Calibration</InputLabel><Select label="Calibration" value={match.calibration?.path ?? ''} onChange={(e) => void run(async () => { await api.importCalibrationPath(matchId, e.target.value); await reload() })}><MenuItem value="">None</MenuItem>{activeCalibrations.map((item) => <MenuItem key={item.path} value={item.path}>{item.name}</MenuItem>)}</Select></FormControl>
        <Button component="label" disabled={busy}>Upload .veo<input hidden type="file" accept=".veo" onChange={(e) => { const file = e.target.files?.[0]; if (file) void run(async () => { await api.uploadCalibration(matchId, file); await reload() }); e.target.value = '' }} /></Button>
        <TextField size="small" label="Local .veo path" value={calibrationPath} onChange={(e) => setCalibrationPath(e.target.value)} /><Button disabled={busy || !calibrationPath} onClick={() => void run(async () => { await api.importCalibrationPath(matchId, calibrationPath); await reload() })}>Select .veo</Button>
      </Stack>
      {match.calibration && <Typography variant="caption">Calibration version: {match.calibration.version ?? 'not provided'}</Typography>}
      {match.media.map((item) => <Stack key={item.id} direction="row" alignItems="center" gap={1} sx={{ py: .5 }} flexWrap="wrap"><Typography variant="body2" sx={{ minWidth: 200 }}>{item.name}</Typography><Typography variant="caption" color="text.secondary">{item.width}×{item.height} · {item.fps?.toFixed(2) ?? '—'} fps · {item.duration_s?.toFixed(1) ?? '—'} s · {item.size_bytes ? `${(item.size_bytes / 1048576).toFixed(1)} MB` : 'size unknown'}</Typography>{isVideo(item) ? <Select size="small" value={item.role ?? 'unknown'} onChange={(e) => void run(async () => { await api.setRole(item.id, e.target.value); await reload() })}><MenuItem value="unknown">Choose role</MenuItem><MenuItem value="tracking">Tracking</MenuItem><MenuItem value="dual_lens">Dual lenses</MenuItem><MenuItem value="clip">Clip</MenuItem></Select> : <><Button size="small" onClick={() => { video.current?.pause(); setPlaying(false); setStillImage(item); setFrameRender(null) }}>View image</Button><Button disabled={busy || !context || item.width !== 2048 || item.height !== 2048} size="small" onClick={() => void run(() => renderImage(item))}>Render frame</Button></>}</Stack>)}
    </AccordionDetails></Accordion>
    <Stack direction="row" alignItems="center" flexWrap="wrap" gap={1} mb={1}>
      <ToggleButtonGroup size="small" exclusive value={mode} onChange={(_, value: ViewMode | null) => value && switchView(value)}><ToggleButton value="tracking">Tracking</ToggleButton><ToggleButton value="panorama">Panoramic</ToggleButton><ToggleButton value="dual_lens">Dual lenses</ToggleButton></ToggleButtonGroup>
      <Select size="small" value={source?.id ?? ''} displayEmpty onChange={(e) => { video.current?.pause(); setPlaying(false); setSourceId(e.target.value); setFrameRender(null); setStillImage(null); generation.current++ }}><MenuItem value="">No source assigned</MenuItem>{roleMedia.map((item) => <MenuItem key={item.id} value={item.id}>{item.name}</MenuItem>)}</Select>
      {mode === 'dual_lens' && <ToggleButtonGroup size="small" exclusive value={crop} onChange={(_, value: string | null) => value && setCrop(value)}><ToggleButton value="stacked">Stacked</ToggleButton><ToggleButton value="left">LEFT</ToggleButton><ToggleButton value="right">RIGHT</ToggleButton></ToggleButtonGroup>}
      {mode === 'panorama' && <Select size="small" value={width} onChange={(e) => { setWidth(Number(e.target.value)); setFrameRender(null); generation.current++ }}><MenuItem value={1920}>1920×1080</MenuItem><MenuItem value={3840}>3840×2160</MenuItem></Select>}
    </Stack>
    <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', lg: 'minmax(0, 2.15fr) minmax(300px, 1fr)' }, gap: 2 }}>
      <Box minWidth={0}>
        {stillImage ? <Box><img src={api.url(stillImage.url)} alt={stillImage.name} style={{ width: '100%', maxHeight: '65vh', objectFit: 'contain' }} /><Typography variant="caption">{stillImage.name} · Image preview. Use Render frame for a calibrated stacked frame.</Typography><Button onClick={() => setStillImage(null)}>Return to video</Button></Box> : frameRender ? <PanoramaViewer panorama={frameRender} points={displayPoints} onPixelClick={(x, y) => void measure(x, y)} /> : source && (mode !== 'panorama' || context) ? <MatchViewer key={source.id} media={source} mode={mode} crop={crop} context={context} width={width} initialTime={time} videoRef={video} onTime={setTime} onPlayback={setPlaying} points={displayPoints} onClick={(x, y) => void measure(x, y)} /> : <Alert severity="info">{mode === 'panorama' && !context ? 'Add a calibration to enable panoramic rendering.' : 'Assign a source in Assets & calibration to use this view.'}</Alert>}
        <Stack direction="row" alignItems="center" gap={1} mt={1} flexWrap="wrap">
          <Button disabled={playbackDisabled} onClick={() => { if (video.current?.paused) void video.current.play().catch((e: Error) => setError(e.message)); else video.current?.pause() }}>{playing ? 'Pause' : 'Play'}</Button>
          <Button disabled={playbackDisabled} onClick={() => { video.current?.pause(); seek(time - 1 / (source?.fps || 30)) }}>−1 frame</Button><Button disabled={playbackDisabled} onClick={() => { video.current?.pause(); seek(time + 1 / (source?.fps || 30)) }}>+1 frame</Button>
          <TextField label="Time (s)" size="small" type="number" value={Number(time.toFixed(3))} disabled={playbackDisabled} onChange={(e) => seek(Number(e.target.value))} sx={{ width: 120 }} /><Typography variant="caption">/ {duration.toFixed(1)} s</Typography>
          <Button disabled={busy || playbackDisabled} onClick={() => void run(async () => { video.current?.pause(); const frame = await api.extractFrame(source!.id, video.current?.currentTime ?? time); await reload(); if (context && source?.role === 'dual_lens') await renderImage(frame) })}>Extract frame / CPU render</Button>
          {frameRender && <Button onClick={() => { setFrameRender(null); generation.current++ }}>Return to video</Button>}
        </Stack><Slider aria-label="Match timeline" value={Math.min(time, duration)} max={duration || 1} step={source?.fps ? 1 / source.fps : .04} disabled={playbackDisabled} onChange={(_, value) => seek(value as number)} />
      </Box>
      <Box minWidth={0}><Typography variant="subtitle1">Pitch · X length / Z width</Typography>{context ? <PitchSvg length={context.field_length} width={context.field_width} points={points} /> : <Alert severity="info">Pitch dimensions become available after selecting a calibration.</Alert>}
        {mode === 'tracking' ? <Alert severity="info" sx={{ mt: 1 }}>{context?.tracking_geometry.reason ?? 'Metric clicks disabled: tracking camera pose and FOV are not available.'}</Alert> : <Stack spacing={1} mt={1}><FormControl size="small"><InputLabel>Click mode / landmark</InputLabel><Select label="Click mode / landmark" value={landmark} onChange={(e) => setLandmark(e.target.value)}><MenuItem value="">Free point</MenuItem>{context?.landmarks.map((item) => <MenuItem key={item.id} value={item.id}>{item.name} · {item.x.toFixed(2)}, {item.z.toFixed(2)} m</MenuItem>)}</Select></FormControl><Typography variant="caption">Validation measures calibration error; it does not correct the geometry.</Typography></Stack>}
        <Button onClick={() => { setPoints([]); generation.current++ }}>Clear points</Button>
      </Box>
    </Box><Divider sx={{ my: 1 }} /><ValidationTable points={points} />
  </Box>
}
