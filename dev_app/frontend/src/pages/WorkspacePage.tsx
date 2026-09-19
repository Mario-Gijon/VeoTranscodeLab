import ArrowBackIcon from '@mui/icons-material/ArrowBack'
import ChevronLeftIcon from '@mui/icons-material/ChevronLeft'
import ChevronRightIcon from '@mui/icons-material/ChevronRight'
import ClearIcon from '@mui/icons-material/Clear'
import PlayArrowIcon from '@mui/icons-material/PlayArrow'
import RefreshIcon from '@mui/icons-material/Refresh'
import UploadFileIcon from '@mui/icons-material/UploadFile'
import { Alert, Box, Button, Card, CardContent, Chip, Divider, FormControl, InputLabel, MenuItem, Paper, Select, Stack, TextField, ToggleButton, ToggleButtonGroup, Typography } from '@mui/material'
import { ChangeEvent, useEffect, useMemo, useRef, useState } from 'react'
import { api } from '../api/client'
import PanoramaViewer from '../components/PanoramaViewer'
import PitchSvg from '../components/PitchSvg'
import ValidationTable from '../components/ValidationTable'
import type { Calibration, GroundPoint, MatchWorkspace, MediaItem, RenderedPanorama } from '../types'

interface Props { matchId: string; calibrations: Calibration[]; onBack: () => void }
const formatTime = (seconds: number) => new Date(seconds * 1000).toISOString().slice(11, 19)
const isRenderable = (media: MediaItem | undefined) => media?.kind === 'image' || media?.kind === 'frame'

export default function WorkspacePage({ matchId, calibrations, onBack }: Props) {
  const [workspace, setWorkspace] = useState<MatchWorkspace | null>(null)
  const [selectedMediaId, setSelectedMediaId] = useState('')
  const [panorama, setPanorama] = useState<RenderedPanorama | null>(null)
  const [points, setPoints] = useState<GroundPoint[]>([])
  const [clickMode, setClickMode] = useState<'free' | 'landmark'>('free')
  const [landmarkId, setLandmarkId] = useState('')
  const [resolution, setResolution] = useState('1920x1080')
  const [videoTime, setVideoTime] = useState(0)
  const [pathValue, setPathValue] = useState('')
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  const video = useRef<HTMLVideoElement>(null)
  const selectedMedia = workspace?.media.find((item) => item.id === selectedMediaId)
  const naturalMediaUrl = selectedMedia ? api.url(selectedMedia.url) : ''
  const resolutionParts = useMemo(() => resolution.split('x').map(Number), [resolution])

  const reload = async () => { try { const result = await api.match(matchId); setWorkspace(result); if (!selectedMediaId && result.media[0]) setSelectedMediaId(result.media[0].id) } catch (reason) { setError(reason instanceof Error ? reason.message : 'Could not load workspace.') } }
  // Reload when navigating to another local workspace; reload intentionally owns current UI state.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  useEffect(() => { void reload() }, [matchId])
  const renderCurrent = async (mediaId = selectedMediaId) => {
    if (!workspace || !mediaId) return
    setBusy(true); setError('')
    try { const result = await api.render(workspace.id, mediaId, resolutionParts[0], resolutionParts[1]); setPanorama(result); setPoints([]); setLandmarkId(result.landmarks[0]?.id ?? '') }
    catch (reason) { setError(reason instanceof Error ? reason.message : 'Could not render panorama.') }
    finally { setBusy(false) }
  }
  const selectMedia = (media: MediaItem) => { setSelectedMediaId(media.id); setPanorama(null); setPoints([]); if (isRenderable(media)) setTimeout(() => void renderCurrent(media.id), 0) }
  const uploadMedia = async (event: ChangeEvent<HTMLInputElement>) => { const file = event.target.files?.[0]; if (!file || !workspace) return; setBusy(true); try { const item = await api.uploadMedia(workspace.id, file); await reload(); selectMedia(item) } catch (reason) { setError(reason instanceof Error ? reason.message : 'Could not import media.') } finally { setBusy(false); event.target.value = '' } }
  const importPath = async () => { if (!workspace || !pathValue.trim()) return; setBusy(true); try { const item = await api.importMediaPath(workspace.id, pathValue.trim()); setPathValue(''); await reload(); selectMedia(item) } catch (reason) { setError(reason instanceof Error ? reason.message : 'Could not import media path.') } finally { setBusy(false) } }
  const uploadCalibration = async (event: ChangeEvent<HTMLInputElement>) => { const file = event.target.files?.[0]; if (!file || !workspace) return; setBusy(true); try { await api.uploadCalibration(workspace.id, file); await reload() } catch (reason) { setError(reason instanceof Error ? reason.message : 'Could not upload calibration.') } finally { setBusy(false); event.target.value = '' } }
  const changeCalibration = async (path: string) => { if (!workspace) return; setBusy(true); try { await api.importCalibrationPath(workspace.id, path); await reload(); setPanorama(null); setPoints([]) } catch (reason) { setError(reason instanceof Error ? reason.message : 'Could not select calibration.') } finally { setBusy(false) } }
  const extractCurrent = async () => { if (!selectedMedia) return; setBusy(true); try { const frame = await api.extractFrame(selectedMedia.id, videoTime); await reload(); selectMedia(frame) } catch (reason) { setError(reason instanceof Error ? reason.message : 'Could not extract frame.') } finally { setBusy(false) } }
  const moveFrame = (direction: number) => { if (!video.current || !selectedMedia?.fps) return; video.current.currentTime = Math.max(0, Math.min(selectedMedia.duration_s ?? Infinity, video.current.currentTime + direction / selectedMedia.fps)); setVideoTime(video.current.currentTime) }
  const clickPixel = async (x: number, y: number) => { if (!panorama) return; try { const point = await api.groundPoint(panorama.id, x, y, clickMode === 'landmark' ? landmarkId : null); setPoints((items) => [...items, { ...point, point_index: items.length + 1 }]) } catch (reason) { setError(reason instanceof Error ? reason.message : 'Could not map pixel to ground.') } }

  if (!workspace) return <Box sx={{ p: 3 }}><Typography>Loading workspace…</Typography></Box>
  return <Box sx={{ maxWidth: 1800, mx: 'auto', px: { xs: 1.5, md: 2.5 }, py: 2 }}>
    <Stack direction="row" alignItems="center" gap={1} mb={2}><Button startIcon={<ArrowBackIcon />} onClick={onBack}>Library</Button><Divider orientation="vertical" flexItem /><Box><Typography variant="h5">{workspace.name}</Typography><Typography variant="body2" color="text.secondary">{workspace.description || 'Panorama-to-pitch validation workspace'}</Typography></Box></Stack>
    {error && <Alert severity="error" sx={{ mb: 2 }} onClose={() => setError('')}>{error}</Alert>}
    <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', xl: '280px minmax(0, 1fr)' }, gap: 2 }}>
      <Stack spacing={2}>
        <Card><CardContent><Typography variant="subtitle1" fontWeight={700}>Source & calibration</Typography><Stack spacing={1.25} mt={1.5}><FormControl fullWidth size="small"><InputLabel id="workspace-calibration">Calibration</InputLabel><Select labelId="workspace-calibration" label="Calibration" value={workspace.calibration?.path ?? ''} onChange={(event) => void changeCalibration(event.target.value)}><MenuItem value=""><em>Choose calibration</em></MenuItem>{calibrations.map((item) => <MenuItem key={item.id} value={item.path}>{item.name}</MenuItem>)}</Select></FormControl>{workspace.calibration && <Chip size="small" color="secondary" label={`Field ${workspace.calibration.field_length} × ${workspace.calibration.field_width} m`} />}
          <Button component="label" size="small" startIcon={<UploadFileIcon />}>Upload .veo<input hidden type="file" accept=".veo,application/json" onChange={uploadCalibration} /></Button>
          <Divider /><Button component="label" variant="contained" size="small" startIcon={<UploadFileIcon />}>Import media<input hidden type="file" accept="image/*,video/*" onChange={uploadMedia} /></Button><Stack direction="row" gap={.5}><TextField size="small" placeholder="Existing local path" value={pathValue} onChange={(event) => setPathValue(event.target.value)} /><Button size="small" onClick={() => void importPath()}>Add</Button></Stack><Typography variant="caption" color="text.secondary">Path imports reference an existing local file; uploads are copied to the runtime workspace.</Typography></Stack></CardContent></Card>
        <Card><CardContent><Typography variant="subtitle1" fontWeight={700}>Workspace media</Typography><Stack spacing={.75} mt={1.25}>{workspace.media.length === 0 ? <Typography variant="body2" color="text.secondary">Add a 2048×2048 stacked frame, video, or clip.</Typography> : workspace.media.map((item) => <Paper key={item.id} variant="outlined" onClick={() => selectMedia(item)} sx={{ cursor: 'pointer', p: 1, borderColor: selectedMediaId === item.id ? 'primary.main' : 'divider', bgcolor: selectedMediaId === item.id ? 'primary.dark' : 'transparent' }}><Typography variant="body2" noWrap>{item.name}</Typography><Typography variant="caption" color="text.secondary">{item.kind} · {item.width}×{item.height}{item.fps ? ` · ${item.fps.toFixed(2)} fps` : ''}</Typography></Paper>)}</Stack></CardContent></Card>
      </Stack>
      <Stack spacing={2} minWidth={0}>
        {selectedMedia?.kind === 'video' || selectedMedia?.kind === 'clip' ? <Card><CardContent><Stack direction={{ xs: 'column', md: 'row' }} gap={2} alignItems="center"><video ref={video} controls src={naturalMediaUrl} style={{ maxWidth: 560, width: '100%', borderRadius: 6 }} onTimeUpdate={(event) => setVideoTime(event.currentTarget.currentTime)} /><Stack spacing={1} minWidth={190}><Typography variant="subtitle2">Frame selection</Typography><Typography variant="body2">{formatTime(videoTime)} / {selectedMedia.duration_s ? formatTime(selectedMedia.duration_s) : '—'}</Typography><Stack direction="row"><Button aria-label="Previous frame" onClick={() => moveFrame(-1)}><ChevronLeftIcon /></Button><Button aria-label="Next frame" onClick={() => moveFrame(1)}><ChevronRightIcon /></Button></Stack><Button variant="contained" startIcon={<PlayArrowIcon />} disabled={busy} onClick={() => void extractCurrent()}>Use current frame</Button><Typography variant="caption" color="text.secondary">Extracted frames become reusable workspace media.</Typography></Stack></Stack></CardContent></Card> : null}
        {selectedMedia && isRenderable(selectedMedia) && <Card><CardContent><Stack direction={{ xs: 'column', sm: 'row' }} justifyContent="space-between" gap={1} alignItems={{ sm: 'center' }}><Box><Typography variant="subtitle1" fontWeight={700}>Panorama renderer</Typography><Typography variant="caption" color="text.secondary">Uses the validated Python renderer and selected .veo metadata.</Typography></Box><Stack direction="row" gap={1}><Select size="small" value={resolution} onChange={(event) => setResolution(event.target.value)}><MenuItem value="1920x1080">1920 × 1080</MenuItem><MenuItem value="3840x2160">3840 × 2160</MenuItem></Select><Button variant="outlined" startIcon={<RefreshIcon />} disabled={busy} onClick={() => void renderCurrent()}>Render</Button></Stack></Stack></CardContent></Card>}
        {panorama ? <>
          <Stack direction={{ xs: 'column', sm: 'row' }} gap={1} alignItems={{ sm: 'center' }}><ToggleButtonGroup size="small" exclusive value={clickMode} onChange={(_, value: 'free' | 'landmark' | null) => value && setClickMode(value)}><ToggleButton value="free">Free point</ToggleButton><ToggleButton value="landmark">Selected landmark</ToggleButton></ToggleButtonGroup>{clickMode === 'landmark' && <FormControl size="small" sx={{ minWidth: 250 }}><InputLabel id="landmark">Landmark</InputLabel><Select labelId="landmark" label="Landmark" value={landmarkId} onChange={(event) => setLandmarkId(event.target.value)}>{panorama.landmarks.map((item) => <MenuItem key={item.id} value={item.id}>{item.name} · X {item.x.toFixed(2)}, Z {item.z.toFixed(2)}</MenuItem>)}</Select></FormControl>}<Button color="inherit" size="small" startIcon={<ClearIcon />} onClick={() => setPoints([])}>Clear points</Button></Stack>
          <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', lg: 'minmax(0, 2.05fr) minmax(300px, 1fr)' }, gap: 2, alignItems: 'start' }}><Card><CardContent><Typography variant="subtitle1" fontWeight={700} mb={1}>Panorama</Typography><PanoramaViewer panorama={panorama} points={points} onPixelClick={(x, y) => void clickPixel(x, y)} /></CardContent></Card><Card><CardContent><Typography variant="subtitle1" fontWeight={700} mb={1}>Pitch reference</Typography><PitchSvg length={panorama.field_length} width={panorama.field_width} points={points} /></CardContent></Card></Box>
          <ValidationTable points={points} />
        </> : <Card><CardContent><Typography variant="h6">Choose source media</Typography><Typography color="text.secondary">An image/frame is rendered immediately when selected. Video media can first be positioned and extracted as a frame.</Typography></CardContent></Card>}
      </Stack>
    </Box>
  </Box>
}
