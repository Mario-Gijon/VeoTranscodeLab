import { Accordion, AccordionDetails, AccordionSummary, Alert, Box, Button, Dialog, DialogActions, DialogContent, DialogTitle, LinearProgress, Stack, TextField, Typography } from '@mui/material'
import { useEffect, useState } from 'react'
import { api } from '../api/client'
import type { MatchWorkspace, VeoInspection, VeoJob } from '../types'

const roles = [['ball_tracking', 'Seguimiento de balón'], ['panorama', 'Doble lente'], ['calibration', 'Calibración'], ['camera_directions', 'Direcciones de cámara']] as const
const bytes = (value: number) => value >= 1e9 ? (value / 1e9).toFixed(2) + ' GB' : (value / 1e6).toFixed(1) + ' MB'
const statusLabel: Record<string, string> = { waiting: 'En espera', downloading: 'Descargando', complete: 'Completado', failed: 'Fallido', cancelled: 'Cancelado' }

export default function VeoImportDialog({ open, retryMatch, onClose, onComplete }: { open: boolean; retryMatch?: MatchWorkspace | null; onClose: () => void; onComplete: (id: string) => void }) {
  const [url, setUrl] = useState(retryMatch?.metadata?.veo_url || '')
  const [inspection, setInspection] = useState<VeoInspection | null>(null)
  const [job, setJob] = useState<VeoJob | null>(null)
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  const active = !!job && ['queued', 'downloading'].includes(job.status)
  const jobId = job?.identifier
  const run = async (action: () => Promise<unknown>) => {
    setBusy(true); setError('')
    try { await action() } catch (reason) { setError(reason instanceof Error ? reason.message : 'La operación de Veo ha fallado') }
    finally { setBusy(false) }
  }
  useEffect(() => {
    if (!open) return
    const saved = window.sessionStorage.getItem('veo-import-job')
    if (retryMatch) void api.matchJob(retryMatch.id).then((value) => setJob(value?.status === 'completed' ? null : value)).catch((e: Error) => setError(e.message))
    else if (saved) void api.veoJob(saved).then((value) => { if (['queued', 'downloading'].includes(value.status)) setJob(value) }).catch(() => window.sessionStorage.removeItem('veo-import-job'))
  }, [open, retryMatch])
  useEffect(() => {
    if (!active || !jobId) return
    const timer = window.setInterval(() => { void api.veoJob(jobId).then(setJob).catch((e: Error) => setError(e.message)) }, 700)
    return () => window.clearInterval(timer)
  }, [active, jobId])
  const ready = !!inspection?.ready && !!inspection.resolved_assets && !active && !busy && job?.status !== 'completed'
  const assetCount = Object.keys(job?.assets || {}).length || 3
  return <Dialog open={open} onClose={busy ? undefined : onClose} fullWidth maxWidth="sm"><DialogTitle>{active ? 'Importando partido' : retryMatch ? 'Reanudar importación · ' + retryMatch.name : 'Importar desde Veo'}</DialogTitle><DialogContent><Stack gap={2} mt={1}>
    {error && <Alert severity="error">{error}</Alert>}
    {!active && <>
      <TextField label="URL de Veo" value={url} disabled={busy || !!retryMatch} onChange={(e) => { setUrl(e.target.value); setInspection(null); setJob(null) }} placeholder="https://app.veo.co/matches/…" />
      <Button variant="outlined" disabled={!url.trim() || busy} onClick={() => void run(async () => setInspection(await api.inspectVeo(url)))}>{busy ? 'Inspeccionando partido…' : inspection ? 'Reintentar inspección' : 'Inspeccionar'}</Button>
      {busy && <LinearProgress />}
    </>}
    {inspection && !active && <>
      {inspection.metadata.title && <Typography variant="subtitle2">{inspection.metadata.title}</Typography>}
      {roles.map(([role, label]) => { const candidate = inspection.candidates.find((item) => item.identifier === inspection.resolved_assets[role]); return <Stack direction="row" key={role} gap={2} alignItems="center"><Typography variant="body2" sx={{ width: 130 }}>{label}</Typography><Typography variant="body2" noWrap color={candidate ? 'primary.main' : 'text.secondary'}>{candidate ? '✓ ' + candidate.filename : 'Falta'}</Typography></Stack> })}
      {!inspection.ready && <Alert severity="info">{inspection.access_error || 'Veo no mostró los tres archivos necesarios después de activar Interactivo. Reintenta la inspección.'}</Alert>}
    </>}
    {job && <>
      {roles.map(([role, label]) => { const item = job.assets?.[role]; const ratio = item?.bytes_total ? Math.min(100, item.bytes_downloaded / item.bytes_total * 100) : null; return <Box key={role}><Stack direction="row" justifyContent="space-between" mb={.5}><Typography variant="subtitle2">{label}</Typography><Typography variant="caption" color="text.secondary">{statusLabel[item?.status || 'waiting'] || 'En espera'}{ratio !== null ? ' · ' + Math.round(ratio) + '%' : ''}</Typography></Stack><LinearProgress variant={item?.status === 'downloading' && ratio === null ? 'indeterminate' : 'determinate'} value={ratio ?? 0} sx={{ height: 5, borderRadius: 1 }} />{!!item?.bytes_downloaded && <Typography variant="caption" color="text.secondary">{bytes(item.bytes_downloaded)}{item.bytes_total ? ' / ' + bytes(item.bytes_total) : ' transferidos'}</Typography>}</Box> })}
      <Box><Stack direction="row" justifyContent="space-between"><Typography variant="body2">Total</Typography><Typography variant="caption">{job.progress != null ? Math.round(job.progress * 100) + '%' : (job.completed_assets || 0) + ' / ' + assetCount + ' archivos completados'}</Typography></Stack><LinearProgress variant={active && job.progress == null ? 'indeterminate' : 'determinate'} value={(job.progress ?? 0) * 100} /></Box>
      {job.error && <Alert severity="error">{job.error}</Alert>}{job.status === 'cancelled' && <Alert severity="info">Importación cancelada. Los archivos completados se han guardado; puedes reintentar los restantes.</Alert>}{job.status === 'completed' && <Alert severity="success">Importación completada. Los archivos seleccionados están listos.</Alert>}
      {active ? <Button onClick={() => void run(async () => setJob(await api.cancelVeoJob(job.identifier)))}>Cancelar importación</Button> : job.match_id && <Button variant={job.status === 'completed' ? 'contained' : 'outlined'} onClick={() => onComplete(job.match_id!)}>Abrir {job.status === 'completed' ? 'partido' : 'partido parcial'}</Button>}
    </>}
    <Accordion disableGutters elevation={0}><AccordionSummary><Typography variant="caption" color="text.secondary">Diagnóstico técnico</Typography></AccordionSummary><AccordionDetails>{inspection && <Box component="pre" sx={{ fontSize: 11, maxHeight: 260, overflow: 'auto', whiteSpace: 'pre-wrap', overflowWrap: 'anywhere' }}>{JSON.stringify(inspection, null, 2)}</Box>}</AccordionDetails></Accordion>
  </Stack></DialogContent><DialogActions><Button disabled={busy} onClick={onClose}>{active ? 'Continuar en segundo plano' : 'Cerrar'}</Button>{job?.status !== 'completed' && <Button variant="contained" disabled={!ready} onClick={() => void run(async () => { const next = await api.importVeo(inspection!.id, inspection!.resolved_assets, job?.match_id || retryMatch?.id); setJob(next); window.sessionStorage.setItem('veo-import-job', next.identifier) })}>{job || retryMatch ? 'Reintentar archivos restantes' : 'Importar partido'}</Button>}</DialogActions></Dialog>
}
