import { Accordion, AccordionDetails, AccordionSummary, Alert, Box, Button, Dialog, DialogActions, DialogContent, DialogTitle, LinearProgress, Stack, TextField, Typography } from '@mui/material'
import { useEffect, useState } from 'react'
import { api } from '../api/client'
import type { MatchWorkspace, VeoInspection, VeoJob, VeoSession } from '../types'

const roles = [['ball_tracking', 'Tracking'], ['panorama', 'Dual lens'], ['calibration', 'Calibration']] as const
const bytes = (value: number) => value >= 1e9 ? (value / 1e9).toFixed(2) + ' GB' : (value / 1e6).toFixed(1) + ' MB'
const statusLabel: Record<string, string> = { waiting: 'Waiting', downloading: 'Downloading', complete: 'Complete', failed: 'Failed', cancelled: 'Cancelled' }

export default function VeoImportDialog({ open, retryMatch, onClose, onComplete }: { open: boolean; retryMatch?: MatchWorkspace | null; onClose: () => void; onComplete: (id: string) => void }) {
  const [url, setUrl] = useState(retryMatch?.metadata?.veo_url || '')
  const [session, setSession] = useState<VeoSession | null>(null)
  const [inspection, setInspection] = useState<VeoInspection | null>(null)
  const [job, setJob] = useState<VeoJob | null>(null)
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  const active = !!job && ['queued', 'downloading'].includes(job.status)
  const jobId = job?.identifier
  const run = async (action: () => Promise<unknown>) => {
    setBusy(true); setError('')
    try { await action() } catch (reason) { setError(reason instanceof Error ? reason.message : 'Veo operation failed') }
    finally { setBusy(false) }
  }
  useEffect(() => {
    if (!open) return
    const refresh = () => { void api.veoSession().then(setSession).catch((e: Error) => setError(e.message)) }
    refresh()
    const timer = window.setInterval(refresh, 2500)
    const saved = window.sessionStorage.getItem('veo-import-job')
    if (retryMatch) void api.matchJob(retryMatch.id).then((value) => setJob(value?.status === 'completed' ? null : value)).catch((e: Error) => setError(e.message))
    else if (saved) void api.veoJob(saved).then((value) => { if (['queued', 'downloading'].includes(value.status)) setJob(value) }).catch(() => window.sessionStorage.removeItem('veo-import-job'))
    return () => window.clearInterval(timer)
  }, [open, retryMatch])
  useEffect(() => {
    if (!active || !jobId) return
    const timer = window.setInterval(() => { void api.veoJob(jobId).then(setJob).catch((e: Error) => setError(e.message)) }, 700)
    return () => window.clearInterval(timer)
  }, [active, jobId])
  const ready = !!inspection?.ready && !!inspection.resolved_assets && !active && !busy && job?.status !== 'completed'
  return <Dialog open={open} onClose={busy ? undefined : onClose} fullWidth maxWidth="sm"><DialogTitle>{active ? 'Importing match' : retryMatch ? 'Resume import · ' + retryMatch.name : 'Import from Veo'}</DialogTitle><DialogContent><Stack gap={2} mt={1}>
    {error && <Alert severity="error">{error}</Alert>}{session?.error && <Alert severity="warning">{session.error}</Alert>}
    {!active && <>
      <TextField label="Veo URL" value={url} disabled={busy || !!retryMatch} onChange={(e) => { setUrl(e.target.value); setInspection(null); setJob(null) }} placeholder="https://app.veo.co/matches/…" />
      <Stack direction="row" alignItems="center" justifyContent="space-between" gap={1}><Typography variant="body2" color="text.secondary">Session: {session?.connect_browser_running ? 'Sign in, then close the browser' : session?.profile_exists ? 'Saved · checked on inspection' : 'Not connected'}</Typography><Button disabled={busy} onClick={() => void run(async () => setSession(await api.connectVeo()))}>{session?.profile_exists ? 'Reconnect' : 'Connect Veo'}</Button></Stack>
      <Button variant="outlined" disabled={!url.trim() || busy || session?.connect_browser_running} onClick={() => void run(async () => setInspection(await api.inspectVeo(url)))}>{busy ? 'Inspecting match…' : inspection ? 'Retry inspection' : 'Inspect'}</Button>
      {busy && <LinearProgress />}
    </>}
    {inspection && !active && <>
      {inspection.metadata.title && <Typography variant="subtitle2">{inspection.metadata.title}</Typography>}
      {roles.map(([role, label]) => { const candidate = inspection.candidates.find((item) => item.identifier === inspection.resolved_assets[role]); return <Stack direction="row" key={role} gap={2} alignItems="center"><Typography variant="body2" sx={{ width: 90 }}>{label}</Typography><Typography variant="body2" noWrap color={candidate ? 'primary.main' : 'text.secondary'}>{candidate ? '✓ ' + candidate.filename : 'Missing'}</Typography></Stack> })}
      {!inspection.ready && <Alert severity="info">All three files are needed for a Veo import. Check access to this match, then retry inspection.</Alert>}
    </>}
    {job && <>
      {roles.map(([role, label]) => { const item = job.assets?.[role]; const ratio = item?.bytes_total ? Math.min(100, item.bytes_downloaded / item.bytes_total * 100) : null; return <Box key={role}><Stack direction="row" justifyContent="space-between" mb={.5}><Typography variant="subtitle2">{label}</Typography><Typography variant="caption" color="text.secondary">{statusLabel[item?.status || 'waiting'] || 'Waiting'}{ratio !== null ? ' · ' + Math.round(ratio) + '%' : ''}</Typography></Stack><LinearProgress variant={item?.status === 'downloading' && ratio === null ? 'indeterminate' : 'determinate'} value={ratio ?? 0} sx={{ height: 5, borderRadius: 1 }} />{!!item?.bytes_downloaded && <Typography variant="caption" color="text.secondary">{bytes(item.bytes_downloaded)}{item.bytes_total ? ' / ' + bytes(item.bytes_total) : ' transferred'}</Typography>}</Box> })}
      <Box><Stack direction="row" justifyContent="space-between"><Typography variant="body2">Overall</Typography><Typography variant="caption">{job.progress != null ? Math.round(job.progress * 100) + '%' : (job.completed_assets || 0) + ' / 3 files complete'}</Typography></Stack><LinearProgress variant={active && job.progress == null ? 'indeterminate' : 'determinate'} value={(job.progress ?? 0) * 100} /></Box>
      {job.error && <Alert severity="error">{job.error}</Alert>}{job.status === 'cancelled' && <Alert severity="info">Import cancelled. Completed files are saved; you can retry the remaining files.</Alert>}{job.status === 'completed' && <Alert severity="success">Match saved. All three files are ready.</Alert>}
      {active ? <Button onClick={() => void run(async () => setJob(await api.cancelVeoJob(job.identifier)))}>Cancel import</Button> : job.match_id && <Button variant={job.status === 'completed' ? 'contained' : 'outlined'} onClick={() => onComplete(job.match_id!)}>Open {job.status === 'completed' ? 'match' : 'partial match'}</Button>}
    </>}
    <Accordion disableGutters elevation={0}><AccordionSummary><Typography variant="caption" color="text.secondary">Developer diagnostics</Typography></AccordionSummary><AccordionDetails><Button size="small" disabled={busy || active} onClick={() => void run(async () => setSession(await api.disconnectVeo()))}>Disconnect Veo</Button>{inspection && <Box component="pre" sx={{ fontSize: 11, maxHeight: 260, overflow: 'auto', whiteSpace: 'pre-wrap', overflowWrap: 'anywhere' }}>{JSON.stringify(inspection, null, 2)}</Box>}</AccordionDetails></Accordion>
  </Stack></DialogContent><DialogActions><Button disabled={busy} onClick={onClose}>{active ? 'Continue in background' : 'Close'}</Button>{job?.status !== 'completed' && <Button variant="contained" disabled={!ready} onClick={() => void run(async () => { const next = await api.importVeo(inspection!.id, inspection!.resolved_assets, job?.match_id || retryMatch?.id); setJob(next); window.sessionStorage.setItem('veo-import-job', next.identifier) })}>{job || retryMatch ? 'Retry remaining files' : 'Import match'}</Button>}</DialogActions></Dialog>
}
