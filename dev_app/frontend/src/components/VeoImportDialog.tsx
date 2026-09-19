import { Alert, Button, Dialog, DialogActions, DialogContent, DialogTitle, LinearProgress, MenuItem, Select, Stack, Table, TableBody, TableCell, TableHead, TableRow, TextField, Typography } from '@mui/material'
import { useEffect, useState } from 'react'
import { api } from '../api/client'
import type { VeoInspection, VeoJob, VeoSession } from '../types'

export default function VeoImportDialog({ open, onClose, onComplete }: { open: boolean; onClose: () => void; onComplete: (id: string) => void }) {
  const [url, setUrl] = useState('')
  const [session, setSession] = useState<VeoSession | null>(null)
  const [inspection, setInspection] = useState<VeoInspection | null>(null)
  const [selection, setSelection] = useState<Record<string, string>>({})
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
    if (saved) void api.veoJob(saved).then(setJob).catch(() => window.sessionStorage.removeItem('veo-import-job'))
    return () => window.clearInterval(timer)
  }, [open])
  useEffect(() => {
    if (!active || !jobId) return
    const timer = window.setInterval(() => {
      void api.veoJob(jobId).then(setJob).catch((e: Error) => setError(e.message))
    }, 1000)
    return () => window.clearInterval(timer)
  }, [active, jobId])

  return <Dialog open={open} onClose={onClose} fullWidth maxWidth="lg"><DialogTitle>Import from Veo</DialogTitle><DialogContent><Stack spacing={2} mt={1}>
    {error && <Alert severity="error">{error}</Alert>}{session?.error && <Alert severity="warning">{session.error}</Alert>}
    <Typography variant="body2">Connect your dedicated Veo session, inspect an authorized match, then review the files before downloading.</Typography>
    <Stack direction="row" gap={1} alignItems="center" flexWrap="wrap"><Button disabled={busy || active} onClick={() => void run(async () => setSession(await api.connectVeo()))}>Connect Veo</Button><Button disabled={busy || active} onClick={() => void run(async () => setSession(await api.disconnectVeo()))}>Disconnect Veo</Button><Typography variant="caption">{session?.connect_browser_running ? 'Login browser open — close it when finished' : session?.profile_exists ? 'Local profile available · login validity checked during inspection' : 'No local session'}</Typography></Stack>
    <TextField label="Authorized Veo match URL" value={url} disabled={active || busy} onChange={(e) => setUrl(e.target.value)} />
    <Button variant="outlined" disabled={!url || busy || active || session?.connect_browser_running} onClick={() => void run(async () => { setInspection(await api.inspectVeo(url)); setSelection({}); setJob(null) })}>{busy ? 'Working…' : 'Inspect match'}</Button>
    {inspection && <>
      <Typography variant="subtitle1">{inspection.metadata.title || 'Match title not supplied'} · {inspection.candidates.length} resources</Typography>
      {Object.entries(inspection.metadata).filter(([key]) => key !== 'title').map(([key, value]) => <Typography key={key} variant="body2">{key}: {value}</Typography>)}
      {!inspection.candidates.length && <Alert severity="info">No assets discovered. Check access in Connect Veo. Some player modes only load their resources after activation; no private endpoints are guessed.</Alert>}
      <Typography variant="body2">Choose a role for each file to import. Unknown resources remain visible; manifests cannot be downloaded as complete videos.</Typography>
      <Table size="small"><TableHead><TableRow><TableCell>File / resource</TableCell><TableCell>Media metadata</TableCell><TableCell>Evidence</TableCell><TableCell>Import as</TableCell></TableRow></TableHead><TableBody>
        {inspection.candidates.map((candidate) => <TableRow key={candidate.identifier}><TableCell sx={{ maxWidth: 280, overflowWrap: 'anywhere' }}>{candidate.filename}<Typography variant="caption" display="block">{candidate.content_type ?? 'Content type unknown'}</Typography></TableCell><TableCell>{candidate.width && candidate.height ? `${candidate.width}×${candidate.height}` : 'Dimensions unknown'}<br />{candidate.fps?.toFixed(2) ?? '—'} fps · {candidate.duration_s?.toFixed(1) ?? '—'} s{candidate.field_length && <Typography variant="body2">Field {candidate.field_length}×{candidate.field_width} m · calibration {candidate.calibration_version ?? 'unknown version'}</Typography>}</TableCell><TableCell>{candidate.role} · {Math.round(candidate.confidence * 100)}%<Typography variant="caption" display="block">{candidate.reasons.join('; ')}</Typography></TableCell><TableCell><Select size="small" value={Object.keys(selection).find((role) => selection[role] === candidate.identifier) ?? ''} disabled={active || candidate.role === 'manifest'} displayEmpty onChange={(e) => { const next = Object.fromEntries(Object.entries(selection).filter(([, id]) => id !== candidate.identifier)); if (e.target.value) next[e.target.value] = candidate.identifier; setSelection(next) }}><MenuItem value="">Skip</MenuItem>{candidate.role === 'calibration' ? <MenuItem value="calibration">Calibration</MenuItem> : [<MenuItem key="panorama" value="panorama">Dual-lens source</MenuItem>, <MenuItem key="tracking" value="ball_tracking">Tracking video</MenuItem>]}</Select></TableCell></TableRow>)}
      </TableBody></Table>
    </>}
    {job && <><Typography>{job.status} · {job.current_asset ?? 'Import job'} · {(job.bytes_downloaded / 1048576).toFixed(1)} MB in current file</Typography><LinearProgress variant="determinate" value={(job.progress ?? 0) * 100} />{job.error && <Alert severity="error">{job.error}</Alert>}{active && <Button color="warning" onClick={() => void run(async () => setJob(await api.cancelVeoJob(job.identifier)))}>Cancel import</Button>}{job.match_id && !active && <Button onClick={() => onComplete(job.match_id!)}>Open {job.status === 'completed' ? 'match' : 'partially imported match'}</Button>}</>}
  </Stack></DialogContent><DialogActions><Button onClick={onClose}>Close</Button><Button variant="contained" disabled={!inspection || !Object.keys(selection).length || busy || active} onClick={() => void run(async () => { const next = await api.importVeo(inspection!.id, selection); setJob(next); window.sessionStorage.setItem('veo-import-job', next.identifier) })}>Import match</Button></DialogActions></Dialog>
}
