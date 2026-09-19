import AddIcon from '@mui/icons-material/Add'
import OpenInNewIcon from '@mui/icons-material/OpenInNew'
import { Alert, Box, Button, Card, CardActions, CardContent, Dialog, DialogActions, DialogContent, DialogTitle, FormControl, InputLabel, MenuItem, Select, Stack, TextField, Typography } from '@mui/material'
import { useState } from 'react'
import { api } from '../api/client'
import type { Calibration, MatchWorkspace } from '../types'

interface Props { matches: MatchWorkspace[]; calibrations: Calibration[]; onCreated: (match: MatchWorkspace) => void; onOpen: (id: string) => void }

export default function LibraryPage({ matches, calibrations, onCreated, onOpen }: Props) {
  const [open, setOpen] = useState(false)
  const [name, setName] = useState('')
  const [description, setDescription] = useState('')
  const [calibrationId, setCalibrationId] = useState('')
  const [error, setError] = useState('')
  const [saving, setSaving] = useState(false)
  const submit = async () => {
    setSaving(true); setError('')
    try { const match = await api.createMatch(name, description, calibrationId || null); onCreated(match); setOpen(false); setName(''); setDescription(''); setCalibrationId('') }
    catch (reason) { setError(reason instanceof Error ? reason.message : 'Could not create workspace.') }
    finally { setSaving(false) }
  }
  return <Box sx={{ maxWidth: 1300, mx: 'auto', px: { xs: 2, md: 3 }, py: 3 }}>
    <Stack direction={{ xs: 'column', sm: 'row' }} alignItems={{ sm: 'center' }} justifyContent="space-between" gap={2} mb={3}>
      <Box><Typography variant="h4" component="h1">Match library</Typography><Typography color="text.secondary">Local workspaces keep their calibration and reusable source media together.</Typography></Box>
      <Button variant="contained" startIcon={<AddIcon />} onClick={() => setOpen(true)}>Create workspace</Button>
    </Stack>
    {matches.length === 0 ? <Card><CardContent><Typography variant="h6">No workspaces yet</Typography><Typography color="text.secondary">Create a match workspace, choose a .veo calibration, and add a frame or transcode.</Typography></CardContent></Card> :
      <Box sx={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(290px, 1fr))', gap: 2 }}>{matches.map((match) => <Card key={match.id}><CardContent><Typography variant="h6">{match.name}</Typography><Typography variant="body2" color="text.secondary" minHeight={42}>{match.description || 'No description'}</Typography><Stack mt={2} spacing={.5}><Typography variant="caption">Calibration: {match.calibration?.name ?? 'Not selected'}</Typography><Typography variant="caption">Field: {match.calibration ? `${match.calibration.field_length} × ${match.calibration.field_width} m` : '—'}</Typography><Typography variant="caption">Media: {match.media_count ?? match.media.length}</Typography></Stack></CardContent><CardActions><Button endIcon={<OpenInNewIcon />} onClick={() => onOpen(match.id)}>Open workspace</Button></CardActions></Card>)}</Box>}
    <Dialog open={open} onClose={() => setOpen(false)} fullWidth maxWidth="sm"><DialogTitle>Create match workspace</DialogTitle><DialogContent><Stack spacing={2} mt={1}>{error && <Alert severity="error">{error}</Alert>}<TextField autoFocus required label="Match name" value={name} onChange={(event) => setName(event.target.value)} /><TextField label="Description" multiline minRows={2} value={description} onChange={(event) => setDescription(event.target.value)} /><FormControl fullWidth><InputLabel id="library-calibration">Calibration</InputLabel><Select labelId="library-calibration" label="Calibration" value={calibrationId} onChange={(event) => setCalibrationId(event.target.value)}><MenuItem value="">Add later</MenuItem>{calibrations.map((calibration) => <MenuItem key={calibration.id} value={calibration.id}>{calibration.name} · {calibration.field_length} × {calibration.field_width} m</MenuItem>)}</Select></FormControl></Stack></DialogContent><DialogActions><Button onClick={() => setOpen(false)}>Cancel</Button><Button variant="contained" disabled={!name.trim() || saving} onClick={submit}>Create</Button></DialogActions></Dialog>
  </Box>
}
