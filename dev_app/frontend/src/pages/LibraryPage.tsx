import AddIcon from '@mui/icons-material/Add'
import MoreHorizIcon from '@mui/icons-material/MoreHoriz'
import SportsSoccerIcon from '@mui/icons-material/SportsSoccer'
import { Alert, Box, Button, Chip, Dialog, DialogActions, DialogContent, DialogTitle, IconButton, Menu, MenuItem, Paper, Stack, TextField, Typography } from '@mui/material'
import { useState } from 'react'
import { api } from '../api/client'
import ManageFilesDialog from '../components/ManageFilesDialog'
import VeoImportDialog from '../components/VeoImportDialog'
import type { Calibration, MatchWorkspace } from '../types'

interface Props { matches: MatchWorkspace[]; calibrations: Calibration[]; onRefresh: () => Promise<void>; onOpen: (id: string) => void }

export default function LibraryPage({ matches, calibrations, onRefresh, onOpen }: Props) {
  const [dialog, setDialog] = useState<'new' | 'rename' | 'delete' | null>(null)
  const [veoOpen, setVeoOpen] = useState(false)
  const [retryMatch, setRetryMatch] = useState<MatchWorkspace | null>(null)
  const [selected, setSelected] = useState<MatchWorkspace | null>(null)
  const [files, setFiles] = useState<MatchWorkspace | null>(null)
  const [anchor, setAnchor] = useState<HTMLElement | null>(null)
  const [name, setName] = useState('')
  const [error, setError] = useState('')
  const [saving, setSaving] = useState(false)
  const submit = async () => {
    setSaving(true); setError('')
    try {
      if (dialog === 'delete' && selected) await api.deleteMatch(selected.id)
      else if (dialog === 'rename' && selected) await api.renameMatch(selected.id, name)
      else setFiles(await api.createMatch(name, '', null))
      await onRefresh(); setDialog(null)
    } catch (reason) { setError(reason instanceof Error ? reason.message : 'Could not update match.') }
    finally { setSaving(false) }
  }
  return <Box sx={{ maxWidth: 1440, mx: 'auto', px: { xs: 2, md: 4 }, py: 3 }}>
    <Stack direction={{ xs: 'column', sm: 'row' }} justifyContent="space-between" gap={2} mb={3}>
      <Box><Typography variant="h5" component="h1">Matches</Typography><Typography variant="body2" color="text.secondary">Your footage, pitch and analysis in one place.</Typography></Box>
      <Stack direction="row" gap={1} alignItems="center"><Button variant="outlined" onClick={() => { setRetryMatch(null); setVeoOpen(true) }}>Import from Veo</Button><Button variant="contained" startIcon={<AddIcon />} onClick={() => { setName(''); setError(''); setDialog('new') }}>New Match</Button></Stack>
    </Stack>
    {matches.length === 0 ? <Paper variant="outlined" sx={{ px: 3, py: 6, textAlign: 'center' }}><SportsSoccerIcon sx={{ fontSize: 36, color: 'primary.main', mb: 1 }} /><Typography variant="h6">Start with your first match</Typography><Typography color="text.secondary" variant="body2">Import from Veo or create a match with local files. You can add more files later.</Typography></Paper> :
      <Box sx={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(300px, 1fr))', gap: 1.5 }}>{matches.map((match) => <Paper variant="outlined" key={match.id} sx={{ p: 2 }}>
        <Stack direction="row" alignItems="center" mb={2}><Typography variant="subtitle1" fontWeight={650} noWrap flex={1}>{match.name}</Typography><IconButton size="small" aria-label={'Actions for ' + match.name} onClick={(e) => { setSelected(match); setAnchor(e.currentTarget) }}><MoreHorizIcon /></IconButton></Stack>
        <Stack direction="row" flexWrap="wrap" gap={.75} mb={2}>{[
          ['Tracking', match.media.some((item) => item.role === 'tracking')], ['Dual lens', match.media.some((item) => item.role === 'dual_lens')], ['Calibration', !!match.calibration],
        ].map(([label, available]) => <Chip key={String(label)} size="small" variant="outlined" sx={{ color: available ? 'primary.main' : 'text.disabled', borderColor: available ? 'primary.dark' : 'divider' }} label={(available ? '✓ ' : '– ') + label} />)}</Stack>
        <Stack direction="row" justifyContent="space-between" alignItems="center"><Button size="small" variant="contained" onClick={() => onOpen(match.id)}>Open match</Button>{match.import_status && match.import_status !== 'completed' && <Typography variant="caption" color="text.secondary">{['queued', 'downloading'].includes(match.import_status) ? 'Import in progress' : 'Import incomplete'}</Typography>}</Stack>
      </Paper>)}</Box>}
    <Menu anchorEl={anchor} open={!!anchor} onClose={() => setAnchor(null)}><MenuItem onClick={() => { setName(selected?.name || ''); setError(''); setDialog('rename'); setAnchor(null) }}>Rename</MenuItem><MenuItem onClick={() => { if (selected) void api.match(selected.id).then(setFiles).catch((e: Error) => setError(e.message)); setAnchor(null) }}>Manage files</MenuItem>{selected?.metadata?.veo_url && <MenuItem onClick={() => { setRetryMatch(selected); setVeoOpen(true); setAnchor(null) }}>Resume Veo import</MenuItem>}<MenuItem sx={{ color: 'error.main' }} onClick={() => { setError(''); setDialog('delete'); setAnchor(null) }}>Delete match</MenuItem></Menu>
    {error && !dialog && <Alert severity="error" sx={{ mt: 2 }} onClose={() => setError('')}>{error}</Alert>}
    <Dialog open={!!dialog} onClose={saving ? undefined : () => setDialog(null)} fullWidth maxWidth="xs"><DialogTitle>{dialog === 'delete' ? 'Delete “' + selected?.name + '”?' : dialog === 'rename' ? 'Rename match' : 'New Match'}</DialogTitle><DialogContent><Stack gap={2} mt={1}>{error && <Alert severity="error">{error}</Alert>}{dialog === 'delete' ? <Typography variant="body2">This removes files managed by Veo Match Lab. External referenced files will not be deleted.</Typography> : <><TextField autoFocus required label="Match name" value={name} onChange={(e) => setName(e.target.value)} onKeyDown={(e) => { if (e.key === 'Enter' && name.trim() && !saving) void submit() }} />{dialog === 'new' && <Typography variant="body2" color="text.secondary">Next, optionally add Tracking, Dual lens, Calibration, clips or images.</Typography>}</>}</Stack></DialogContent><DialogActions><Button disabled={saving} onClick={() => setDialog(null)}>Cancel</Button><Button variant="contained" color={dialog === 'delete' ? 'error' : 'primary'} disabled={saving || (dialog !== 'delete' && !name.trim())} onClick={() => void submit()}>{dialog === 'delete' ? 'Delete match' : dialog === 'rename' ? 'Save' : 'Create match'}</Button></DialogActions></Dialog>
    {files && <ManageFilesDialog match={files} calibrations={calibrations} onUpdated={(match) => { setFiles(match); void onRefresh() }} onClose={() => { setFiles(null); void onRefresh() }} />}
    {veoOpen && <VeoImportDialog key={retryMatch?.id || 'new'} open retryMatch={retryMatch} onClose={() => { setVeoOpen(false); void onRefresh() }} onComplete={(id) => { setVeoOpen(false); onOpen(id) }} />}
  </Box>
}
