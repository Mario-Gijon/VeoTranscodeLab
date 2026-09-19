import { Box, Chip, Paper, Stack, Table, TableBody, TableCell, TableContainer, TableHead, TableRow, Typography } from '@mui/material'
import type { GroundPoint } from '../types'

const coordinate = (x: number | null, z: number | null) => x === null || z === null ? '—' : `${x.toFixed(2)}, ${z.toFixed(2)}`
const statusColor = (status: GroundPoint['status']) => status === 'inside_field' ? 'success' : status === 'outside_field' ? 'warning' : 'default'

export default function ValidationTable({ points }: { points: GroundPoint[] }) {
  const errors = points.map((point) => point.field_error_m).filter((value): value is number => value !== null)
  const mean = errors.length ? errors.reduce((total, value) => total + value, 0) / errors.length : null
  const ordered = [...errors].sort((a, b) => a - b)
  const measured = points.filter((point) => point.field_error_m !== null)
  const median = errors.length ? (ordered[Math.floor((errors.length - 1) / 2)] + ordered[Math.floor(errors.length / 2)]) / 2 : 0
  const rms = errors.length ? Math.sqrt(errors.reduce((sum, value) => sum + value ** 2, 0) / errors.length) : 0
  const meanX = measured.reduce((sum, point) => sum + (point.delta_x ?? 0), 0) / (measured.length || 1)
  const meanZ = measured.reduce((sum, point) => sum + (point.delta_z ?? 0), 0) / (measured.length || 1)
  return <Paper variant="outlined" sx={{ overflow: 'hidden' }}>
    {errors.length > 0 && <Typography variant="body2" sx={{ px: 1.5, pt: 1 }}>Landmarks n={errors.length} · median {median.toFixed(3)} m · RMS {rms.toFixed(3)} m · max {Math.max(...errors).toFixed(3)} m · mean ΔX {meanX.toFixed(3)} m · mean ΔZ {meanZ.toFixed(3)} m</Typography>}
    <Stack direction="row" justifyContent="space-between" alignItems="center" p={1.5} gap={1} flexWrap="wrap">
      <Typography variant="subtitle1" fontWeight={700}>Validation points</Typography>
      <Stack direction="row" gap={.75}><Chip size="small" label={`${points.length} selected`} /><Chip size="small" color="secondary" label={mean === null ? 'No landmark error yet' : `Mean error ${mean.toFixed(2)} m`} /></Stack>
    </Stack>
    {points.length === 0 ? <Typography color="text.secondary" sx={{ px: 1.5, pb: 2 }}>Click the panorama or a raw lens to add validation points.</Typography> :
      <TableContainer sx={{ maxHeight: 280 }}><Table stickyHeader size="small"><TableHead><TableRow><TableCell>#</TableCell><TableCell>View / time</TableCell><TableCell>Landmark</TableCell><TableCell>Pixel / UV</TableCell><TableCell>Expected X/Z</TableCell><TableCell>Measured X/Z</TableCell><TableCell>Δ X/Z</TableCell><TableCell>Error</TableCell><TableCell>Status</TableCell></TableRow></TableHead>
        <TableBody>{points.map((point, index) => <TableRow key={`${index}-${point.pixel_x}-${point.pixel_y}`} hover><TableCell>{point.point_index ?? index + 1}</TableCell><TableCell>{point.view?.replaceAll('_', ' ') ?? 'panorama'}<br />{point.timestamp_s?.toFixed(3) ?? '—'} s</TableCell><TableCell>{point.landmark_name ?? 'Free point'}</TableCell><TableCell>({point.pixel_x}, {point.pixel_y})<br /><Box component="span" color="text.secondary">{point.u.toFixed(5)}, {point.v.toFixed(5)}</Box></TableCell><TableCell>{coordinate(point.expected_x, point.expected_z)}</TableCell><TableCell>{coordinate(point.measured_x, point.measured_z)}</TableCell><TableCell>{coordinate(point.delta_x, point.delta_z)}</TableCell><TableCell>{point.field_error_m === null ? '—' : `${point.field_error_m.toFixed(2)} m`}</TableCell><TableCell><Chip size="small" color={statusColor(point.status)} label={point.status.replaceAll('_', ' ')} /></TableCell></TableRow>)}</TableBody>
      </Table></TableContainer>}
  </Paper>
}
