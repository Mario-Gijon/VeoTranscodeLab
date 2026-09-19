import { Box, Chip, Paper, Stack, Table, TableBody, TableCell, TableContainer, TableHead, TableRow, Typography } from '@mui/material'
import type { GroundPoint } from '../types'

const coordinate = (x: number | null, z: number | null) => x === null || z === null ? '—' : `${x.toFixed(2)}, ${z.toFixed(2)}`
const statusColor = (status: GroundPoint['status']) => status === 'inside_field' ? 'success' : status === 'outside_field' ? 'warning' : 'default'

export default function ValidationTable({ points }: { points: GroundPoint[] }) {
  const errors = points.map((point) => point.field_error_m).filter((value): value is number => value !== null)
  const mean = errors.length ? errors.reduce((total, value) => total + value, 0) / errors.length : null
  return <Paper variant="outlined" sx={{ overflow: 'hidden' }}>
    <Stack direction="row" justifyContent="space-between" alignItems="center" p={1.5} gap={1} flexWrap="wrap">
      <Typography variant="subtitle1" fontWeight={700}>Validation points</Typography>
      <Stack direction="row" gap={.75}><Chip size="small" label={`${points.length} selected`} /><Chip size="small" color="secondary" label={mean === null ? 'No landmark error yet' : `Mean error ${mean.toFixed(2)} m`} /></Stack>
    </Stack>
    {points.length === 0 ? <Typography color="text.secondary" sx={{ px: 1.5, pb: 2 }}>Click the panorama to add validation points.</Typography> :
      <TableContainer sx={{ maxHeight: 280 }}><Table stickyHeader size="small"><TableHead><TableRow><TableCell>#</TableCell><TableCell>Landmark</TableCell><TableCell>Pixel / UV</TableCell><TableCell>Expected X/Z</TableCell><TableCell>Measured X/Z</TableCell><TableCell>Δ X/Z</TableCell><TableCell>Error</TableCell><TableCell>Status</TableCell></TableRow></TableHead>
        <TableBody>{points.map((point, index) => <TableRow key={`${index}-${point.pixel_x}-${point.pixel_y}`} hover><TableCell>{index + 1}</TableCell><TableCell>{point.landmark_name ?? 'Free point'}</TableCell><TableCell>({point.pixel_x}, {point.pixel_y})<br /><Box component="span" color="text.secondary">{point.u.toFixed(5)}, {point.v.toFixed(5)}</Box></TableCell><TableCell>{coordinate(point.expected_x, point.expected_z)}</TableCell><TableCell>{coordinate(point.measured_x, point.measured_z)}</TableCell><TableCell>{coordinate(point.delta_x, point.delta_z)}</TableCell><TableCell>{point.field_error_m === null ? '—' : `${point.field_error_m.toFixed(2)} m`}</TableCell><TableCell><Chip size="small" color={statusColor(point.status)} label={point.status.replaceAll('_', ' ')} /></TableCell></TableRow>)}</TableBody>
      </Table></TableContainer>}
  </Paper>
}
