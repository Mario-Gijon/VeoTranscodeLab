import { CssBaseline, ThemeProvider, createTheme } from '@mui/material'
import { createRoot } from 'react-dom/client'
import App from './App'

const theme = createTheme({
  palette: { mode: 'dark', primary: { main: '#78a9ff' }, secondary: { main: '#73d7ad' }, background: { default: '#141619', paper: '#1e2125' } },
  shape: { borderRadius: 4 },
  typography: { fontFamily: 'Inter, ui-sans-serif, system-ui, sans-serif', h5: { fontWeight: 700 } },
  components: { MuiPaper: { styleOverrides: { root: { backgroundImage: 'none' } } }, MuiCard: { styleOverrides: { root: { backgroundImage: 'none', border: '1px solid #273548' } } }, MuiTableCell: { styleOverrides: { root: { borderColor: '#2a3748' } } } },
})

createRoot(document.getElementById('root')!).render(<ThemeProvider theme={theme}><CssBaseline /><App /></ThemeProvider>)
