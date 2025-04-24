import { AppBar, Toolbar, Typography, Button, Box, IconButton, Tooltip } from '@mui/material'
import { useNavigate, useLocation } from 'react-router-dom'
import LightModeIcon from '@mui/icons-material/LightMode'
import DarkModeIcon from '@mui/icons-material/DarkMode'
import { useThemeContext } from '../contexts/ThemeContext'
import { mlbbColors } from '../theme'

export default function NavBar() {
  const navigate = useNavigate()
  const location = useLocation()
  const { mode, toggleTheme } = useThemeContext()

  const routes = [
    { path: '/', label: 'Home' },
    { path: '/draft', label: 'Draft Simulator' },
    { path: '/analytics', label: 'Analytics' },
    { path: '/roles', label: 'Roles' }
  ]

  return (
    <AppBar 
      position="sticky"
      sx={{
        background: `linear-gradient(90deg, ${mlbbColors.blue.main} 0%, ${mlbbColors.red.main} 100%)`
      }}
    >
      <Toolbar>
        <Typography variant="h6" sx={{ flexGrow: 0, mr: 4 }}>
          MLBB Counter
        </Typography>
        <Box sx={{ display: 'flex', gap: 2, flexGrow: 1 }}>
          {routes.map(route => (
            <Button
              key={route.path}
              color="inherit"
              onClick={() => navigate(route.path)}
              variant={location.pathname === route.path ? 'outlined' : 'text'}
              sx={{
                borderColor: 'rgba(255, 255, 255, 0.5)',
                '&:hover': {
                  borderColor: 'white'
                }
              }}
            >
              {route.label}
            </Button>
          ))}
        </Box>
        <Tooltip title={`Switch to ${mode === 'light' ? 'dark' : 'light'} mode`}>
          <IconButton color="inherit" onClick={toggleTheme}>
            {mode === 'light' ? <DarkModeIcon /> : <LightModeIcon />}
          </IconButton>
        </Tooltip>
      </Toolbar>
    </AppBar>
  )
}