import { createTheme, alpha } from '@mui/material/styles';
import { PaletteMode } from '@mui/material';

// MLBB-inspired color palette
const mlbbColors = {
  blue: {
    main: '#1976d2',
    light: '#42a5f5',
    dark: '#1565c0',
    contrastText: '#ffffff'
  },
  red: {
    main: '#d32f2f',
    light: '#ef5350',
    dark: '#c62828',
    contrastText: '#ffffff'
  },
  gold: {
    main: '#ffd700',
    light: '#ffe44d',
    dark: '#c7a900',
    contrastText: '#000000'
  }
};

// Design tokens
const tokens = {
  borderRadius: {
    sm: '4px',
    md: '8px',
    lg: '12px'
  },
  spacing: {
    xs: '4px',
    sm: '8px',
    md: '16px',
    lg: '24px',
    xl: '32px'
  },
  animation: {
    fast: '150ms',
    medium: '300ms',
    slow: '500ms'
  }
};

// Theme generator based on mode
const getTheme = (mode: PaletteMode) => createTheme({
  palette: {
    mode,
    primary: mlbbColors.blue,
    error: mlbbColors.red,
    warning: mlbbColors.gold,
    background: {
      default: mode === 'light' ? '#f5f5f5' : '#121212',
      paper: mode === 'light' ? '#ffffff' : '#1e1e1e'
    },
    text: {
      primary: mode === 'light' ? 'rgba(0, 0, 0, 0.87)' : 'rgba(255, 255, 255, 0.87)',
      secondary: mode === 'light' ? 'rgba(0, 0, 0, 0.6)' : 'rgba(255, 255, 255, 0.6)'
    }
  },
  typography: {
    fontFamily: '"Roboto", "Helvetica", "Arial", sans-serif',
    h1: {
      fontSize: '2.5rem',
      fontWeight: 600
    },
    h2: {
      fontSize: '2rem',
      fontWeight: 600
    },
    h3: {
      fontSize: '1.75rem',
      fontWeight: 600
    },
    h4: {
      fontSize: '1.5rem',
      fontWeight: 600
    },
    h5: {
      fontSize: '1.25rem',
      fontWeight: 600
    },
    h6: {
      fontSize: '1rem',
      fontWeight: 600
    }
  },
  components: {
    MuiCard: {
      styleOverrides: {
        root: ({ theme }) => ({
          borderRadius: tokens.borderRadius.md,
          transition: `all ${tokens.animation.medium} ease-in-out`,
          '&:hover': {
            boxShadow: theme.shadows[4]
          }
        })
      }
    },
    MuiButton: {
      styleOverrides: {
        root: {
          borderRadius: tokens.borderRadius.sm,
          textTransform: 'none',
          transition: `all ${tokens.animation.fast} ease-in-out`
        },
        containedPrimary: ({ theme }) => ({
          background: `linear-gradient(45deg, ${theme.palette.primary.main}, ${theme.palette.primary.light})`,
          '&:hover': {
            background: `linear-gradient(45deg, ${theme.palette.primary.dark}, ${theme.palette.primary.main})`
          }
        })
      }
    },
    MuiPaper: {
      styleOverrides: {
        root: ({ theme }) => ({
          backgroundImage: 'none',
          ...(theme.palette.mode === 'dark' && {
            backgroundColor: alpha(theme.palette.background.paper, 0.9)
          })
        })
      }
    },
    MuiChip: {
      styleOverrides: {
        root: {
          borderRadius: tokens.borderRadius.sm
        }
      }
    },
    MuiTooltip: {
      styleOverrides: {
        tooltip: ({ theme }) => ({
          backgroundColor: alpha(theme.palette.background.paper, 0.95),
          border: `1px solid ${theme.palette.divider}`,
          borderRadius: tokens.borderRadius.sm,
          boxShadow: theme.shadows[2],
          color: theme.palette.text.primary,
          padding: tokens.spacing.md
        })
      }
    }
  }
});

export { getTheme, tokens, mlbbColors };