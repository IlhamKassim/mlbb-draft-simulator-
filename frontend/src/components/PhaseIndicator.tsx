import { Box, Typography, LinearProgress, useTheme } from '@mui/material';
import { mlbbColors } from '../theme';

interface PhaseIndicatorProps {
  phase: 'ban' | 'pick';
  totalBans: number;
  totalPicks: number;
  blueTurn: boolean;
}

export default function PhaseIndicator({ 
  phase, 
  totalBans, 
  totalPicks,
  blueTurn 
}: PhaseIndicatorProps) {
  const theme = useTheme();

  const getProgress = () => {
    if (phase === 'ban') {
      return (totalBans / 6) * 100;
    }
    return ((totalPicks / 10) * 100);
  };

  const getStatusText = () => {
    const team = blueTurn ? 'Blue' : 'Red';
    const action = phase === 'ban' ? 'Banning' : 'Picking';
    return `${team} Team ${action}`;
  };

  return (
    <Box 
      sx={{
        p: 2,
        borderRadius: 1,
        background: `linear-gradient(135deg,
          ${theme.palette.background.paper} 0%,
          ${theme.palette.background.paper} 95%,
          ${blueTurn ? mlbbColors.blue.main : mlbbColors.red.main}40 100%
        )`,
        border: `1px solid ${theme.palette.divider}`,
        transition: 'all 0.3s ease'
      }}
    >
      <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
        <Typography 
          variant="subtitle1"
          sx={{ 
            color: blueTurn ? mlbbColors.blue.main : mlbbColors.red.main,
            transition: 'color 0.3s ease'
          }}
        >
          {getStatusText()}
        </Typography>
        <Typography variant="subtitle1" color="text.secondary">
          {phase === 'ban' ? `${totalBans}/6` : `${totalPicks}/10`}
        </Typography>
      </Box>
      <LinearProgress
        variant="determinate"
        value={getProgress()}
        sx={{
          height: 8,
          borderRadius: 4,
          backgroundColor: theme.palette.mode === 'dark' 
            ? 'rgba(255, 255, 255, 0.1)' 
            : 'rgba(0, 0, 0, 0.1)',
          '& .MuiLinearProgress-bar': {
            backgroundColor: blueTurn ? mlbbColors.blue.main : mlbbColors.red.main,
            transition: 'background-color 0.3s ease'
          }
        }}
      />
      <Box sx={{ mt: 2, display: 'flex', gap: 1 }}>
        {Array(phase === 'ban' ? 6 : 10).fill(null).map((_, i) => (
          <Box
            key={i}
            sx={{
              width: 8,
              height: 8,
              borderRadius: '50%',
              backgroundColor: i < (phase === 'ban' ? totalBans : totalPicks)
                ? blueTurn ? mlbbColors.blue.main : mlbbColors.red.main
                : theme.palette.mode === 'dark' 
                  ? 'rgba(255, 255, 255, 0.1)' 
                  : 'rgba(0, 0, 0, 0.1)',
              transition: 'all 0.3s ease'
            }}
          />
        ))}
      </Box>
    </Box>
  );
}