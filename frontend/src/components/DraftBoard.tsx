import { Box, Paper, Typography, Grid, useTheme, Grow } from '@mui/material';
import HeroCard from './HeroCard';
import { mlbbColors } from '../theme';

interface DraftBoardProps {
  bluePicks: Array<string>;
  redPicks: Array<string>;
  blueBans: Array<string>;
  redBans: Array<string>;
  blueTurn: boolean;
  heroStats?: Record<string, {
    role?: string;
    winRate?: number;
    pickRate?: number;
    banRate?: number;
  }>;
}

export default function DraftBoard({
  bluePicks,
  redPicks,
  blueBans,
  redBans,
  blueTurn,
  heroStats = {}
}: DraftBoardProps) {
  const theme = useTheme();

  // Debug logs for DraftBoard
  console.log('DraftBoard - heroStats object:', heroStats);
  console.log('DraftBoard - Number of heroes in heroStats:', Object.keys(heroStats).length);
  console.log('DraftBoard - bluePicks:', bluePicks);
  console.log('DraftBoard - redPicks:', redPicks);

  const renderTeamSection = (
    team: 'blue' | 'red',
    picks: string[],
    bans: string[]
  ) => {
    const isBlue = team === 'blue';
    const teamColor = isBlue ? mlbbColors.blue : mlbbColors.red;

    return (
      <Box sx={{ width: '100%' }}>
        <Paper
          elevation={2}
          sx={{
            p: 2,
            background: `linear-gradient(45deg, 
              ${theme.palette.background.paper} 0%, 
              ${theme.palette.background.paper} 95%, 
              ${teamColor.main}40 100%
            )`,
            borderLeft: `4px solid ${teamColor.main}`,
            mb: 2
          }}
        >
          <Typography
            variant="h6"
            sx={{
              color: teamColor.main,
              display: 'flex',
              alignItems: 'center',
              gap: 1
            }}
          >
            {team.toUpperCase()} TEAM
            {blueTurn === isBlue && (
              <Box
                sx={{
                  width: 8,
                  height: 8,
                  borderRadius: '50%',
                  backgroundColor: teamColor.main,
                  animation: `${theme.transitions.create(['opacity'], {
                    duration: '1s',
                    easing: theme.transitions.easing.easeInOut
                  })} infinite alternate`
                }}
              />
            )}
          </Typography>

          {/* Picks Section */}
          <Typography
            variant="subtitle2"
            color="text.secondary"
            sx={{ mt: 2, mb: 1 }}
          >
            Picks ({picks.length}/5)
          </Typography>
          <Grid container spacing={1}>
            {Array(5).fill(null).map((_, i) => (
              <Grid item xs={12/5} key={`pick-${i}`}>
                <Grow in={!!picks[i]} timeout={300 * (i + 1)}>
                  <Box>
                    {picks[i] ? (
                      <HeroCard
                        name={picks[i]}
                        team={team}
                        type="pick"
                        isSelected={true}
                        {...heroStats[picks[i]]}
                      />
                    ) : (
                      <Paper
                        sx={{
                          height: 200,
                          border: `1px dashed ${theme.palette.divider}`,
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'center'
                        }}
                      >
                        <Typography color="text.disabled">
                          Empty
                        </Typography>
                      </Paper>
                    )}
                  </Box>
                </Grow>
              </Grid>
            ))}
          </Grid>

          {/* Bans Section */}
          <Typography
            variant="subtitle2"
            color="text.secondary"
            sx={{ mt: 2, mb: 1 }}
          >
            Bans ({bans.length}/3)
          </Typography>
          <Grid container spacing={1}>
            {Array(3).fill(null).map((_, i) => (
              <Grid item xs={4} key={`ban-${i}`}>
                <Grow in={!!bans[i]} timeout={300 * (i + 1)}>
                  <Box>
                    {bans[i] ? (
                      <HeroCard
                        name={bans[i]}
                        team={team}
                        type="ban"
                        isSelected={true}
                        {...heroStats[bans[i]]}
                      />
                    ) : (
                      <Paper
                        sx={{
                          height: 200,
                          border: `1px dashed ${theme.palette.divider}`,
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'center'
                        }}
                      >
                        <Typography color="text.disabled">
                          Empty
                        </Typography>
                      </Paper>
                    )}
                  </Box>
                </Grow>
              </Grid>
            ))}
          </Grid>
        </Paper>
      </Box>
    );
  };

  return (
    <Box
      sx={{
        display: 'flex',
        gap: 2,
        p: 2,
        flexDirection: { xs: 'column', md: 'row' }
      }}
    >
      {renderTeamSection('blue', bluePicks, blueBans)}
      {renderTeamSection('red', redPicks, redBans)}
    </Box>
  );
}