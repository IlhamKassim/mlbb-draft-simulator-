import { useState, useEffect, useCallback } from 'react'
import {
  Container,
  Paper,
  Typography,
  Box,
  Button,
  Fade,
  Divider,
  useTheme,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Tooltip,
  IconButton,
  SelectChangeEvent
} from '@mui/material'
import RestartAltIcon from '@mui/icons-material/RestartAlt'
import AutorenewIcon from '@mui/icons-material/Autorenew'
import InfoOutlinedIcon from '@mui/icons-material/InfoOutlined'
import HelpOutlineIcon from '@mui/icons-material/HelpOutline'
import DraftBoard from '../components/DraftBoard'
import HeroGrid from '../components/HeroGrid'
import WinProbabilityGauge from '../components/WinProbabilityGauge'
import { useHeroes, useDraftPrediction } from '../api/api'
import { mlbbColors } from '../theme'
import LoadingSkeleton from '../components/LoadingSkeleton'
import PhaseIndicator from '../components/PhaseIndicator'

interface HeroStats {
  role: string;
  winRate?: number;
  pickRate?: number;
  banRate?: number;
}

// Test array with minimal hero data for debugging
const testHeroes = [
  { name: 'Test Hero 1', role: 'Tank' },
  { name: 'Test Hero 2', role: 'Fighter' },
  { name: 'Test Hero 3', role: 'Marksman' }
];

// Debug code: Use test heroes instead of API data
const useTestHeroes = false; // Set to false to use real data

export default function Draft() {
  const theme = useTheme()
  const [draftState, setDraftState] = useState({
    blue_picks: [] as string[],
    red_picks: [] as string[],
    blue_bans: [] as string[],
    red_bans: [] as string[]
  })
  const [currentPhase, setCurrentPhase] = useState<'ban' | 'pick'>('ban')
  const [blueTurn, setBlueTurn] = useState(true)
  const [currentPatch, setCurrentPatch] = useState<string>('current')
  const [triggerAnimation, setTriggerAnimation] = useState(false)

  const { data: heroStats, isLoading: isLoadingStats } = useHeroes()
  const prediction = useDraftPrediction()

  // Log hero data whenever it changes
  useEffect(() => {
    console.log('Hero Stats:', heroStats)
    console.log('Is Loading Stats:', isLoadingStats)
    
    // Debug processed data
    if (heroStats) {
      const processed = formatHeroStats();
      console.log('Processed Hero Stats:', processed);
      console.log('First hero:', processed[0]);
      
      const objectFormat = getHeroStatsObject();
      console.log('Object Format for DraftBoard:', objectFormat);
    }
  }, [heroStats, isLoadingStats])

  // Debug output of formatHeroStats function with test data
  useEffect(() => {
    if (useTestHeroes) {
      console.log('DEBUG: Using test heroes instead of API data');
      console.log('Test Heroes:', testHeroes);
    }
  }, []);

  // Trigger win probability refresh on any draft state change
  useEffect(() => {
    // Call prediction API if there's at least one hero picked/banned
    if (draftState.blue_picks.length + draftState.red_picks.length + 
        draftState.blue_bans.length + draftState.red_bans.length > 0) {
      
      // Add patch version to API call if selected
      const apiPayload = {
        ...draftState,
        ...(currentPatch !== 'current' && { patch_version: currentPatch })
      };
      
      // Trigger animation for gauge
      setTriggerAnimation(prev => !prev);
      
      // Call API
      prediction.mutate(apiPayload);
    }
  }, [draftState, currentPatch])

  // Helper function to determine the next team and phase
  const updateDraftProgress = useCallback(() => {
    // Current values
    let newBlueTurn = blueTurn;
    let newPhase = currentPhase;
    
    // Standard MLBB draft rules
    if (newPhase === 'ban') {
      const totalBans = draftState.blue_bans.length + draftState.red_bans.length;
      if (totalBans < 6) {
        // Still in ban phase, alternate turns
        newBlueTurn = !newBlueTurn;
      } else {
        // Ban phase complete, start pick phase with blue
        newPhase = 'pick';
        newBlueTurn = true;
      }
    } else {
      // Pick phase
      const totalPicks = draftState.blue_picks.length + draftState.red_picks.length;
      if (totalPicks < 10) {
        // Mobile Legends draft order: B R R B B R R B B R
        // Blue team picks 1st, 4th, 5th, 8th, 9th (indices 0, 3, 4, 7, 8)
        const bluePickIndices = [0, 3, 4, 7, 8];
        newBlueTurn = bluePickIndices.includes(totalPicks);
      }
    }
    
    // Update state
    setBlueTurn(newBlueTurn);
    setCurrentPhase(newPhase);
  }, [draftState, currentPhase, blueTurn]);

  const handleHeroSelect = (hero: string) => {
    setDraftState(prev => {
      const newState = { ...prev }
      if (currentPhase === 'pick') {
        if (blueTurn) {
          newState.blue_picks = [...prev.blue_picks, hero]
        } else {
          newState.red_picks = [...prev.red_picks, hero]
        }
      } else {
        if (blueTurn) {
          newState.blue_bans = [...prev.blue_bans, hero]
        } else {
          newState.red_bans = [...prev.red_bans, hero]
        }
      }
      return newState
    })

    // Update draft progress (turn and phase)
    updateDraftProgress();
  }

  const resetDraft = () => {
    setDraftState({
      blue_picks: [],
      red_picks: [],
      blue_bans: [],
      red_bans: []
    })
    setCurrentPhase('ban')
    setBlueTurn(true)
  }

  const forceRefresh = () => {
    const apiPayload = {
      ...draftState,
      ...(currentPatch !== 'current' && { patch_version: currentPatch })
    };
    setTriggerAnimation(prev => !prev);
    prediction.mutate(apiPayload);
  }

  const handlePatchChange = (event: SelectChangeEvent<string>) => {
    setCurrentPatch(event.target.value);
    // Refresh win probability with new patch
    forceRefresh();
  };

  const getDisabledHeroes = () => [
    ...draftState.blue_picks,
    ...draftState.red_picks,
    ...draftState.blue_bans,
    ...draftState.red_bans
  ]

  const formatHeroStats = () => {
    if (!heroStats) return []
    
    // Handle different API response formats
    if (Array.isArray(heroStats)) {
      // This is the /draft/heroes endpoint format
      return heroStats.map(hero => ({
        name: hero.name,
        role: Array.isArray(hero.roles) && hero.roles.length > 0 ? hero.roles[0] : 'Unknown',
        winRate: hero.winRate,
        pickRate: hero.pickRate,
        banRate: hero.banRate,
        imageUrl: hero.imageUrl
      }));
    } else if (heroStats.hero_stats && heroStats.hero_stats.heroes) {
      // This is the /stats endpoint format
      const heroes = heroStats.hero_stats.heroes;
      const winRates = heroStats.hero_stats.win_rates || {};
      const pickRates = heroStats.hero_stats.pick_rates || {};
      const banRates = heroStats.hero_stats.ban_rates || {};
      const heroRoles = heroStats.hero_stats.hero_roles || {};
      
      return heroes.map(name => ({
        name,
        role: heroRoles[name] ? heroRoles[name][0] : 'Unknown',
        winRate: winRates[name] || 0,
        pickRate: pickRates[name] || 0,
        banRate: banRates[name] || 0
      }));
    } else {
      // Fallback format
      return Object.entries(heroStats).map(([name, stats]) => ({
        name,
        role: (stats as HeroStats).role || 'Unknown',
        winRate: (stats as HeroStats).winRate,
        pickRate: (stats as HeroStats).pickRate,
        banRate: (stats as HeroStats).banRate
      }));
    }
  }

  // Convert hero array to object format for DraftBoard
  const getHeroStatsObject = () => {
    if (!heroStats || !Array.isArray(heroStats)) return {};
    
    return heroStats.reduce((acc, hero) => {
      acc[hero.name] = {
        role: Array.isArray(hero.roles) && hero.roles.length > 0 ? hero.roles[0] : 'Unknown',
        winRate: hero.winRate,
        pickRate: hero.pickRate,
        banRate: hero.banRate,
        imageUrl: hero.imageUrl
      };
      return acc;
    }, {} as Record<string, { 
      role?: string, 
      winRate?: number,
      pickRate?: number,
      banRate?: number,
      imageUrl?: string
    }>);
  }

  // Enhanced debug function for inspecting hero object structure
  const logHeroStructure = (heroes: any[] | undefined) => {
    if (!heroes || heroes.length === 0) {
      console.log('Hero array is empty or undefined');
      return;
    }
    
    const firstHero = heroes[0];
    console.log('First hero object structure:', JSON.stringify(firstHero, null, 2));
    console.log('Hero properties:', Object.keys(firstHero));
    console.log('Total heroes:', heroes.length);
    
    // Check if heroes have required properties for rendering
    const missingProps = heroes.filter(h => !h.name).length;
    if (missingProps > 0) {
      console.error(`WARNING: ${missingProps} heroes missing required 'name' property!`);
    }
  };
  
  useEffect(() => {
    if (heroStats && Array.isArray(heroStats)) {
      console.log('Analyzing API hero data structure:');
      logHeroStructure(heroStats);
    }
  }, [heroStats]);

  // Available patches for selection (example)
  const patchVersions = [
    { value: 'current', label: 'Current Patch' },
    { value: '1.7.42', label: 'Patch 1.7.42' },
    { value: '1.7.36', label: 'Patch 1.7.36' },
    { value: '1.7.28', label: 'Patch 1.7.28' }
  ];

  return (
    <Container maxWidth="xl" sx={{ mt: 4, pb: 4 }}>
      <Box sx={{ mb: 4, display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 2 }}>
        <Typography variant="h4">
          Draft Simulator
        </Typography>
        
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
          {/* Patch Version Selector */}
          <FormControl size="small" sx={{ minWidth: 150 }}>
            <InputLabel id="patch-version-label">Patch Version</InputLabel>
            <Select
              labelId="patch-version-label"
              value={currentPatch}
              label="Patch Version"
              onChange={handlePatchChange}
            >
              {patchVersions.map(patch => (
                <MenuItem key={patch.value} value={patch.value}>
                  {patch.label}
                </MenuItem>
              ))}
            </Select>
          </FormControl>
          
          <Box sx={{ display: 'flex', gap: 1 }}>
            <Tooltip title="Recalculate win probabilities and recommendations">
              <Button
                variant="outlined"
                color="primary"
                startIcon={<AutorenewIcon />}
                onClick={forceRefresh}
                disabled={prediction.isLoading || 
                  (draftState.blue_picks.length === 0 && 
                   draftState.red_picks.length === 0 &&
                   draftState.blue_bans.length === 0 &&
                   draftState.red_bans.length === 0)}
              >
                Refresh
              </Button>
            </Tooltip>
            <Tooltip title="Start a new draft">
              <Button
                variant="outlined"
                color="error"
                startIcon={<RestartAltIcon />}
                onClick={resetDraft}
              >
                Reset
              </Button>
            </Tooltip>
          </Box>
        </Box>
      </Box>

      {/* Phase Indicator */}
      <Box sx={{ mb: 4 }}>
        <PhaseIndicator
          phase={currentPhase}
          totalBans={draftState.blue_bans.length + draftState.red_bans.length}
          totalPicks={draftState.blue_picks.length + draftState.red_picks.length}
          blueTurn={blueTurn}
        />
      </Box>

      <Box sx={{ display: 'flex', gap: 4, flexDirection: { xs: 'column', md: 'row' } }}>
        {/* Main Draft Board */}
        <Box sx={{ flex: 1 }}>
          {isLoadingStats ? (
            <LoadingSkeleton type="draft-board" />
          ) : (
            <DraftBoard
              bluePicks={draftState.blue_picks}
              redPicks={draftState.red_picks}
              blueBans={draftState.blue_bans}
              redBans={draftState.red_bans}
              blueTurn={blueTurn}
              heroStats={getHeroStatsObject()}
            />
          )}
        </Box>

        {/* Win Probability & Recommendations */}
        <Paper 
          sx={{ 
            width: { xs: '100%', md: '300px' },
            p: 3,
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            gap: 3,
            background: `linear-gradient(135deg, 
              ${theme.palette.background.paper} 0%, 
              ${theme.palette.background.paper} 80%,
              ${mlbbColors.blue.main}20 100%
            )`
          }}
          elevation={3}
        >
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <Typography variant="h6" align="center">
              Win Probability
            </Typography>
            <Tooltip title="The win probability is calculated based on hero picks, bans, synergies, and counter relationships. Values closer to 50% indicate a balanced draft.">
              <IconButton size="small">
                <HelpOutlineIcon fontSize="small" />
              </IconButton>
            </Tooltip>
          </Box>
          
          <WinProbabilityGauge
            blueWinProbability={prediction.data?.blue_win_probability || 0.5}
            isLoading={prediction.isLoading}
            size={180}
            triggerAnimation={triggerAnimation}
          />

          {prediction.isLoading ? (
            <LoadingSkeleton type="recommendations" count={3} />
          ) : prediction.data?.recommendations ? (
            <>
              <Divider sx={{ width: '100%' }} />
              <Box sx={{ width: '100%' }}>
                <Typography variant="subtitle1" gutterBottom sx={{ display: 'flex', alignItems: 'center' }}>
                  Recommended Heroes
                  <Tooltip title="These heroes have the highest predicted win probability for the current draft state">
                    <InfoOutlinedIcon fontSize="small" sx={{ ml: 1, opacity: 0.7 }} />
                  </Tooltip>
                </Typography>
                {prediction.data.recommendations.map((rec, idx) => (
                  <Fade key={rec.hero} in timeout={300 * (idx + 1)}>
                    <Box
                      sx={{
                        p: 1.5,
                        mb: 1,
                        borderRadius: 1,
                        border: `1px solid ${theme.palette.divider}`,
                        cursor: 'pointer',
                        transition: 'all 0.2s ease',
                        '&:hover': {
                          bgcolor: 'action.hover',
                          transform: 'translateX(4px)'
                        },
                        display: 'flex',
                        justifyContent: 'space-between',
                        alignItems: 'center'
                      }}
                      onClick={() => handleHeroSelect(rec.hero)}
                    >
                      <Box>
                        <Typography variant="body2">
                          {rec.hero}
                        </Typography>
                        <Typography variant="caption" color="text.secondary">
                          {rec.description}
                        </Typography>
                      </Box>
                      <Typography
                        variant="body2"
                        sx={{
                          fontWeight: 'bold',
                          color: theme.palette.primary.main
                        }}
                      >
                        {(rec.win_probability * 100).toFixed(1)}%
                      </Typography>
                    </Box>
                  </Fade>
                ))}
              </Box>
            </>
          ) : null}
        </Paper>
      </Box>

      {/* Hero Selection Grid */}
      <Paper 
        sx={{ 
          mt: 4, 
          p: 3,
          background: `linear-gradient(135deg,
            ${theme.palette.background.paper} 0%,
            ${theme.palette.background.paper} 95%,
            ${blueTurn ? mlbbColors.blue.main : mlbbColors.red.main}20 100%
          )`,
          transition: 'background 0.3s ease'
        }}
        elevation={2}
      >
        <Typography variant="h6" gutterBottom color={blueTurn ? 'primary' : 'error'}>
          {blueTurn ? 'Blue' : 'Red'} Team's Turn • {currentPhase.toUpperCase()}
        </Typography>
        {isLoadingStats ? (
          <LoadingSkeleton type="hero-card" count={12} />
        ) : (
          <HeroGrid
            heroes={formatHeroStats()}
            onHeroSelect={handleHeroSelect}
            disabledHeroes={getDisabledHeroes()}
            isLoading={isLoadingStats}
          />
        )}
      </Paper>
    </Container>
  )
}