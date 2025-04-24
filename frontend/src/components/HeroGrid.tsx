import { useState, useMemo } from 'react';
import { 
  Box, 
  TextField, 
  ToggleButtonGroup, 
  ToggleButton, 
  Grid,
  Fade,
  InputAdornment,
  styled,
  Typography
} from '@mui/material';
import SearchIcon from '@mui/icons-material/Search';
import HeroCard from './HeroCard';
import { tokens } from '../theme';

interface Hero {
  name: string;
  role: string;
  winRate?: number;
  pickRate?: number;
  banRate?: number;
  imageUrl?: string;
}

interface HeroGridProps {
  heroes: Hero[];
  onHeroSelect: (hero: string) => void;
  disabledHeroes?: string[];
  selectedHero?: string;
  isLoading?: boolean;
}

const StyledToggleButtonGroup = styled(ToggleButtonGroup)(({ theme }) => ({
  display: 'flex',
  flexWrap: 'wrap',
  gap: theme.spacing(1),
  marginBottom: theme.spacing(2),

  '& .MuiToggleButton-root': {
    borderRadius: tokens.borderRadius.sm,
    textTransform: 'none',
    flexGrow: 1,
    minWidth: 100,

    '&.Mui-selected': {
      background: theme.palette.primary.main,
      color: theme.palette.primary.contrastText,
      '&:hover': {
        background: theme.palette.primary.dark,
        color: theme.palette.primary.contrastText
      }
    }
  }
}));

const ROLES = ['All', 'Tank', 'Fighter', 'Assassin', 'Mage', 'Marksman', 'Support'];

export default function HeroGrid({
  heroes,
  onHeroSelect,
  disabledHeroes = [],
  selectedHero,
  isLoading = false
}: HeroGridProps) {
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedRole, setSelectedRole] = useState('All');

  const filteredHeroes = useMemo(() => {
    if (!heroes || !Array.isArray(heroes)) {
      console.log('Heroes data is not an array or is empty', heroes);
      return [];
    }

    return heroes.filter(hero => {
      // Ensure hero and hero.name exist
      if (!hero || !hero.name) return false;
      
      const matchesSearch = hero.name.toLowerCase().includes(searchQuery.toLowerCase());
      const matchesRole = selectedRole === 'All' || hero.role === selectedRole;
      return matchesSearch && matchesRole;
    });
  }, [heroes, searchQuery, selectedRole]);

  const handleRoleChange = (_: React.MouseEvent<HTMLElement>, newRole: string) => {
    if (newRole !== null) {
      setSelectedRole(newRole);
    }
  };

  if (!heroes || heroes.length === 0) {
    return (
      <Box sx={{ textAlign: 'center', py: 4 }}>
        <Typography>
          No heroes available. Please check your connection to the server.
        </Typography>
      </Box>
    );
  }

  return (
    <Box>
      {/* Search and Filter Controls */}
      <Box sx={{ mb: 3 }}>
        <TextField
          fullWidth
          variant="outlined"
          placeholder="Search heroes..."
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          sx={{ mb: 2 }}
          InputProps={{
            startAdornment: (
              <InputAdornment position="start">
                <SearchIcon />
              </InputAdornment>
            )
          }}
        />

        <StyledToggleButtonGroup
          value={selectedRole}
          exclusive
          onChange={handleRoleChange}
          aria-label="hero role filter"
        >
          {ROLES.map(role => (
            <ToggleButton key={role} value={role}>
              {role}
            </ToggleButton>
          ))}
        </StyledToggleButtonGroup>
      </Box>

      {/* Hero Grid */}
      <Grid container spacing={2}>
        {filteredHeroes.length === 0 && !isLoading ? (
          <Box sx={{ width: '100%', p: 4, textAlign: 'center' }}>
            <Typography color="text.secondary">
              No heroes match your search criteria.
            </Typography>
          </Box>
        ) : (
          filteredHeroes.map((hero) => (
            <Grid item xs={6} sm={4} md={3} lg={2} key={hero.name}>
              <Fade in={true} timeout={300}>
                <Box>
                  <HeroCard
                    name={hero.name}
                    role={hero.role}
                    winRate={hero.winRate}
                    pickRate={hero.pickRate}
                    banRate={hero.banRate}
                    isSelected={selectedHero === hero.name}
                    isDisabled={disabledHeroes.includes(hero.name)}
                    onClick={() => onHeroSelect(hero.name)}
                    isLoading={isLoading}
                    imageUrl={hero.imageUrl}
                  />
                </Box>
              </Fade>
            </Grid>
          ))
        )}
      </Grid>
    </Box>
  );
}