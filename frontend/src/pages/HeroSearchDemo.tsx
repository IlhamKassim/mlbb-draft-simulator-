import { useState } from 'react';
import { Container, Typography, Box, Paper } from '@mui/material';
import HeroSearch from '../components/HeroSearch';
import { Hero } from '../types/hero';

// Sample heroes data
const sampleHeroes: Hero[] = [
  { id: '1', name: 'Layla', role: 'Marksman', iconUrl: '/static/img/layla.png' },
  { id: '2', name: 'Tigreal', role: 'Tank', iconUrl: '/static/img/tigreal.png' },
  { id: '3', name: 'Eudora', role: 'Mage', iconUrl: '/static/img/eudora.png' },
  { id: '4', name: 'Zilong', role: 'Fighter', iconUrl: '/static/img/zilong.png' },
  { id: '5', name: 'Saber', role: 'Assassin', iconUrl: '/static/img/saber.png' },
  { id: '6', name: 'Rafaela', role: 'Support', iconUrl: '/static/img/rafaela.png' },
];

const HeroSearchDemo = () => {
  const [selectedHero, setSelectedHero] = useState<Hero | null>(null);

  const handleHeroSelect = (hero: Hero) => {
    setSelectedHero(hero);
  };

  return (
    <Container maxWidth="md">
      <Box py={4}>
        <Typography variant="h4" gutterBottom>
          Hero Search Demo
        </Typography>

        <Paper sx={{ p: 2, mb: 2 }}>
          <HeroSearch
            heroes={sampleHeroes}
            onSelect={handleHeroSelect}
            placeholder="Search for a hero..."
          />
        </Paper>

        {selectedHero && (
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6">Selected Hero:</Typography>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mt: 1 }}>
              <img
                src={selectedHero.iconUrl}
                alt={selectedHero.name}
                style={{ width: 60, height: 60, borderRadius: 8 }}
              />
              <Box>
                <Typography variant="subtitle1">{selectedHero.name}</Typography>
                <Typography variant="body2" color="text.secondary">
                  {selectedHero.role}
                </Typography>
              </Box>
            </Box>
          </Paper>
        )}
      </Box>
    </Container>
  );
};

export default HeroSearchDemo;