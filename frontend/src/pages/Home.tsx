import { Container, Grid, Card, CardContent, CardActions, Button, Typography } from '@mui/material'
import { useNavigate } from 'react-router-dom'
import { useHeroStats, useSideBias } from '../api/api'

export default function Home() {
  const navigate = useNavigate()
  const { data: heroStats } = useHeroStats()
  const { data: sideBiasData } = useSideBias(10, 1)

  const features = [
    {
      title: 'Draft Simulator',
      description: 'Get real-time win probability predictions and hero recommendations during the draft phase',
      path: '/draft'
    },
    {
      title: 'Analytics Dashboard',
      description: 'Explore hero side bias, synergies, and counter relationships through interactive visualizations',
      path: '/analytics'
    },
    {
      title: 'Role Analysis',
      description: 'Track role trends over patches and understand role matchup dynamics',
      path: '/roles'
    }
  ]

  const getMostBiasedSide = () => {
    if (!sideBiasData?.[0]) return null
    const hero = sideBiasData[0]
    const side = hero.effect_size > 0 ? 'Blue' : 'Red'
    return `${hero.hero} (${side} side, ${Math.abs(hero.effect_size * 100).toFixed(1)}% bias)`
  }

  return (
    <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
      <Typography variant="h4" gutterBottom>
        MLBB Counter System
      </Typography>

      <Grid container spacing={3} sx={{ mb: 4 }}>
        <Grid item xs={12} md={4}>
          <Card sx={{ height: '100%' }}>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Heroes Analyzed
              </Typography>
              <Typography variant="h3" color="primary">
                {heroStats ? Object.keys(heroStats).length : '-'}
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={4}>
          <Card sx={{ height: '100%' }}>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Most Side-Biased Hero
              </Typography>
              <Typography variant="body1" color="text.secondary">
                {getMostBiasedSide() || 'Loading...'}
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={4}>
          <Card sx={{ height: '100%' }}>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Latest Update
              </Typography>
              <Typography variant="body1" color="text.secondary">
                Data updated daily with the latest match statistics
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      <Typography variant="h5" gutterBottom sx={{ mt: 6, mb: 3 }}>
        Features
      </Typography>

      <Grid container spacing={3}>
        {features.map((feature) => (
          <Grid item xs={12} md={4} key={feature.path}>
            <Card sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
              <CardContent sx={{ flexGrow: 1 }}>
                <Typography variant="h6" gutterBottom>
                  {feature.title}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  {feature.description}
                </Typography>
              </CardContent>
              <CardActions>
                <Button size="small" onClick={() => navigate(feature.path)}>
                  Explore
                </Button>
              </CardActions>
            </Card>
          </Grid>
        ))}
      </Grid>
    </Container>
  )
}