import { useState } from 'react'
import { Container, Typography, Box, ToggleButtonGroup, ToggleButton } from '@mui/material'
import TimeSeriesChart from '../components/Charts/TimeSeriesChart'
import Heatmap from '../components/Charts/Heatmap'
import { useRoleTrends, useRoleMatchups } from '../api/api'

type Metric = 'pick_rate' | 'win_rate' | 'ban_rate'

export default function Roles() {
  const [metric, setMetric] = useState<Metric>('pick_rate')
  
  const { data: trendData, isLoading: isLoadingTrends } = useRoleTrends()
  const { data: matchupData, isLoading: isLoadingMatchups } = useRoleMatchups()

  const handleMetricChange = (_: React.MouseEvent<HTMLElement>, newMetric: Metric) => {
    if (newMetric) setMetric(newMetric)
  }

  return (
    <Container maxWidth="xl" sx={{ mt: 4, mb: 4 }}>
      <Typography variant="h4" gutterBottom>
        Role Analysis
      </Typography>

      <Box sx={{ mb: 4 }}>
        <Typography variant="h6" gutterBottom>Role Trends Over Time</Typography>
        <Box sx={{ mb: 2 }}>
          <ToggleButtonGroup
            value={metric}
            exclusive
            onChange={handleMetricChange}
            aria-label="metric selection"
          >
            <ToggleButton value="pick_rate" aria-label="pick rate">
              Pick Rate
            </ToggleButton>
            <ToggleButton value="win_rate" aria-label="win rate">
              Win Rate
            </ToggleButton>
            <ToggleButton value="ban_rate" aria-label="ban rate">
              Ban Rate
            </ToggleButton>
          </ToggleButtonGroup>
        </Box>
        
        {isLoadingTrends ? (
          <Typography>Loading trend data...</Typography>
        ) : trendData ? (
          <TimeSeriesChart
            data={trendData}
            title={`Role ${metric.replace('_', ' ').replace(/(?:^|\s)\S/g, c => c.toUpperCase())} by Patch`}
            metric={metric}
          />
        ) : null}
      </Box>

      <Box sx={{ mt: 6 }}>
        <Typography variant="h6" gutterBottom>Role Matchup Analysis</Typography>
        {isLoadingMatchups ? (
          <Typography>Loading matchup data...</Typography>
        ) : matchupData ? (
          <Heatmap
            data={matchupData.matrix}
            xLabels={matchupData.roles}
            yLabels={matchupData.roles}
            title="Role Matchup Win Rates"
            colorScale={[
              [0, '#d32f2f'],
              [0.5, '#fff'],
              [1, '#1976d2']
            ]}
          />
        ) : null}
      </Box>
    </Container>
  )
}