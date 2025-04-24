import { useState } from 'react'
import { Container, Tabs, Tab, Box, Typography, Slider } from '@mui/material'
import SideBiasChart from '../components/Charts/SideBiasChart'
import Heatmap from '../components/Charts/Heatmap'
import { useSideBias, useSynergyMatrix, useCounterMatrix } from '../api/api'

export default function Analytics() {
  const [tab, setTab] = useState(0)
  const [minGames, setMinGames] = useState(10)

  const { data: sideBiasData, isLoading: isLoadingSideBias } = useSideBias(minGames)
  const { data: synergyData, isLoading: isLoadingSynergy } = useSynergyMatrix(minGames)
  const { data: counterData, isLoading: isLoadingCounter } = useCounterMatrix(minGames)

  const handleTabChange = (_: React.SyntheticEvent, newValue: number) => {
    setTab(newValue)
  }

  const handleMinGamesChange = (_: Event, newValue: number | number[]) => {
    setMinGames(newValue as number)
  }

  return (
    <Container maxWidth="xl" sx={{ mt: 4, mb: 4 }}>
      <Typography variant="h4" gutterBottom>
        Hero Analytics
      </Typography>

      <Box sx={{ mb: 2 }}>
        <Typography id="min-games-slider" gutterBottom>
          Minimum Games: {minGames}
        </Typography>
        <Slider
          value={minGames}
          onChange={handleMinGamesChange}
          min={5}
          max={100}
          step={5}
          aria-labelledby="min-games-slider"
          sx={{ maxWidth: 300 }}
        />
      </Box>

      <Box sx={{ borderBottom: 1, borderColor: 'divider', mb: 3 }}>
        <Tabs value={tab} onChange={handleTabChange}>
          <Tab label="Side Bias" />
          <Tab label="Synergy" />
          <Tab label="Counter" />
        </Tabs>
      </Box>

      <TabPanel value={tab} index={0}>
        {isLoadingSideBias ? (
          <Typography>Loading side bias data...</Typography>
        ) : sideBiasData ? (
          <SideBiasChart data={sideBiasData} minGames={minGames} />
        ) : null}
      </TabPanel>

      <TabPanel value={tab} index={1}>
        {isLoadingSynergy ? (
          <Typography>Loading synergy data...</Typography>
        ) : synergyData ? (
          <Heatmap 
            data={synergyData.matrix}
            xLabels={synergyData.heroes}
            yLabels={synergyData.heroes}
            title="Hero Synergy Matrix"
          />
        ) : null}
      </TabPanel>

      <TabPanel value={tab} index={2}>
        {isLoadingCounter ? (
          <Typography>Loading counter data...</Typography>
        ) : counterData ? (
          <Heatmap 
            data={counterData.matrix}
            xLabels={counterData.heroes}
            yLabels={counterData.heroes}
            title="Hero Counter Matrix"
            colorScale={[
              [0, '#1976d2'],
              [0.5, '#fff'],
              [1, '#d32f2f']
            ]}
          />
        ) : null}
      </TabPanel>
    </Container>
  )
}

interface TabPanelProps {
  children?: React.ReactNode
  index: number
  value: number
}

function TabPanel(props: TabPanelProps) {
  const { children, value, index, ...other } = props

  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`analytics-tabpanel-${index}`}
      aria-labelledby={`analytics-tab-${index}`}
      {...other}
    >
      {value === index && <Box>{children}</Box>}
    </div>
  )
}