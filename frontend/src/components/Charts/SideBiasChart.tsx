import { useEffect, useRef } from 'react'
import Plot from 'plotly.js-dist'

interface SideBiasData {
  hero: string
  effect_size: number
  confidence_interval: [number, number]
  games_played: number
}

interface SideBiasChartProps {
  data: SideBiasData[]
  minGames?: number
}

export default function SideBiasChart({ data, minGames = 10 }: SideBiasChartProps) {
  const chartRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!chartRef.current || !data.length) return

    // Filter and sort data by effect size
    const filteredData = data
      .filter(d => d.games_played >= minGames)
      .sort((a, b) => b.effect_size - a.effect_size)

    const trace = {
      type: 'scatter',
      x: filteredData.map(d => d.effect_size),
      y: filteredData.map(d => d.hero),
      error_x: {
        type: 'data',
        array: filteredData.map(d => d.confidence_interval[1] - d.effect_size),
        arrayminus: filteredData.map(d => d.effect_size - d.confidence_interval[0]),
        color: '#444'
      },
      mode: 'markers',
      marker: {
        color: filteredData.map(d => 
          d.effect_size > 0 ? '#1976d2' : '#d32f2f'
        ),
        size: 8
      },
      hovertemplate: 
        '%{y}<br>' +
        'Effect Size: %{x:.3f}<br>' +
        'Games: %{customdata}<br>' +
        '<extra></extra>',
      customdata: filteredData.map(d => d.games_played)
    }

    const layout = {
      title: 'Hero Side Bias (Cohen\'s h)',
      xaxis: {
        title: 'Effect Size (Blue Side Advantage →)',
        zeroline: true,
        zerolinewidth: 2,
        zerolinecolor: '#444'
      },
      yaxis: {
        title: 'Hero',
        automargin: true
      },
      showlegend: false,
      height: Math.max(400, filteredData.length * 20),
      margin: { l: 120 }
    }

    Plot.newPlot(chartRef.current, [trace], layout)
  }, [data, minGames])

  return <div ref={chartRef} />
}