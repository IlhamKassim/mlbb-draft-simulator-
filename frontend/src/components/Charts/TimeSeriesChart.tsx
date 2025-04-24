import { useEffect, useRef } from 'react'
import Plot from 'plotly.js-dist'

interface TimeSeriesData {
  patches: string[]
  roles: string[]
  data: Record<string, number[][]>
}

interface TimeSeriesChartProps {
  data: TimeSeriesData
  title: string
  metric: 'pick_rate' | 'win_rate' | 'ban_rate'
}

export default function TimeSeriesChart({ data, title, metric }: TimeSeriesChartProps) {
  const chartRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!chartRef.current || !data) return

    const traces = data.roles.map((role) => ({
      type: 'scatter',
      mode: 'lines+markers',
      name: role,
      x: data.patches,
      y: data.data[metric].map(row => row[data.roles.indexOf(role)]),
      line: {
        width: 2
      },
      marker: {
        size: 6
      }
    }))

    const layout = {
      title,
      xaxis: {
        title: 'Patch Version',
        tickangle: -45
      },
      yaxis: {
        title: metric.split('_').map(word => 
          word.charAt(0).toUpperCase() + word.slice(1)
        ).join(' '),
        tickformat: ',.1%'
      },
      height: 500,
      margin: { b: 100 },
      showlegend: true,
      legend: {
        orientation: 'h',
        yanchor: 'bottom',
        y: -0.3,
        xanchor: 'center',
        x: 0.5
      }
    }

    Plot.newPlot(chartRef.current, traces, layout)

    return () => {
      if (chartRef.current) {
        Plot.purge(chartRef.current)
      }
    }
  }, [data, title, metric])

  return <div ref={chartRef} />
}