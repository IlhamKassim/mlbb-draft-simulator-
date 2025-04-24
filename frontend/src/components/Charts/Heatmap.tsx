import { useEffect, useRef } from 'react'
import Plot from 'plotly.js-dist'

interface HeatmapProps {
  data: number[][]
  xLabels: string[]
  yLabels: string[]
  title: string
  colorScale?: Array<[number, string]>
}

export default function Heatmap({ 
  data, 
  xLabels, 
  yLabels, 
  title,
  colorScale = [
    [0, '#d32f2f'],
    [0.5, '#fff'],
    [1, '#1976d2']
  ]
}: HeatmapProps) {
  const chartRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!chartRef.current || !data.length) return

    const trace = {
      type: 'heatmap',
      z: data,
      x: xLabels,
      y: yLabels,
      colorscale: colorScale,
      hoverongaps: false,
      hovertemplate: 
        'X: %{x}<br>' +
        'Y: %{y}<br>' +
        'Value: %{z:.3f}<br>' +
        '<extra></extra>'
    }

    const layout = {
      title,
      xaxis: {
        side: 'bottom',
        tickangle: 45
      },
      yaxis: {
        automargin: true
      },
      height: Math.max(500, yLabels.length * 20),
      margin: { l: 120, b: 120 }
    }

    Plot.newPlot(chartRef.current, [trace], layout)

    return () => {
      if (chartRef.current) {
        Plot.purge(chartRef.current)
      }
    }
  }, [data, xLabels, yLabels, title, colorScale])

  return <div ref={chartRef} />
}