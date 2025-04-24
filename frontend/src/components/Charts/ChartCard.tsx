import { Card, CardContent, CardHeader, Skeleton, useTheme } from '@mui/material';
import Plot from 'react-plotly.js';
import { tokens } from '../../theme';

interface ChartCardProps {
  title: string;
  data: any[];
  layout?: Partial<Plotly.Layout>;
  isLoading?: boolean;
  height?: number;
}

export default function ChartCard({
  title,
  data,
  layout = {},
  isLoading = false,
  height = 400
}: ChartCardProps) {
  const theme = useTheme();

  const baseLayout: Partial<Plotly.Layout> = {
    paper_bgcolor: 'transparent',
    plot_bgcolor: 'transparent',
    font: {
      family: theme.typography.fontFamily,
      color: theme.palette.text.primary
    },
    margin: { t: 10, r: 10, l: 50, b: 50 },
    height,
    xaxis: {
      gridcolor: theme.palette.divider,
      zerolinecolor: theme.palette.divider
    },
    yaxis: {
      gridcolor: theme.palette.divider,
      zerolinecolor: theme.palette.divider
    }
  };

  const config: Partial<Plotly.Config> = {
    displayModeBar: false,
    responsive: true
  };

  if (isLoading) {
    return (
      <Card>
        <CardHeader title={title} />
        <CardContent>
          <Skeleton variant="rectangular" height={height} />
        </CardContent>
      </Card>
    );
  }

  return (
    <Card sx={{ 
      transition: `all ${tokens.animation.medium}`,
      '&:hover': {
        transform: 'translateY(-4px)',
        boxShadow: theme.shadows[8]
      }
    }}>
      <CardHeader title={title} />
      <CardContent>
        <Plot
          data={data}
          layout={{ ...baseLayout, ...layout }}
          config={config}
          style={{ width: '100%' }}
        />
      </CardContent>
    </Card>
  );
}