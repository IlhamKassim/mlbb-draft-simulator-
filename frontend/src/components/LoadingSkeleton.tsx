import { Box, Skeleton, Grid, Paper } from '@mui/material';

interface LoadingSkeletonProps {
  type: 'hero-card' | 'draft-board' | 'recommendations';
  count?: number;
}

export default function LoadingSkeleton({ type, count = 1 }: LoadingSkeletonProps) {
  const renderHeroCard = () => (
    <Box>
      <Skeleton variant="rectangular" height={140} />
      <Box sx={{ pt: 1 }}>
        <Skeleton width="80%" />
        <Skeleton width="60%" />
      </Box>
    </Box>
  );

  const renderDraftBoard = () => (
    <Box sx={{ width: '100%' }}>
      <Paper sx={{ p: 2, mb: 2 }}>
        <Skeleton width={120} height={32} />
        <Box sx={{ mt: 2 }}>
          <Skeleton width={80} />
          <Grid container spacing={1} sx={{ mt: 1 }}>
            {Array(5).fill(null).map((_, i) => (
              <Grid item xs={12/5} key={i}>
                {renderHeroCard()}
              </Grid>
            ))}
          </Grid>
        </Box>
        <Box sx={{ mt: 2 }}>
          <Skeleton width={80} />
          <Grid container spacing={1} sx={{ mt: 1 }}>
            {Array(3).fill(null).map((_, i) => (
              <Grid item xs={4} key={i}>
                {renderHeroCard()}
              </Grid>
            ))}
          </Grid>
        </Box>
      </Paper>
    </Box>
  );

  const renderRecommendations = () => (
    <Box sx={{ width: '100%' }}>
      <Skeleton width="60%" height={24} sx={{ mb: 2 }} />
      {Array(count).fill(null).map((_, i) => (
        <Box key={i} sx={{ mb: 1 }}>
          <Skeleton width="100%" height={48} />
        </Box>
      ))}
    </Box>
  );

  switch (type) {
    case 'hero-card':
      return (
        <Grid container spacing={2}>
          {Array(count).fill(null).map((_, i) => (
            <Grid item xs={6} sm={4} md={3} lg={2} key={i}>
              {renderHeroCard()}
            </Grid>
          ))}
        </Grid>
      );
    case 'draft-board':
      return renderDraftBoard();
    case 'recommendations':
      return renderRecommendations();
    default:
      return null;
  }
}