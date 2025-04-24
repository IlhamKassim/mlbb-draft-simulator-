import { useEffect, useRef, useState } from 'react';
import { Box, Typography, useTheme, keyframes, CircularProgress, Tooltip } from '@mui/material';
import InfoOutlinedIcon from '@mui/icons-material/InfoOutlined';
import { tokens, mlbbColors } from '../theme';

const fadeIn = keyframes`
  from {
    opacity: 0;
    transform: scale(0.9);
  }
  to {
    opacity: 1;
    transform: scale(1);
  }
`;

const numberChange = keyframes`
  from {
    transform: translateY(100%);
    opacity: 0;
  }
  to {
    transform: translateY(0);
    opacity: 1;
  }
`;

interface WinProbabilityGaugeProps {
  blueWinProbability: number;
  isLoading?: boolean;
  size?: number;
  triggerAnimation?: boolean; // Added to force animation when values change
}

export default function WinProbabilityGauge({
  blueWinProbability,
  isLoading = false,
  size = 200,
  triggerAnimation = false
}: WinProbabilityGaugeProps) {
  const theme = useTheme();
  const prevValueRef = useRef(blueWinProbability);
  const [animatedValue, setAnimatedValue] = useState(blueWinProbability);
  const [isAnimating, setIsAnimating] = useState(false);

  // Smooth transition effect when win probability changes
  useEffect(() => {
    if (Math.abs(blueWinProbability - prevValueRef.current) > 0.01) {
      setIsAnimating(true);
      
      // Animate the value change
      const startValue = prevValueRef.current;
      const endValue = blueWinProbability;
      const duration = 1000; // 1 second animation
      const startTime = Date.now();
      
      const animateValue = () => {
        const elapsed = Date.now() - startTime;
        const progress = Math.min(elapsed / duration, 1);
        
        // Easing function for smooth transition
        const easeOutCubic = (x: number) => 1 - Math.pow(1 - x, 3);
        const easedProgress = easeOutCubic(progress);
        
        // Interpolate between start and end values
        const currentValue = startValue + (endValue - startValue) * easedProgress;
        setAnimatedValue(currentValue);
        
        if (progress < 1) {
          requestAnimationFrame(animateValue);
        } else {
          setIsAnimating(false);
          setAnimatedValue(endValue);
          prevValueRef.current = endValue;
        }
      };
      
      requestAnimationFrame(animateValue);
    } else {
      prevValueRef.current = blueWinProbability;
      setAnimatedValue(blueWinProbability);
    }
  }, [blueWinProbability, triggerAnimation]);

  if (isLoading) {
    return (
      <Box
        sx={{
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          gap: 2
        }}
      >
        <CircularProgress size={size * 0.8} />
        <Typography variant="caption" color="text.secondary">
          Calculating win probability...
        </Typography>
      </Box>
    );
  }

  const bluePercentage = (animatedValue * 100).toFixed(1);
  const redPercentage = ((1 - animatedValue) * 100).toFixed(1);
  const isIncreasing = animatedValue > (prevValueRef.current - 0.01); // Small threshold to handle rounding errors
  const favoredTeam = animatedValue > 0.5 ? 'BLUE' : 'RED';
  const favoredTeamEmoji = animatedValue > 0.5 ? '🔵' : '🔴';

  return (
    <Box
      sx={{
        position: 'relative',
        width: size,
        height: size,
        animation: `${fadeIn} ${tokens.animation.medium} ease-out`
      }}
    >
      {/* Background Circle */}
      <CircularProgress
        variant="determinate"
        value={100}
        size={size}
        sx={{
          position: 'absolute',
          color: theme.palette.mode === 'dark' 
            ? 'rgba(255, 255, 255, 0.1)' 
            : 'rgba(0, 0, 0, 0.1)'
        }}
      />

      {/* Red Team Progress */}
      <CircularProgress
        variant="determinate"
        value={100 - (animatedValue * 100)}
        size={size}
        sx={{
          position: 'absolute',
          color: mlbbColors.red.main,
          transform: 'rotate(180deg)',
          transition: `all ${tokens.animation.slow} ease-in-out`
        }}
      />

      {/* Blue Team Progress */}
      <CircularProgress
        variant="determinate"
        value={animatedValue * 100}
        size={size}
        sx={{
          position: 'absolute',
          color: mlbbColors.blue.main,
          transition: `all ${tokens.animation.slow} ease-in-out`
        }}
      />

      {/* Center Content */}
      <Box
        sx={{
          position: 'absolute',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center'
        }}
      >
        <Typography
          variant="h4"
          sx={{
            animation: isAnimating ? `${numberChange} ${tokens.animation.medium} ease-out` : 'none',
            color: animatedValue > 0.5 
              ? mlbbColors.blue.main 
              : mlbbColors.red.main
          }}
        >
          {animatedValue > 0.5 ? bluePercentage : redPercentage}%
        </Typography>
        <Typography
          variant="caption"
          color="text.secondary"
          sx={{ 
            animation: isAnimating ? `${fadeIn} ${tokens.animation.medium} ease-out` : 'none',
            mt: 0.5,
            display: 'flex',
            alignItems: 'center',
            gap: 0.5
          }}
        >
          {favoredTeam} FAVORED
          {isIncreasing ? ' ↑' : ' ↓'}
          <Tooltip 
            title={
              <Box>
                <Typography variant="body2">
                  {favoredTeamEmoji} <strong>{favoredTeam} team has a {animatedValue > 0.5 ? bluePercentage : redPercentage}% chance to win</strong> based on:
                </Typography>
                <ul style={{ margin: '8px 0', paddingLeft: '20px' }}>
                  <li>Current heroes picked and banned</li>
                  <li>Team composition synergies</li>
                  <li>Counter relationships</li>
                  <li>Historical win rates</li>
                </ul>
                <Typography variant="body2">
                  Values closer to 50% indicate a more balanced draft.
                </Typography>
              </Box>
            }
            arrow
          >
            <InfoOutlinedIcon fontSize="small" sx={{ opacity: 0.7, cursor: 'help' }} />
          </Tooltip>
        </Typography>
      </Box>
    </Box>
  );
}