import { useState } from 'react';
import { 
  Card, 
  CardContent, 
  CardMedia, 
  Typography, 
  Box,
  Skeleton,
  Chip,
  Tooltip,
  styled,
  keyframes,
  Badge,
  Avatar
} from '@mui/material';
import { tokens, mlbbColors } from '../theme';

// Role icon imports
import ShieldIcon from '@mui/icons-material/Shield';
import FitnessCenterIcon from '@mui/icons-material/FitnessCenter';
import FlashOnIcon from '@mui/icons-material/FlashOn';
import WhatShotIcon from '@mui/icons-material/Whatshot';
import GpsFixedIcon from '@mui/icons-material/GpsFixed';
import LocalHospitalIcon from '@mui/icons-material/LocalHospital';

interface HeroCardProps {
  name: string;
  imageUrl?: string;
  role?: string;
  winRate?: number;
  pickRate?: number;
  banRate?: number;
  isSelected?: boolean;
  isDisabled?: boolean;
  team?: 'blue' | 'red';
  type?: 'pick' | 'ban';
  onClick?: () => void;
  isLoading?: boolean;
}

const glowAnimation = (color: string) => keyframes`
  0% {
    box-shadow: 0 0 5px ${color};
  }
  50% {
    box-shadow: 0 0 20px ${color};
  }
  100% {
    box-shadow: 0 0 5px ${color};
  }
`;

const StyledCard = styled(Card, {
  shouldForwardProp: (prop) => 
    !['isSelected', 'isDisabled', 'team', 'type'].includes(prop as string)
})<{
  isSelected?: boolean;
  isDisabled?: boolean;
  team?: 'blue' | 'red';
  type?: 'pick' | 'ban';
}>(({ theme, isSelected, isDisabled, team, type }) => ({
  height: '100%',
  cursor: isDisabled ? 'not-allowed' : 'pointer',
  opacity: isDisabled ? 0.5 : 1,
  transition: `all ${tokens.animation.medium} cubic-bezier(0.4, 0, 0.2, 1)`,
  transform: isSelected ? 'scale(1.05)' : 'scale(1)',
  position: 'relative',
  
  '&:hover': !isDisabled && {
    transform: 'scale(1.05)',
    boxShadow: theme.shadows[8]
  },

  ...(isSelected && team && {
    animation: `${glowAnimation(team === 'blue' ? mlbbColors.blue.main : mlbbColors.red.main)} 2s infinite`
  }),

  ...(type === 'ban' && {
    '&::after': {
      content: '""',
      position: 'absolute',
      top: 0,
      left: 0,
      right: 0,
      bottom: 0,
      background: `repeating-linear-gradient(
        45deg,
        ${theme.palette.error.main}20,
        ${theme.palette.error.main}20 10px,
        ${theme.palette.error.main}10 10px,
        ${theme.palette.error.main}10 20px
      )`,
      pointerEvents: 'none'
    }
  })
}));

const StyledRoleIcon = styled(Avatar, {
  shouldForwardProp: (prop) => prop !== 'roleColor'
})<{ roleColor: string }>(({ theme, roleColor }) => ({
  width: 30,
  height: 30,
  backgroundColor: `${roleColor}CC`,
  position: 'absolute',
  top: 5,
  right: 5,
  border: `2px solid ${theme.palette.background.paper}`,
  '& .MuiSvgIcon-root': {
    fontSize: '1rem'
  },
  zIndex: 2
}));

const getHeroImageUrl = (heroName: string): string => {
  // Helper function to standardize hero names for image URLs
  // 1. Convert to lowercase
  // 2. Handle spaces by replacing with dashes or underscores
  const standardizedName = heroName.toLowerCase().replace(/\s+/g, '-');
  return `${import.meta.env.VITE_API_BASE_URL}/static/hero-icons/${standardizedName}.png`;
};

// Role-specific colors and icons
const getRoleInfo = (role: string | undefined) => {
  switch (role?.toLowerCase()) {
    case 'tank':
      return { color: '#3F51B5', icon: <ShieldIcon /> };
    case 'fighter':
      return { color: '#F44336', icon: <FitnessCenterIcon /> };
    case 'assassin':
      return { color: '#9C27B0', icon: <FlashOnIcon /> };
    case 'mage':
      return { color: '#00BCD4', icon: <WhatShotIcon /> };
    case 'marksman':
      return { color: '#FF9800', icon: <GpsFixedIcon /> };
    case 'support':
      return { color: '#4CAF50', icon: <LocalHospitalIcon /> };
    default:
      return { color: '#757575', icon: <FitnessCenterIcon /> };
  }
};

export default function HeroCard({
  name,
  imageUrl,
  role,
  winRate,
  pickRate,
  banRate,
  isSelected = false,
  isDisabled = false,
  team,
  type,
  onClick,
  isLoading = false
}: HeroCardProps) {
  const [isHovered, setIsHovered] = useState(false);
  const [imgError, setImgError] = useState(false);
  const roleInfo = getRoleInfo(role);

  // Debug logs for HeroCard
  console.log('HeroCard - Rendering hero:', name);
  console.log('HeroCard - Props received:', {
    name, imageUrl, role, winRate, pickRate, banRate, isSelected, isDisabled, team, type
  });

  const tooltipContent = (
    <Box>
      <Typography variant="subtitle2" gutterBottom>
        {name} - {role}
      </Typography>
      {winRate !== undefined && (
        <Typography variant="body2" color="text.secondary">
          Win Rate: {(winRate * 100).toFixed(1)}%
        </Typography>
      )}
      {pickRate !== undefined && (
        <Typography variant="body2" color="text.secondary">
          Pick Rate: {(pickRate * 100).toFixed(1)}%
        </Typography>
      )}
      {banRate !== undefined && (
        <Typography variant="body2" color="text.secondary">
          Ban Rate: {(banRate * 100).toFixed(1)}%
        </Typography>
      )}
    </Box>
  );

  if (isLoading) {
    return (
      <Card>
        <Skeleton variant="rectangular" height={140} />
        <CardContent>
          <Skeleton variant="text" width="80%" />
          <Skeleton variant="text" width="60%" />
        </CardContent>
      </Card>
    );
  }

  const getCardImage = () => {
    // If custom imageUrl is provided, use that
    if (imageUrl) {
      return imageUrl;
    }

    // Otherwise generate URL based on hero name
    const generatedUrl = getHeroImageUrl(name);
    
    // If we've already tried and got an error, switch to SVG
    if (imgError) {
      return generatedUrl.replace('.png', '.svg');
    }
    
    return generatedUrl;
  };

  return (
    <Tooltip title={tooltipContent} arrow placement="top">
      <StyledCard
        isSelected={isSelected}
        isDisabled={isDisabled}
        team={team}
        type={type}
        onClick={isDisabled ? undefined : onClick}
        onMouseEnter={() => setIsHovered(true)}
        onMouseLeave={() => setIsHovered(false)}
      >
        {role && (
          <StyledRoleIcon roleColor={roleInfo.color}>
            {roleInfo.icon}
          </StyledRoleIcon>
        )}
        
        <CardMedia
          component="img"
          height="140"
          image={getCardImage()}
          alt={name}
          sx={{ objectFit: 'cover', p: 1 }}
          onError={(e) => {
            const target = e.target as HTMLImageElement;
            
            if (!imgError) {
              // First error - try SVG
              setImgError(true);
              target.src = target.src.replace('.png', '.svg');
            } else {
              // Second error - use ultimate fallback
              target.onerror = null; // Prevent infinite loop
              target.src = '/placeholder-hero.png';
              console.warn(`Hero icon missing for ${name}, using placeholder.`);
            }
          }}
        />
        <CardContent sx={{ p: 1, '&:last-child': { pb: 1 } }}>
          <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <Typography 
              variant="subtitle2" 
              noWrap 
              sx={{ 
                maxWidth: 'calc(100% - 26px)',
                fontWeight: 'bold' 
              }}
            >
              {name}
            </Typography>
            
            {winRate !== undefined && (
              <Chip
                label={`${(winRate * 100).toFixed(0)}%`}
                size="small"
                color={winRate > 0.5 ? "success" : winRate < 0.45 ? "error" : "default"}
                sx={{ 
                  height: 20,
                  minWidth: 45,
                  fontSize: '0.7rem',
                  opacity: isHovered ? 1 : 0.7,
                  transition: `opacity ${tokens.animation.fast}`
                }}
              />
            )}
          </Box>
          
          <Box sx={{ display: 'flex', alignItems: 'center', mt: 0.5, gap: 0.5 }}>
            {role && (
              <Typography 
                variant="caption" 
                color="text.secondary"
                sx={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: 0.5
                }}
              >
                {roleInfo.icon && 
                  <Box sx={{ fontSize: '0.8rem', color: roleInfo.color }}>
                    {roleInfo.icon}
                  </Box>
                }
                {role}
              </Typography>
            )}
            
            {pickRate !== undefined && isHovered && (
              <Typography 
                variant="caption" 
                color="text.secondary"
                sx={{ ml: 'auto' }}
              >
                PR: {(pickRate * 100).toFixed(1)}%
              </Typography>
            )}
          </Box>
        </CardContent>
      </StyledCard>
    </Tooltip>
  );
}