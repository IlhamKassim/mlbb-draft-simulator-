import React, { useState, useMemo, useCallback, useRef } from 'react';
import { styled } from '@mui/material/styles';
import {
  TextField,
  List,
  ListItem,
  ListItemAvatar,
  ListItemText,
  Avatar,
  Chip,
  Paper,
  Box,
  Popper,
  ClickAwayListener
} from '@mui/material';
import { Hero, Role, HeroSearchProps } from '../types/hero';
import { debounce } from 'lodash';

const ROLES: Role[] = ['Tank', 'Fighter', 'Assassin', 'Marksman', 'Mage', 'Support'];

const RoleChips = styled('div')(({ theme }) => ({
  display: 'flex',
  flexWrap: 'wrap',
  gap: theme.spacing(1),
  marginTop: theme.spacing(1),
  marginBottom: theme.spacing(1)
}));

const StyledPopper = styled(Popper)(({ theme }) => ({
  width: '100%',
  zIndex: theme.zIndex.modal,
  marginTop: theme.spacing(1)
}));

const HeroList = styled(List)(({ theme }) => ({
  maxHeight: '300px',
  overflow: 'auto',
  backgroundColor: theme.palette.background.paper,
  borderRadius: theme.shape.borderRadius
}));

const HeroSearch: React.FC<HeroSearchProps> = ({
  heroes,
  onSelect,
  disabled = false,
  placeholder = "Search heroes..."
}) => {
  const [query, setQuery] = useState('');
  const [roles, setRoles] = useState<Set<Role>>(new Set());
  const [open, setOpen] = useState(false);
  const [selectedIndex, setSelectedIndex] = useState(0);
  const anchorRef = useRef<HTMLDivElement>(null);

  const debouncedSetQuery = useCallback(
    debounce((value: string) => setQuery(value), 150),
    []
  );

  const filteredHeroes = useMemo(() => {
    return heroes
      .filter(h => h.name.toLowerCase().includes(query.toLowerCase()))
      .filter(h => roles.size === 0 || roles.has(h.role));
  }, [heroes, query, roles]);

  const handleInputChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    debouncedSetQuery(event.target.value);
    setOpen(true);
    setSelectedIndex(0);
  };

  const handleRoleToggle = (role: Role) => {
    const newRoles = new Set(roles);
    if (newRoles.has(role)) {
      newRoles.delete(role);
    } else {
      newRoles.add(role);
    }
    setRoles(newRoles);
    setSelectedIndex(0);
  };

  const handleKeyDown = (event: React.KeyboardEvent) => {
    if (!open) return;

    switch (event.key) {
      case 'ArrowDown':
        event.preventDefault();
        setSelectedIndex(prev => 
          prev < filteredHeroes.length - 1 ? prev + 1 : prev
        );
        break;
      case 'ArrowUp':
        event.preventDefault();
        setSelectedIndex(prev => prev > 0 ? prev - 1 : 0);
        break;
      case 'Enter':
        event.preventDefault();
        if (filteredHeroes[selectedIndex]) {
          handleSelect(filteredHeroes[selectedIndex]);
        }
        break;
      case 'Escape':
        setOpen(false);
        break;
    }
  };

  const handleSelect = (hero: Hero) => {
    onSelect(hero);
    setOpen(false);
    setQuery('');
    debouncedSetQuery('');
  };

  const handleClickAway = () => {
    setOpen(false);
  };

  return (
    <ClickAwayListener onClickAway={handleClickAway}>
      <Box ref={anchorRef}>
        <TextField
          fullWidth
          placeholder={placeholder}
          onChange={handleInputChange}
          onFocus={() => setOpen(true)}
          onKeyDown={handleKeyDown}
          disabled={disabled}
          size="small"
        />
        
        <RoleChips>
          {ROLES.map(role => (
            <Chip
              key={role}
              label={role}
              onClick={() => handleRoleToggle(role)}
              color={roles.has(role) ? "primary" : "default"}
              variant={roles.has(role) ? "filled" : "outlined"}
            />
          ))}
        </RoleChips>

        <StyledPopper
          open={open && filteredHeroes.length > 0}
          anchorEl={anchorRef.current}
          placement="bottom-start"
        >
          <Paper elevation={3}>
            <HeroList>
              {filteredHeroes.map((hero, index) => (
                <ListItem
                  button
                  key={hero.id}
                  onClick={() => handleSelect(hero)}
                  selected={index === selectedIndex}
                >
                  <ListItemAvatar>
                    <Avatar src={hero.iconUrl} alt={hero.name} />
                  </ListItemAvatar>
                  <ListItemText
                    primary={hero.name}
                    secondary={hero.role}
                  />
                </ListItem>
              ))}
            </HeroList>
          </Paper>
        </StyledPopper>
      </Box>
    </ClickAwayListener>
  );
};

export default HeroSearch;