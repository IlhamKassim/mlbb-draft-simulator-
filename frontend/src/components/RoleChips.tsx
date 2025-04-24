import React from 'react';
import { Chip, Stack } from '@mui/material';
import { Role } from '../types/hero';

interface RoleChipsProps {
    selected: Role[];
    onToggle: (role: Role) => void;
    disabled?: boolean;
}

const ROLES: Role[] = ['Tank', 'Fighter', 'Assassin', 'Marksman', 'Mage', 'Support'];

export const RoleChips: React.FC<RoleChipsProps> = ({ selected, onToggle, disabled }) => {
    return (
        <Stack direction="row" spacing={1} sx={{ my: 1, flexWrap: 'wrap', gap: 1 }}>
            {ROLES.map(role => (
                <Chip
                    key={role}
                    label={role}
                    onClick={() => onToggle(role)}
                    color={selected.includes(role) ? "primary" : "default"}
                    variant={selected.includes(role) ? "filled" : "outlined"}
                    disabled={disabled}
                    sx={{
                        '&:hover': {
                            bgcolor: selected.includes(role) ? 'primary.dark' : 'action.hover',
                        },
                    }}
                />
            ))}
        </Stack>
    );
};