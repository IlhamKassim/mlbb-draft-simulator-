export type Role = 'Tank' | 'Fighter' | 'Assassin' | 'Marksman' | 'Mage' | 'Support';

export interface Hero {
    id: string;
    name: string;
    role: Role;
    iconUrl: string;
}

export interface HeroSearchProps {
    heroes: Hero[];
    onSelect: (hero: Hero) => void;
    disabled?: boolean;
    placeholder?: string;
}