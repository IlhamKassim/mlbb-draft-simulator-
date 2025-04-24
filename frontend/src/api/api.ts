import axios from 'axios';
import { useQuery, useMutation } from '@tanstack/react-query';

// Use the hardcoded API URL from vite config
const API_URL = '__API_URL__' || 'http://localhost:8008';

// Debug log the API URL being used
console.log('API URL being used:', API_URL);

// Create an axios instance with the API URL
const apiClient = axios.create({
  baseURL: API_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Debug interceptor to log all requests
apiClient.interceptors.request.use(request => {
  console.log('API Request:', request.method?.toUpperCase(), request.baseURL + request.url);
  return request;
});

// Debug interceptor to log all responses or errors
apiClient.interceptors.response.use(
  response => {
    console.log('API Response:', response.status, response.config.url);
    return response;
  },
  error => {
    console.error('API Error:', 
      error.response?.status || 'Network Error', 
      error.config?.url,
      error.message
    );
    return Promise.reject(error);
  }
);

export interface DraftState {
  blue_picks: string[];
  red_picks: string[];
  blue_bans: string[];
  red_bans: string[];
  patch_version?: string;
}

export interface HeroRecommendation {
  hero: string;
  win_probability: number;
  description?: string;
}

export interface PredictionResponse {
  blue_win_probability: number;
  features_used: Record<string, number>;
  recommendations: HeroRecommendation[];
  current_phase: string;
  blue_turn: boolean;
}

// API hook for draft prediction
export const useDraftPrediction = () => {
  return useMutation(['draft-prediction'], async (draftState: DraftState) => {
    const { data } = await apiClient.post('/predict', draftState);
    return data;
  });
};

// API hook for side bias analysis
export const useSideBias = (minGames: number = 10, topN?: number) => {
  const params: Record<string, string> = { min_games: minGames.toString() };
  if (topN) params.top_n = topN.toString();
  
  return useQuery(['side-bias', minGames, topN], async () => {
    const { data } = await apiClient.get('/side-bias', { params });
    return data.bias_data;
  });
};

// API hook for hero stats
export const useHeroStats = (patchVersion?: string) => {
  return useQuery(['hero-stats', patchVersion], async () => {
    const { data } = await apiClient.get(`/stats${patchVersion ? `?patch_version=${patchVersion}` : ''}`);
    return data;
  });
};

// API hook for synergy matrix
export const useSynergyMatrix = (minGames: number = 10) => {
  return useQuery(['synergy-matrix', minGames], async () => {
    const { data } = await apiClient.get('/stats/synergy', { 
      params: { min_games: minGames } 
    });
    return data;
  });
};

// API hook for counter matrix
export const useCounterMatrix = (minGames: number = 10) => {
  return useQuery(['counter-matrix', minGames], async () => {
    const { data } = await apiClient.get('/stats/counter', { 
      params: { min_games: minGames } 
    });
    return data;
  });
};

export interface RoleTrendData {
  patches: string[];
  roles: string[];
  data: {
    pick_rate: number[][];
    win_rate: number[][];
    ban_rate: number[][];
  };
}

export interface RoleMatchupData {
  roles: string[];
  matrix: number[][];
}

// API hook for role trends
export const useRoleTrends = () => {
  return useQuery(['role-trends'], async () => {
    const { data } = await apiClient.get('/stats/role-trends');
    return data;
  });
};

// API hook for role matchups
export const useRoleMatchups = () => {
  return useQuery(['role-matchups'], async () => {
    const { data } = await apiClient.get('/stats/role-matchups');
    return data;
  });
};

export interface Hero {
  id: string;
  name: string;
  roles: string[];
  specialty: string;
  difficulty: string;
  description: string;
  counters: string[];
  countered_by: string[];
}

export interface HeroRole {
  description: string;
  characteristics: string[];
  playstyle: string;
}

export interface HeroRolesData {
  roles: Record<string, HeroRole>;
  role_counters: Record<string, string[]>;
}

// API hooks for hero data
export const useHeroes = () => {
  return useQuery(['heroes'], async () => {
    console.log('Fetching heroes from:', `${API_URL}/draft/heroes`);
    const { data } = await apiClient.get('/draft/heroes');
    console.log('Heroes data received:', data ? `${data.length} heroes` : 'empty data');
    return data;
  });
};

export const useHeroDetails = (heroName: string) => {
  return useQuery(['heroDetails', heroName], async () => {
    const { data } = await apiClient.get(`/hero_details/${heroName}`);
    return data;
  }, {
    enabled: !!heroName,
  });
};

export const useHeroRoles = () => {
  return useQuery(['heroRoles'], async () => {
    const { data } = await apiClient.get('/hero_roles');
    return data;
  });
};