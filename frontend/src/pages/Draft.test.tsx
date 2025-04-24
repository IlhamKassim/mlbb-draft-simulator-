import React from 'react';
import { render, screen, waitFor, fireEvent } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { MemoryRouter } from 'react-router-dom';
import { ThemeProvider } from '../contexts/ThemeContext';
import Draft from './Draft';
import * as api from '../api/api';

// Mock the API hooks
jest.mock('../api/api', () => ({
  useHeroes: jest.fn(),
  useDraftPrediction: jest.fn(),
}));

// Create test data
const mockHeroData = [
  { 
    id: 'layla', 
    name: 'Layla', 
    roles: ['Marksman'],
    specialty: 'Damage',
    difficulty: 'Easy',
    description: 'Layla is a marksman hero with high damage output.',
    imageUrl: '/static/hero-icons/layla.png',
    winRate: 0.52,
    pickRate: 0.15,
    banRate: 0.02
  },
  { 
    id: 'tigreal', 
    name: 'Tigreal', 
    roles: ['Tank'],
    specialty: 'Control',
    difficulty: 'Easy',
    description: 'Tigreal is a tank hero with crowd control abilities.',
    imageUrl: '/static/hero-icons/tigreal.png',
    winRate: 0.51,
    pickRate: 0.12,
    banRate: 0.01
  },
  { 
    id: 'fanny', 
    name: 'Fanny', 
    roles: ['Assassin'],
    specialty: 'Charge',
    difficulty: 'Hard',
    description: 'Fanny is an assassin who uses cables to move quickly.',
    imageUrl: '/static/hero-icons/fanny.png',
    winRate: 0.48,
    pickRate: 0.08,
    banRate: 0.05
  }
];

const mockPrediction = {
  mutate: jest.fn(),
  isLoading: false,
  data: {
    blue_win_probability: 0.6,
    features_used: { 'feature1': 0.5, 'feature2': 0.3 },
    recommendations: [
      { hero: 'Chou', win_probability: 0.65, description: 'Pick Chou (65% win probability)' },
      { hero: 'Gusion', win_probability: 0.62, description: 'Pick Gusion (62% win probability)' }
    ],
    current_phase: 'PICK PHASE',
    blue_turn: true
  }
};

// Setup test component with providers
const renderDraft = () => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: {
        retry: false,
      },
    },
  });
  
  return render(
    <QueryClientProvider client={queryClient}>
      <ThemeProvider>
        <MemoryRouter>
          <Draft />
        </MemoryRouter>
      </ThemeProvider>
    </QueryClientProvider>
  );
};

describe('Draft Page', () => {
  beforeEach(() => {
    // Mock API responses
    (api.useHeroes as jest.Mock).mockReturnValue({
      data: mockHeroData,
      isLoading: false
    });
    
    (api.useDraftPrediction as jest.Mock).mockReturnValue(mockPrediction);
  });
  
  afterEach(() => {
    jest.clearAllMocks();
  });

  test('renders the draft page with title', () => {
    renderDraft();
    
    // Check if the title is displayed
    expect(screen.getByText('Draft Simulator')).toBeInTheDocument();
    
    // Check for phase indicator
    expect(screen.getByText(/BAN/i)).toBeInTheDocument();
  });
  
  test('displays hero grid with all heroes from API', async () => {
    renderDraft();
    
    // Wait for heroes to load
    await waitFor(() => {
      // Check if each mocked hero is displayed
      mockHeroData.forEach(hero => {
        expect(screen.getByText(hero.name)).toBeInTheDocument();
      });
    });
  });
  
  test('shows win probability gauge', async () => {
    renderDraft();
    
    // Check if win probability section is displayed
    expect(screen.getByText('Win Probability')).toBeInTheDocument();
    
    // Check if recommendations are displayed
    await waitFor(() => {
      mockPrediction.data.recommendations.forEach(rec => {
        expect(screen.getByText(rec.hero)).toBeInTheDocument();
      });
    });
  });
  
  test('selects a hero when clicked', async () => {
    renderDraft();
    
    // Click on a hero
    await waitFor(() => {
      fireEvent.click(screen.getByText('Layla'));
    });
    
    // Verify the API was called with the selected hero
    await waitFor(() => {
      expect(mockPrediction.mutate).toHaveBeenCalledWith(
        expect.objectContaining({
          blue_bans: ['Layla'],
          blue_picks: [],
          red_bans: [],
          red_picks: []
        })
      );
    });
  });
  
  test('resets the draft state when reset button clicked', async () => {
    renderDraft();
    
    // First select a hero
    await waitFor(() => {
      fireEvent.click(screen.getByText('Layla'));
    });
    
    // Now click reset
    const resetButton = screen.getByText('Reset');
    fireEvent.click(resetButton);
    
    // Verify the state was reset
    await waitFor(() => {
      expect(mockPrediction.mutate).toHaveBeenCalledWith(
        expect.objectContaining({
          blue_bans: [],
          blue_picks: [],
          red_bans: [],
          red_picks: []
        })
      );
    });
  });
  
  test('updates win probability when refresh button clicked', async () => {
    renderDraft();
    
    // First select a hero
    await waitFor(() => {
      fireEvent.click(screen.getByText('Layla'));
    });
    
    // Clear mock to track new calls
    mockPrediction.mutate.mockClear();
    
    // Click refresh
    const refreshButton = screen.getByText('Refresh');
    fireEvent.click(refreshButton);
    
    // Verify API was called again
    await waitFor(() => {
      expect(mockPrediction.mutate).toHaveBeenCalled();
    });
  });
  
  test('displays loading state while fetching hero data', async () => {
    // Mock loading state
    (api.useHeroes as jest.Mock).mockReturnValueOnce({
      data: null,
      isLoading: true
    });
    
    renderDraft();
    
    // We should see loading indicators instead of hero cards
    expect(screen.queryByText('Layla')).not.toBeInTheDocument();
    
    // Update mock to loaded state
    (api.useHeroes as jest.Mock).mockReturnValueOnce({
      data: mockHeroData,
      isLoading: false
    });
    
    // Re-render
    renderDraft();
    
    // Now we should see heroes
    await waitFor(() => {
      expect(screen.getByText('Layla')).toBeInTheDocument();
    });
  });
});