"""
Monte Carlo Tree Search implementation for MLBB draft simulation.
"""
import math
import random
from typing import List, Dict, Optional, Tuple, NamedTuple
from dataclasses import dataclass
from collections import defaultdict

import numpy as np

class DraftState(NamedTuple):
    """Current state of the draft."""
    blue_picks: List[str]
    red_picks: List[str]
    blue_bans: List[str]
    red_bans: List[str]
    blue_turn: bool  # True if it's blue team's turn
    is_pick_phase: bool  # True for pick phase, False for ban phase

@dataclass
class MCTSNode:
    """Node in the MCTS tree."""
    state: DraftState
    parent: Optional['MCTSNode'] = None
    children: List['MCTSNode'] = None
    wins: int = 0
    visits: int = 0
    untried_actions: List[str] = None
    
    def __post_init__(self):
        if self.children is None:
            self.children = []
        if self.untried_actions is None:
            self.untried_actions = []
            
    def ucb_score(self, exploration: float = math.sqrt(2)) -> float:
        """
        Calculate UCB1 score for node selection.
        
        Args:
            exploration: Exploration parameter (default: sqrt(2))
            
        Returns:
            UCB1 score
        """
        if self.visits == 0:
            return float('inf')
        
        exploitation = self.wins / self.visits
        exploration_term = exploration * math.sqrt(math.log(self.parent.visits) / self.visits)
        return exploitation + exploration_term
    
    def add_child(self, action: str, state: DraftState) -> 'MCTSNode':
        """
        Add a child node.
        
        Args:
            action: Action taken to reach this state
            state: Resulting state
            
        Returns:
            New child node
        """
        child = MCTSNode(
            state=state,
            parent=self
        )
        self.untried_actions.remove(action)
        self.children.append(child)
        return child
    
    def update(self, result: float):
        """
        Update node statistics.
        
        Args:
            result: Win (1.0) or loss (0.0)
        """
        self.wins += result
        self.visits += 1

class MCTSDraftSimulator:
    """MCTS-based draft simulator."""
    
    def __init__(self, model, data_loader, available_heroes: List[str],
                 iterations: int = 1000, exploration: float = math.sqrt(2)):
        """
        Initialize simulator.
        
        Args:
            model: Trained model for win probability prediction
            data_loader: MLBBDataLoader instance
            available_heroes: List of heroes available for drafting
            iterations: Number of MCTS iterations (default: 1000)
            exploration: UCB exploration parameter (default: sqrt(2))
        """
        self.model = model
        self.data_loader = data_loader
        self.available_heroes = available_heroes
        self.iterations = iterations
        self.exploration = exploration
        
    def get_legal_actions(self, state: DraftState) -> List[str]:
        """
        Get legal actions (picks/bans) for current state.
        
        Args:
            state: Current draft state
            
        Returns:
            List of legal hero picks/bans
        """
        used_heroes = set(state.blue_picks + state.red_picks + 
                         state.blue_bans + state.red_bans)
        return [h for h in self.available_heroes if h not in used_heroes]
    
    def is_terminal(self, state: DraftState) -> bool:
        """
        Check if draft is complete.
        
        Args:
            state: Current draft state
            
        Returns:
            True if draft is complete
        """
        return (len(state.blue_picks) == 5 and len(state.red_picks) == 5 and
                len(state.blue_bans) == 3 and len(state.red_bans) == 3)
    
    def make_move(self, state: DraftState, action: str) -> DraftState:
        """
        Apply action to state.
        
        Args:
            state: Current draft state
            action: Hero to pick/ban
            
        Returns:
            New draft state
        """
        blue_picks = list(state.blue_picks)
        red_picks = list(state.red_picks)
        blue_bans = list(state.blue_bans)
        red_bans = list(state.red_bans)
        
        if state.is_pick_phase:
            if state.blue_turn:
                blue_picks.append(action)
            else:
                red_picks.append(action)
        else:
            if state.blue_turn:
                blue_bans.append(action)
            else:
                red_bans.append(action)
                
        # Determine next turn and phase
        max_picks = 5
        max_bans = 3
        
        # First complete bans, then moves to picks
        if not state.is_pick_phase:
            # If we're still in ban phase
            if len(blue_bans) < max_bans or len(red_bans) < max_bans:
                is_pick_phase = False
                # Alternate turns between blue and red
                blue_turn = not state.blue_turn
            else:
                # Ban phase complete, transition to pick phase
                is_pick_phase = True
                blue_turn = True  # Blue picks first in pick phase
        else:
            # We're in pick phase
            total_picks = len(blue_picks) + len(red_picks)
            if total_picks < 2 * max_picks:
                is_pick_phase = True
                # Use the pick order logic: blue, red, red, blue, blue, red, red, blue, blue, red
                # This can be determined by the total number of picks made so far
                if total_picks < 4:  # First 4 picks alternate: B,R,R,B
                    blue_turn = (total_picks % 2 == 0)
                elif total_picks < 7:  # Next 3 picks: B,R,R
                    blue_turn = (total_picks == 4)
                else:  # Last 3 picks: B,B,R
                    blue_turn = (total_picks < 9)
            else:
                # Draft is complete
                is_pick_phase = True  # Keep as pick phase since bans are done
                blue_turn = True  # Not relevant anymore since draft is over
                
        return DraftState(
            blue_picks=blue_picks,
            red_picks=red_picks,
            blue_bans=blue_bans,
            red_bans=red_bans,
            blue_turn=blue_turn,
            is_pick_phase=is_pick_phase
        )
    
    def select(self, node: MCTSNode) -> MCTSNode:
        """
        Select a leaf node using UCB1.
        
        Args:
            node: Root node
            
        Returns:
            Selected leaf node
        """
        while not self.is_terminal(node.state):
            if node.untried_actions:
                return node
            
            node = max(node.children, key=lambda n: n.ucb_score(self.exploration))
            
        return node
    
    def expand(self, node: MCTSNode) -> Optional[MCTSNode]:
        """
        Expand tree by trying an untried action.
        
        Args:
            node: Node to expand
            
        Returns:
            New child node or None if no untried actions
        """
        if not node.untried_actions:
            return None
            
        action = random.choice(node.untried_actions)
        new_state = self.make_move(node.state, action)
        return node.add_child(action, new_state)
    
    def simulate(self, state: DraftState) -> float:
        """
        Run simulation from state to end.
        
        Args:
            state: Starting state
            
        Returns:
            1.0 for blue team win, 0.0 for red team win
        """
        while not self.is_terminal(state):
            actions = self.get_legal_actions(state)
            if not actions:
                break
            action = random.choice(actions)
            state = self.make_move(state, action)
            
        # Use model to predict outcome
        features = self.data_loader.prepare_model_features(
            state.blue_picks, state.red_picks,
            state.blue_bans, state.red_bans
        )
        win_prob = self.model.predict_proba([features])[0][1]
        return float(random.random() < win_prob)
    
    def backpropagate(self, node: MCTSNode, result: float):
        """
        Backpropagate simulation result.
        
        Args:
            node: Leaf node
            result: Simulation result
        """
        while node:
            node.update(result)
            node = node.parent
            
    def get_best_action(self, state: DraftState) -> Tuple[str, float]:
        """
        Run MCTS to find best action.
        
        Args:
            state: Current draft state
            
        Returns:
            Tuple of (best action, win probability)
        """
        root = MCTSNode(state=state)
        root.untried_actions = self.get_legal_actions(state)
        
        for _ in range(self.iterations):
            node = self.select(root)
            
            # Expand if not terminal
            if not self.is_terminal(node.state):
                child = self.expand(node)
                if child:
                    node = child
                    
            # Simulate from new state
            result = self.simulate(node.state)
            
            # Backpropagate result
            self.backpropagate(node, result)
            
        # Select action with highest win rate
        if not root.children:
            return random.choice(root.untried_actions), 0.5
            
        best_child = max(root.children, 
                        key=lambda n: (n.wins / n.visits if n.visits > 0 else 0))
        
        # Find action that led to best child
        for action in self.get_legal_actions(state):
            new_state = self.make_move(state, action)
            # Compare all fields in the state tuple instead of direct comparison
            child_state = best_child.state
            if (new_state.blue_picks == child_state.blue_picks and
                new_state.red_picks == child_state.red_picks and
                new_state.blue_bans == child_state.blue_bans and
                new_state.red_bans == child_state.red_bans and
                new_state.blue_turn == child_state.blue_turn and
                new_state.is_pick_phase == child_state.is_pick_phase):
                return action, best_child.wins / best_child.visits
                
        return random.choice(root.untried_actions), 0.5  # Fallback
    
    def get_action_rankings(self, state: DraftState, 
                          top_k: Optional[int] = None) -> List[Tuple[str, float]]:
        """
        Get ranked list of actions with win probabilities.
        
        Args:
            state: Current draft state
            top_k: Optional limit on number of actions to return
            
        Returns:
            List of (action, win probability) tuples, sorted by win probability
        """
        root = MCTSNode(state=state)
        root.untried_actions = self.get_legal_actions(state)
        
        # Run MCTS iterations
        for _ in range(self.iterations):
            node = self.select(root)
            if not self.is_terminal(node.state):
                child = self.expand(node)
                if child:
                    node = child
            result = self.simulate(node.state)
            self.backpropagate(node, result)
            
        # Rank all children by win rate
        action_rates = []
        for action in self.get_legal_actions(state):
            new_state = self.make_move(state, action)
            found_match = False
            
            for child in root.children:
                child_state = child.state
                # Compare all fields in state tuple
                if (new_state.blue_picks == child_state.blue_picks and
                    new_state.red_picks == child_state.red_picks and
                    new_state.blue_bans == child_state.blue_bans and
                    new_state.red_bans == child_state.red_bans and
                    new_state.blue_turn == child_state.blue_turn and
                    new_state.is_pick_phase == child_state.is_pick_phase):
                    win_rate = child.wins / child.visits if child.visits > 0 else 0
                    action_rates.append((action, win_rate))
                    found_match = True
                    break
            
            if not found_match:
                # Action wasn't explored, use rollout
                result = self.simulate(new_state)
                action_rates.append((action, result))
                
        # Sort by win rate and optionally limit
        action_rates.sort(key=lambda x: x[1], reverse=True)
        if top_k:
            action_rates = action_rates[:top_k]
            
        return action_rates
    
    def get_best_action_node(self, state: DraftState, iterations: int = None) -> 'MCTSNode':
        """
        Run MCTS and return the root node for visualization or analysis.
        
        Args:
            state: Current draft state
            iterations: Optional override for number of iterations
            
        Returns:
            Root MCTSNode after running MCTS
        """
        iterations = iterations or self.iterations
        root = MCTSNode(state=state)
        root.untried_actions = self.get_legal_actions(state)
        
        for _ in range(iterations):
            node = self.select(root)
            
            # Expand if not terminal
            if not self.is_terminal(node.state):
                child = self.expand(node)
                if child:
                    node = child
                    
            # Simulate from new state
            result = self.simulate(node.state)
            
            # Backpropagate result
            self.backpropagate(node, result)
            
        return root