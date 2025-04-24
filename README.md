# MLBB Draft Analytics & Simulation Engine

A comprehensive system for Mobile Legends: Bang Bang (MLBB) draft analysis, prediction, and simulation, combining machine learning with Monte Carlo Tree Search (MCTS) for optimal draft strategies.

## Features

- **Draft Analytics & Prediction**
  - Win probability prediction
  - MCTS-based draft completion simulation
  - Top-5 hero recommendations for each phase
  - Feature importance analysis
  - Side bias detection and visualization

- **Draft Interface**
  - Ban/Pick phases with timer
  - Role-based hero filtering
  - Hero search functionality
  - Real-time updates
  - Drag-and-drop interface

- **Analysis Tools**
  - Team composition analysis
  - Hero synergy & counter matrices
  - Side-specific win rates with confidence intervals
  - Cohen's h effect size analysis
  - Interactive visualizations

## Installation & Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/mlbb_counter_system.git
cd mlbb_counter_system
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install the package in development mode:
```bash
pip install -e .
```

## Data Preparation

1. Place your match data in `data/raw/` directory. Supported formats:
   - CSV with columns: match_id, patch_version, blue_picks, red_picks, blue_bans, red_bans, winner
   - JSON with the same fields

2. Data format requirements:
   - Pick/ban lists should be JSON-serialized arrays of hero names
   - Winner should be 'blue' or 'red'
   - Example CSV row:
     ```
     match_id,patch_version,blue_picks,red_picks,blue_bans,red_bans,winner
     1001,"1.6.44",["Chou","Gusion"],["Franco","Fanny"],["Ling","Lancelot"],["Wanwan","Beatrix"],"blue"
     ```

## Training the Model

1. Train the baseline model:
```bash
python scripts/train.py --data data/raw --output models/baseline
```

This will:
- Load match data from the data directory
- Compute hero features (pick rates, ban rates, win rates)
- Train a GradientBoosting model
- Save the model as `models/baseline/baseline.joblib`

## Running the API

1. Start the FastAPI server:
```bash
uvicorn api.main:app --reload --port 8000
```

2. Access the API documentation at `http://localhost:8000/docs`

### API Usage Examples

1. Get current draft prediction:
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "blue_picks": ["Chou", "Gusion"],
    "red_picks": ["Franco", "Fanny"],
    "blue_bans": ["Ling", "Lancelot", "Hayabusa"],
    "red_bans": ["Wanwan", "Beatrix", "Karrie"]
  }'
```

Response includes:
- Blue team win probability
- Current phase (PICK/BAN)
- Top 5 hero recommendations with win probabilities
- Feature importance scores

2. Get hero statistics:
```bash
curl http://localhost:8000/stats
```

3. Analyze side bias:
```bash
curl "http://localhost:8000/side-bias?min_games=10&top_n=20"
```

## Draft Simulation Examples

1. Early Ban Phase:
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "blue_picks": [],
    "red_picks": [],
    "blue_bans": ["Ling"],
    "red_bans": [],
    "patch_version": "1.6.44"
  }'
```

2. Mid Pick Phase:
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "blue_picks": ["Chou", "Gusion"],
    "red_picks": ["Franco"],
    "blue_bans": ["Ling", "Lancelot", "Hayabusa"],
    "red_bans": ["Wanwan", "Beatrix", "Karrie"],
    "patch_version": "1.6.44"
  }'
```

## Side-Bias Analysis & Plots

The system provides several ways to analyze and visualize side bias:

1. Generate a forest plot of side bias:
```python
from data_loader import MLBBDataLoader

loader = MLBBDataLoader("data/raw")
loader.load_matches()

# Matplotlib static plot
loader.plot_side_bias(min_games=10, top_n=20, save_path="side_bias.png")

# Interactive Plotly plot
fig = loader.plotly_side_bias(min_games=10, top_n=20)
fig.write_html("side_bias.html")
```

2. Get a textual summary:
```python
summary = loader.summarize_side_bias(min_games=10, min_effect=0.2)
print(summary)
```

## Exporting Results

The system supports various export formats:

1. Side Bias Analysis:
```python
# Export to CSV
loader.export_side_bias_csv(
    "side_bias.csv",
    min_games=10,
    compression="gzip"  # Optional compression
)

# Export to JSON with extended statistics
loader.export_side_bias_json(
    "side_bias.json",
    min_games=10,
    extended_stats=True,
    compression="gzip"
)

# Get as pandas DataFrame
df = loader.get_side_bias_dataframe(min_games=10)
```

2. Hero Features:
```python
hero_features = loader.compute_hero_features()
```

The exported data includes:
- Pick/ban/win rates
- Side-specific statistics
- Confidence intervals
- Effect sizes
- Raw game counts

## Docker Deployment

Build and run the containerized API:

```bash
# Build container
docker build -t mlbb-analytics .

# Run container
docker run -p 8000:8000 -v $(pwd)/data:/app/data -v $(pwd)/models:/app/models mlbb-analytics
```

## Visualization Features

The engine provides several visualization tools to analyze drafts and MCTS decision-making:

1. **MCTS Tree Visualization**
```python
from simulator.mcts import MCTSDraftSimulator
from simulator.visualization import plot_mcts_tree

# Run simulation
simulator = MCTSDraftSimulator(model, data_loader, available_heroes)
action, win_prob = simulator.get_best_action(current_state)

# Visualize search tree
plot_mcts_tree(
    simulator.root,
    max_depth=3,
    save_path="mcts_tree.png",
    layout='kamada_kawai'  # or 'spring' for larger trees
)
```

2. **Draft Sequence Analysis**
```python
from simulator.visualization import plot_draft_sequence

# Track draft progression
states = []  # List of DraftState objects
win_probs = []  # Win probability at each step

# Plot the draft sequence with win probabilities
plot_draft_sequence(states, win_probs, save_path="draft_sequence.png")
```

3. **Win Probability Heatmap**
```python
from simulator.visualization import plot_win_probability_heatmap

# Analyze win probabilities across different draft paths
plot_win_probability_heatmap(
    simulator.root,
    depth=3,
    save_path="win_prob_heatmap.png"
)
```

4. **Value Distribution Analysis**
```python
from simulator.visualization import plot_value_distribution

# Analyze distribution of node values in the MCTS tree
plot_value_distribution(
    simulator.root,
    save_path="value_dist.png"
)
```

5. **Export for External Tools**
```python
from simulator.visualization import export_tree_data

# Export tree data for D3.js or other visualization tools
tree_data = export_tree_data(simulator.root, max_depth=3)
with open("tree_data.json", "w") as f:
    json.dump(tree_data, f, indent=2)
```

## Hero Interaction Analysis

The system provides tools to analyze and visualize hero synergies and counters:

1. **Via HTTP API**
```bash
# Get synergy matrix data
curl "http://localhost:8000/stats/synergy?min_games=10"

# Get counter matrix data
curl "http://localhost:8000/stats/counter?min_games=10"
```

2. **Via Python API**
```python
from data_loader import MLBBDataLoader
from simulator.visualization import plot_synergy_matrix, plot_counter_matrix

loader = MLBBDataLoader("data/raw")
loader.load_matches()

# Compute hero features
hero_features = loader.compute_hero_features()

# Plot synergy matrix
plot_synergy_matrix(
    hero_features,
    min_games=10,  # Filter heroes by minimum games played
    highlight_threshold=0.3,  # Highlight strong synergies
    save_path="synergy_matrix.png"
)

# Plot counter matrix
plot_counter_matrix(
    hero_features,
    min_games=10,
    highlight_threshold=0.3,  # Highlight strong counters
    save_path="counter_matrix.png"
)
```

3. **Understanding the Matrices**

- **Synergy Matrix**:
  - Values range from -1 to 1
  - Positive values (blue) indicate good synergy
  - Negative values (red) indicate poor synergy
  - Darker colors indicate stronger effects
  - Annotations show the exact values

- **Counter Matrix**:
  - Values range from -1 to 1
  - Positive values (blue) mean row hero counters column hero
  - Negative values (red) mean column hero counters row hero
  - Darker colors indicate stronger counter relationships

## Example Usage

The repository includes example drafts and a visualization script to demonstrate all features:

1. **Run the Examples**:
```bash
# Create output directory
mkdir -p examples/output

# Run visualization examples
python examples/visualization_examples.py
```

This will generate:
- MCTS trees for early, mid, and late draft stages
- Draft sequence visualization with win probabilities
- Win probability heatmap
- Synergy and counter matrices
- Value distribution analysis

2. **Example Draft States**:
- `examples/early_draft.json`: Initial ban phase
- `examples/mid_draft.json`: Mid-pick phase
- `examples/late_draft.json`: Final pick decisions

3. **Visualization Output**:
All visualizations are saved to `examples/output/` with descriptive filenames.

4. **Interactive Exploration**:
```python
from examples.visualization_examples import load_draft_state
from simulator.mcts import MCTSDraftSimulator

# Load a draft state
state = load_draft_state("examples/mid_draft.json")

# Run simulation and visualize
simulator = MCTSDraftSimulator(model, data_loader, available_heroes)
action, win_prob = simulator.get_best_action(state)

# Analyze results
print(f"Recommended action: {action}")
print(f"Win probability: {win_prob:.1%}")
```

## Visualization Options

Each visualization function accepts customization parameters:

1. **MCTS Tree**:
```python
plot_mcts_tree(
    root_node,
    max_depth=3,  # How deep to visualize
    layout='kamada_kawai',  # or 'spring'
    node_size_scale=1000,  # Adjust node sizes
    colormap='RdYlBu'  # Any matplotlib colormap
)
```

2. **Hero Matrices**:
```python
plot_hero_matrix(
    matrix,
    heroes,
    title="Hero Interactions",
    figsize=(12, 10),
    cmap='RdBu',  # Colormap for values
    annotate=True,  # Show values in cells
    highlight_threshold=0.3  # Bold significant values
)
```

3. **Draft Sequence**:
```python
plot_draft_sequence(
    states,  # List of DraftState objects
    win_probs,  # Win probability at each state
    figsize=(12, 8)
)
```

4. **Win Probability Heatmap**:
```python
plot_win_probability_heatmap(
    root_node,
    depth=3,  # How many moves ahead
    figsize=(10, 8)
)
```

## Contributing

Contributions are welcome! Please read our contributing guidelines and submit pull requests to our repository.

## License

This project is licensed under the MIT License - see the LICENSE file for details.