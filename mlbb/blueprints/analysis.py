"""
Analysis blueprint for handling draft analysis functionality.
"""
from flask import Blueprint, jsonify, render_template, request, current_app, url_for, send_file
import json
import io
import matplotlib.pyplot as plt
from ..utils.data_loader import DataLoader
from examples.visualization_examples import plot_role_trends
import plotly.graph_objects as go

bp = Blueprint('analysis', __name__, url_prefix='/analysis')

@bp.route('/')
def analysis_page():
    """Render the analysis page."""
    # Example data - in a real app, this would come from your database/API
    ally_bans = [
        {"name": "Franco", "img": url_for('static', filename='img/Franco.png')},
        {"name": "Khufra", "img": url_for('static', filename='img/Khufra.png')},
        {"name": "Atlas", "img": url_for('static', filename='img/Atlas.png')}
    ]
    ally_picks = [
        {"name": "Beatrix", "img": url_for('static', filename='img/Beatrix.png')},
        {"name": "Luo Yi", "img": url_for('static', filename='img/Luo Yi.png')},
        {"name": "Gusion", "img": url_for('static', filename='img/Gusion.png')},
        {"name": "Akai", "img": url_for('static', filename='img/Akai.png')},
        {"name": "Angela", "img": url_for('static', filename='img/Angela.png')}
    ]
    
    return render_template('analysis.html',
                         ally_bans=ally_bans,
                         ally_picks=ally_picks)

@bp.route('/recommend', methods=['POST'])
def recommend_picks():
    """Get hero recommendations based on current draft state."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
            
        ally_picks = data.get('ally_picks', [])
        enemy_picks = data.get('enemy_picks', [])
        ally_bans = data.get('ally_bans', [])
        enemy_bans = data.get('enemy_bans', [])
        
        hero_data = DataLoader.get_hero_data()
        pro_stats = DataLoader.get_pro_stats()
        
        # Filter out banned and picked heroes
        available_heroes = {
            name: data for name, data in hero_data.items()
            if name not in ally_picks + enemy_picks + ally_bans + enemy_bans
        }
        
        # Calculate recommendations based on team composition
        recommendations = []
        for name, data in available_heroes.items():
            score = 0
            # Add scoring logic here based on:
            # - Counter picks against enemy team
            # - Synergy with ally team
            # - Current meta (pro_stats)
            # - Role balance
            recommendations.append({
                'hero': name,
                'score': score,
                'reason': 'Placeholder recommendation reason'
            })
        
        # Sort by score
        recommendations.sort(key=lambda x: x['score'], reverse=True)
        
        return jsonify({
            'recommendations': recommendations[:5],
            'meta_analysis': {
                'team_composition': 'Balanced',
                'missing_roles': [],
                'counter_suggestions': []
            }
        })
    except Exception as e:
        current_app.logger.error(f"Error generating recommendations: {str(e)}")
        return jsonify({'error': 'Failed to generate recommendations'}), 500

@bp.route('/role-trends', methods=['GET'])
def get_role_trends():
    """Generate and return interactive role trend analysis visualization"""
    patch_stats = get_patch_statistics()
    patches = sorted(patch_stats.keys())
    
    with open('static/data/hero_roles.json', 'r') as f:
        role_data = json.load(f)
    
    # Create interactive plot
    fig = go.Figure()
    
    for role in set(role_data.values()):
        role_winrates = [
            patch_stats[patch]['roles'].get(role, {}).get('win_rate', 0) 
            for patch in patches
        ]
        fig.add_trace(go.Scatter(
            x=patches,
            y=role_winrates,
            name=role,
            mode='lines+markers',
            hovertemplate="Patch: %{x}<br>Win Rate: %{y:.2%}<extra></extra>"
        ))
    
    fig.update_layout(
        title='Role Win Rates Across Patches',
        xaxis_title='Patch',
        yaxis_title='Win Rate',
        yaxis_tickformat='.0%',
        hovermode='x unified'
    )
    
    return fig.to_html(
        full_html=False,
        include_plotlyjs='cdn',
        config={'displayModeBar': True}
    )