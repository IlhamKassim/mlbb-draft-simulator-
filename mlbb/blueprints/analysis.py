"""
Analysis blueprint for handling draft analysis functionality.
"""
from flask import Blueprint, jsonify, render_template, request, current_app, url_for
from ..utils.data_loader import DataLoader

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