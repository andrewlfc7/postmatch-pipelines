
import subprocess
from joblib import Parallel, delayed

# subprocess.run(['python', 'Post_Match_Dashboard/pipeline/db.py'], check=True)
#
# scripts = [
#     'Post_Match_Dashboard/stats_avg.py',
#     'Post_Match_Dashboard/Post_match.py',
#     'Post_Match_Dashboard/player_dashboard.py',
#     'Post_Match_Dashboard/players_stats.py',
# ]
#
# Parallel(n_jobs=len(scripts))(delayed(subprocess.run)(['python', script], check=True) for script in scripts[1:])

subprocess.run(['python', 'Post_Match_Dashboard/x-api-v1.py'],check=True)
