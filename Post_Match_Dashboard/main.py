import subprocess
import os

subprocess.run(['python', 'Post_Match_Dashboard/pipeline/db.py'],check=True)

subprocess.run(['python', 'Post_Match_Dashboard/stats_avg.py'], check=True)
subprocess.run(['python', 'Post_Match_Dashboard/Post_match.py'], check=True)
subprocess.run(['python', 'Post_Match_Dashboard/player_dashboard.py'], check=True)
subprocess.run(['python', 'Post_Match_Dashboard/players_stats.py'],check=True)

subprocess.run(['python', 'Post_Match_Dashboard/x-api-v1.py'])


