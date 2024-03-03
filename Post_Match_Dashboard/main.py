import subprocess
import os
#
# # Get the current script's directory
# current_directory = os.path.dirname(os.path.abspath(__file__))
#
# # Full paths to the Python scripts
# stats_avg_script = os.path.join(current_directory, 'stats_avg.py')
# post_match_script = os.path.join(current_directory, 'Post_match.py')
# player_dashboard_script = os.path.join(current_directory, 'player_dashboard.py')
# db_script_path = os.path.join(current_directory, 'pipeline', 'db.py')

# Run the scripts
subprocess.run(['python', 'pipeline/db.py'])

subprocess.run(['python', 'stats_avg.py'], check=True)
subprocess.run(['python', 'Post_match.py'], check=True)
subprocess.run(['python', 'player_dashboard.py'], check=True)
subprocess.run(['python', 'pipeline/db.py'])

#
# # twitter_api_script = os.path.join(current_directory, 'twitter-api.py')
# # subprocess.run(['python', twitter_api_script])



#
# import os
# import subprocess
# from joblib import Parallel, delayed
#
# # Get the current script's directory
# current_directory = os.path.dirname(os.path.abspath(__file__))
#
# # Full paths to the Python tasks
# stats_avg_task = os.path.join(current_directory, 'stats_avg.py')
# post_match_task = os.path.join(current_directory, 'Post_match.py')
# player_dashboard_task = os.path.join(current_directory, 'player_dashboard.py')
# db_task_path = os.path.join(current_directory, 'pipeline', 'db.py')
#
# # Function to run a task and handle errors
# def run_task(task_path):
#     try:
#         subprocess.run(['python', task_path], check=True)
#     except subprocess.CalledProcessError as e:
#         print(f"Error running {task_path}: {e}")
#
# # Run the database task first
# print("Running the database task...")
# run_task(db_task_path)
# print("Database task completed.")
#
# # List of tasks to run in parallel
# tasks = [stats_avg_task, post_match_task, player_dashboard_task]
#
# # Run the remaining tasks in parallel using joblib
# print("Running the other tasks in parallel...")
# Parallel(n_jobs=-1)(delayed(run_task)(task) for task in tasks)
#
# print("All tasks have been executed.")
