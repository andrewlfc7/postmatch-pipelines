import subprocess
import os

# Get the current script's directory
current_directory = os.path.dirname(os.path.abspath(__file__))


# Full paths to the Python scripts
# processing_script = os.path.join(current_directory, 'processing.py')

db_script = os.path.join(current_directory, 'db.py')



scraper_script = os.path.join(current_directory, 'scraper/scraper.py')

subprocess.call(['python', scraper_script])
subprocess.call(['python', db_script])
