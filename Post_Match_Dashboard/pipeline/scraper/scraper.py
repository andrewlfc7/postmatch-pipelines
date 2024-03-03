import json
import pandas as pd
import requests
from retrying import retry
import os


from google.cloud.storage.blob import Blob
from google.cloud import storage



from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium import webdriver
from selenium.common.exceptions import TimeoutException
from time import sleep

import os
import json

# Get the current script's directory
current_directory = os.path.dirname(os.path.abspath(__file__))
# Assuming scraper-key is stored in the same directory as db.py
key_file_path = os.path.join(current_directory, 'scraper-key')

# Check if the file exists
if os.path.exists(key_file_path):
    # Read the contents of the file
    with open(key_file_path, 'r') as key_file:
        key_data = key_file.read()

    # Assuming the key data is in JSON format
    key_json = json.loads(key_data)

    # Use key_json as needed in your code
    # For example, you can access individual keys like key_json['key_name']

    # Assuming GOOGLE_APPLICATION_CREDENTIALS is expected to contain the service account key path
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = key_file_path
else:
    print("Error: Scraper key file not found at", key_file_path)


@retry(stop_max_attempt_number=3)
def get_match_event_data(date):
    # Set up the webdriver
    options = webdriver.ChromeOptions()
    options.binary_location = '/usr/bin/google-chrome'
    options.add_argument('--headless')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--disable-gpu')  # Disable GPU acceleration, may help with some rendering issues
    options.add_argument('--window-size=1920,1080')  # Set a specific window size

    user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/95.0.4638.54 Safari/537.36'
    options.add_argument('user-agent={0}'.format(user_agent))
    


    driver = webdriver.Remote("http://localhost:4444", options=options)

    try:
        # Navigate to the WhoScored website
        driver.get('https://www.whoscored.com/Teams/26/Fixtures/England-Liverpool')

        try:
            # Try to find and close the ad pop-up
            close_button = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.CLASS_NAME, 'webpush-swal2-close'))
            )
            close_button.click()
        except TimeoutException:
            # Handle the case where the close button is not found (no ad pop-up)
            print("No ad pop-up or close button not found. Proceeding without closing.")




        date_to_search = date

        # Find the div element containing fixture data

        wait = WebDriverWait(driver, 20)  # Adjust the timeout as needed
        fixture_div_element = wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="team-fixtures"]')))



        # Find all rows within the fixture div element
        fixture_rows = fixture_div_element.find_elements(By.CLASS_NAME, 'divtable-row')

        match_link = ""

        # Iterate through the fixture rows and search for the desired date
        for row in fixture_rows:
            date_element = row.find_element(By.CLASS_NAME, 'date')
            if date_to_search in date_element.text:
                match_link_element = row.find_element(By.CLASS_NAME, 'box')
                match_link = match_link_element.get_attribute('href')
                print(f"Match link for date {date_to_search}: {match_link}")
                break

        # Check if a match link was found
        if match_link:
            # Navigate to the match link
            driver.get(match_link)

            # Wait for the chalkboard link and click on it
            chalkboard = driver.find_element(By.XPATH, '//*[@id="sub-navigation"]/ul/li[4]/a')
            chalkboard.click()

            # Wait for the script element to be present
            wait = WebDriverWait(driver, 15)
            script_element = wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="layout-wrapper"]/script[1]')))

            # Get the text content of the script element
            script_text = script_element.get_attribute("textContent")

            return script_text

    finally:
        # Close the webdriver in the finally block to ensure it is closed even if an exception occurs
        driver.quit()



def data_preprocessing(script_text):
    edited_content = script_text.replace('require.config.params["args"] = ', '')
    # edited_content = edited_content.replace('{\n', '')
    edited_content = edited_content.replace('\n', '')
    edited_content = edited_content.replace(';', '')

    edited_content = edited_content.replace('matchId', '"matchId"')
    edited_content = edited_content.replace('matchCentreData', '"matchCentreData"')
    edited_content = edited_content.replace('matchCentreEventTypeJson', '"matchCentreEventTypeJson"')
    edited_content = edited_content.replace('formationIdNameMappings', '"formationIdNameMappings"')

    # Parse JSON
    parsed_json = json.loads(edited_content)

    return parsed_json


from datetime import datetime

# Get today's date
today_date = datetime.now().strftime("%d-%m-%y")

# Store the formatted date as match_date
match_date = today_date


script_text = get_match_event_data(match_date)


data = data_preprocessing(script_text)


# # Initialize Google Cloud Storage client
storage_client = storage.Client()

# Specify the bucket and blob name
bucket_name = "soccerdb-data"
blob_name = f"{match_date}.json"  # Change the extension to json if that's the format you're working with

# Create a blob with the specified name in the specified bucket
data_blob = storage_client.bucket(bucket_name).blob(blob_name)

# Upload the processed JSON data to the blob
data_blob.upload_from_string(json.dumps(data))



