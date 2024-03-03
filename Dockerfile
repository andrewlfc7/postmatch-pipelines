FROM python:3.9

LABEL authors="andrew"

# Set the working directory in the container
WORKDIR /app

# Copy everything from the context into the /app directory
#COPY . .

COPY Post_Match_Dashboard /app/Post_Match_Dashboard
COPY Football_Analysis_Tools /app/Football_Analysis_Tools
COPY requirements.txt /app/requirements.txt


# Install dependencies
RUN apt-get update && \
    apt-get install -y python3 && \
    apt-get install -y python3-pip && \
    pip3 install --no-cache-dir -r requirements.txt


# Set the environment variable
ENV PYTHONPATH "${PYTHONPATH}:/app"

EXPOSE 80

# Command to run the application
#CMD ["sh", "-c", "python3 Post_Match_Dashboard/pipeline/db.py & sleep 20 && python3 Post_Match_Dashboard/main.py"]


CMD ["sh", "-c", "python3 Post_Match_Dashboard/main.py"]



