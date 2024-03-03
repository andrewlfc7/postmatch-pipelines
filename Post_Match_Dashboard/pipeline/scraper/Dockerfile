# Stage 1: Selenium Standalone for data retrieval
FROM selenium/standalone-chrome:latest

WORKDIR /app


COPY . ./. ./. ./



USER root
RUN apt-get update && \
    apt-get install -y python3 python3-pip && \
    pip3 install --upgrade pip && \
    pip3 install -r requirements.txt && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

COPY start-selenium-standalone.sh /opt/bin/start-selenium-standalone.sh

RUN chmod +x /opt/bin/start-selenium-standalone.sh

#==============================
# Supervisor configuration file
#==============================
COPY selenium.conf /etc/supervisor/conf.d/

# Copying configuration script generator
COPY generate_config /opt/bin/generate_config
RUN chmod +x /opt/bin/generate_config

# Boolean value, maps "--relax-checks"
ENV SE_RELAX_CHECKS true
ENV PYTHONUNBUFFERED=1

EXPOSE 4444
EXPOSE 7900
EXPOSE 80
#CMD ["sh", "-c", "/opt/bin/start-selenium-standalone.sh & sleep 20 && python3 Post_Match_Dashboard/pipeline/scraper.py"]

CMD ["sh", "-c", "/opt/bin/start-selenium-standalone.sh & sleep 20 && python3 scraper.py"]


