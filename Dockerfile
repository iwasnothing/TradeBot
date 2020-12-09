
# Use the official lightweight Python image.
# https://hub.docker.com/_/python
FROM ubuntu:20.10

# Allow statements and log messages to immediately appear in the Knative logs
ENV PYTHONUNBUFFERED True
ENV APCA_API_BASE_URL https://paper-api.alpaca.markets

# Copy local code to the container image.
ENV APP_HOME /app
WORKDIR $APP_HOME

RUN apt-get update -y
RUN apt-get install python3 -y
RUN apt-get install python3-pip -y
RUN apt-get install -y libstdc++6 python-setuptools libgconf-2-4
# Install production dependencies.
RUN pip install --upgrade google-cloud-pubsub
RUN pip install --upgrade google-cloud-secret-manager
RUN pip install --upgrade google-cloud-storage
RUN pip install --upgrade google-cloud-bigquery
RUN pip install Flask gunicorn
RUN pip install turicreate
RUN pip install alpaca-trade-api
RUN pip install numpy
RUN pip install pandas
RUN pip install joblib
RUN pip install scikit-learn
RUN pip install matplotlib
RUN pip install yfinance

COPY . ./
# Run the web service on container startup. Here we use the gunicorn
# webserver, with one worker process and 8 threads.
# For environments with multiple CPU cores, increase the number of workers
# to be equal to the cores available.
CMD [ "gunicorn" ,"--bind", ":8080" ,"--workers", "1" , "--threads", "8" , "--timeout", "0", "main:app" ]
