#!/bin/sh
DIR="/Users/kahingleung/PycharmProjects/newturi"
cp $DIR/main.py main.py
cp $DIR/TradeBot.py TradeBot.py
cp $DIR/StockCorrelation.py StockCorrelation.py
APCA_API_KEY_ID=`gcloud secrets versions access latest --secret=APCA_API_KEY_ID`
APCA_API_SECRET_KEY=`gcloud secrets versions access latest --secret=APCA_API_SECRET_KEY`
echo $APCA_API_KEY_ID
echo $APCA_API_SECRET_KEY
gcloud builds submit --tag gcr.io/iwasnothing-self-learning/helloworld
gcloud run deploy helloworld --image gcr.io/iwasnothing-self-learning/helloworld --platform managed \
--memory=512Mi \
--concurrency=1 \
--cpu=1 \
--max-instances=1 \
--set-env-vars "APCA_API_KEY_ID=$APCA_API_KEY_ID,APCA_API_SECRET_KEY=$APCA_API_SECRET_KEY,APCA_API_BASE_URL=https://paper-api.alpaca.markets" \
--service-account=pubsubsa@iwasnothing-self-learning.iam.gserviceaccount.com \
--no-allow-unauthenticated \
--region=us-central1 \
--timeout=5m
