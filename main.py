import os
from google.cloud import secretmanager_v1
from google.cloud import storage
from google.cloud import pubsub_v1
from google.cloud import bigquery
import alpaca_trade_api as tradeapi
from flask import Flask
from flask import request
from zipfile import ZipFile
from TradeBot import TradeBot
from StockCorrelation import StockCorrelation
from MAPredictor import MAPredictor
from NewsPredictor import NewsPredictor
from os.path import basename
from datetime import date
import json
import base64


app = Flask(__name__)

PRJID="139391369285"
Q1="""
        select distinct ticker1,ticker2 from (
        with dateList as ( select create_dt from `iwasnothing-self-learning.stock_cor.stock_cor_short_list` order by create_dt desc )
        SELECT ticker1,ticker2,accuracy/(10*rmse) as score
        FROM `iwasnothing-self-learning.stock_cor.stock_cor_short_list` 
        where create_dt in (select create_dt from dateList limit 1)
        ORDER BY score desc ) LIMIT 2
"""
def init_vars():
    client = secretmanager_v1.SecretManagerServiceClient()
    secrets = ["APCA_API_KEY_ID", "APCA_API_SECRET_KEY", "NEWS_API_KEY"]
    result = {"APCA_API_BASE_URL": "https://paper-api.alpaca.markets" }
    for s in secrets:
        name = f"projects/{PRJID}/secrets/{s}/versions/latest"
        response = client.access_secret_version(request={'name': name})
        print(response)
        os.environ[s] = response.payload.data.decode('UTF-8')
        result[s] = response.payload.data.decode('UTF-8')
    print(result)
    return result

def download_files(ticker1,ticker2):
    bucket_name = "iwasnothing-cloudml-job-dir"
    wdir = "/app/"
    prefix = ticker1 + "-" + ticker2 + "-"
    filename = prefix + "regression.zip"
    download_blob(bucket_name,  filename, wdir + filename)
    with ZipFile(wdir+filename, 'r') as zipObj:
        zipObj.extractall('/app/'+prefix+"regression")
    print("after unzip")
    dirs = os.listdir("/app")
    for file in dirs:
        print(file)
    filename = prefix + "scalerX.gz"
    download_blob(bucket_name,  filename, wdir + filename)
    filename = prefix + "scalerY.gz"
    download_blob(bucket_name,  filename, wdir + filename)

def zip_model(dirName):
    # create a ZipFile object
    with ZipFile(dirName + '.zip', 'w') as zipObj:
        # Iterate over all the files in directory
        for folderName, subfolders, filenames in os.walk(dirName):
            for filename in filenames:
                # create complete filepath of file in directory
                filePath = os.path.join(folderName, filename)
                # Add file to zip
                zipObj.write(filePath, basename(filePath))
def upload_zip(dirName):
    zip_model(dirName)
    bucket_name = "iwasnothing-cloudml-job-dir"
    wdir = "/app/"
    filename = dirName + ".zip"
    upload_blob(bucket_name, wdir + filename,  filename)
def download_zip(dirName):
    bucket_name = "iwasnothing-cloudml-job-dir"
    wdir = "/app/"
    filename = dirName + ".zip"
    download_blob(bucket_name,  filename, wdir + filename)
    with ZipFile(wdir+filename, 'r') as zipObj:
        zipObj.extractall(wdir+dirName)
    print("after unzip")
    dirs = os.listdir("/app")
    for file in dirs:
        print(file)

def upload_files(ticker1 , ticker2):
    bucket_name = "iwasnothing-cloudml-job-dir"
    wdir = "/app/"
    prefix = ticker1 + "-" + ticker2 + "-"
    filename = prefix + "regression.zip"
    upload_blob(bucket_name, wdir + filename,  filename)
    filename = prefix + "scalerX.gz"
    upload_blob(bucket_name, wdir + filename,  filename)
    filename = prefix + "scalerY.gz"
    upload_blob(bucket_name, wdir + filename,  filename)

def download_list():
    bucket_name = "iwasnothing-cloudml-job-dir"
    wdir = "/app/"
    filename = "list.txt"
    download_blob(bucket_name,  filename, wdir + filename)

@app.route('/genstocks', methods=['GET'])
def genStocks():
    project_id = "iwasnothing-self-learning"
    topic_id = "stock_pair"
    publisher = pubsub_v1.PublisherClient()
    topic_path = publisher.topic_path(project_id, topic_id)
    download_list()
    with open('/app/list.txt', 'r') as fp:
        list = fp.read().splitlines()
        print(list)
        for ticker1 in list:
            for ticker2 in list:
                if ticker2 != ticker1:
                    # The `topic_path` method creates a fully qualified identifier
                    # in the form `projects/{project_id}/topics/{topic_id}`
                    data = {"ticker1": ticker1, "ticker2": ticker2}
                    # Data must be a bytestring
                    data = json.dumps(data)
                    data = data.encode("utf-8")
                    # When you publish a message, the client returns a future.
                    future = publisher.publish(topic_path, data)
                    print(future.result())
    return ('', 204)

@app.route('/correlation', methods=['POST'])
def correlation():
    envelope = request.get_json()
    if not envelope:
        msg = 'no Pub/Sub message received'
        print(f'error: {msg}')
        return f'Bad Request: {msg}', 400

    if not isinstance(envelope, dict) or 'message' not in envelope:
        msg = 'invalid Pub/Sub message format'
        print(f'error: {msg}')
        return f'Bad Request: {msg}', 400

    pubsub_message = envelope['message']

    name = 'World'
    if isinstance(pubsub_message, dict) and 'data' in pubsub_message:
        name = base64.b64decode(pubsub_message['data']).decode('utf-8').strip()
        print(f'Hello {name}!')
        req_json = json.loads(name)
        ticker1 = req_json['ticker1']
        ticker2 = req_json['ticker2']
        print(ticker1,ticker2)
        r = StockCorrelation()
        cor = r.pair_loss(ticker1,ticker2,period='2y',win=5)
        print(cor)
        project_id = "iwasnothing-self-learning"
        topic_id = "stock_filter"
        publisher = pubsub_v1.PublisherClient()
        topic_path = publisher.topic_path(project_id, topic_id)
        data = cor
        # Data must be a bytestring
        data = json.dumps(data)
        data = data.encode("utf-8")
        # When you publish a message, the client returns a future.
        future = publisher.publish(topic_path, data)
        print(future.result())
        return (data, 204)

    return ('', 204)

@app.route('/filter', methods=['POST'])
def filter():
    envelope = request.get_json()
    if not envelope:
        msg = 'no Pub/Sub message received'
        print(f'error: {msg}')
        return f'Bad Request: {msg}', 400

    if not isinstance(envelope, dict) or 'message' not in envelope:
        msg = 'invalid Pub/Sub message format'
        print(f'error: {msg}')
        return f'Bad Request: {msg}', 400

    pubsub_message = envelope['message']

    name = 'World'
    if isinstance(pubsub_message, dict) and 'data' in pubsub_message:
        name = base64.b64decode(pubsub_message['data']).decode('utf-8').strip()
        print(f'Filter {name}!')
        req_json = json.loads(name)
        ticker1 = req_json['ticker1']
        ticker2 = req_json['ticker2']
        rmse = req_json['rmse']
        acc = req_json['accuracy']
        future = req_json['future']
        today = date.today()
        todstr = today.strftime("%Y-%m-%d")
        print(ticker1,ticker2,rmse,acc,future,today)
        if rmse < 0.05 and acc > 0.5:
            print("insert stock")
            bigquery_client = bigquery.Client()
            # Prepares a reference to the dataset
            dataset_ref = bigquery_client.dataset('stock_cor')
            table_ref = dataset_ref.table('stock_cor_short_list')
            table = bigquery_client.get_table(table_ref)  # API call

            rows_to_insert = [
                (ticker1, ticker2, rmse, acc, future, todstr)
            ]
            errors = bigquery_client.insert_rows(table, rows_to_insert)
            print(errors)
    return ('', 204)

@app.route('/predictall', methods=['GET'])
def predictAll():
    # Construct a BigQuery client object.
    project_id = "iwasnothing-self-learning"
    topic_id = "stock_predict"
    publisher = pubsub_v1.PublisherClient()
    topic_path = publisher.topic_path(project_id, topic_id)
    client = bigquery.Client()

    #today = date.today()
    #todstr = today.strftime("%Y-%m-%d")
    query_job = client.query(Q1)  # Make an API request.
    print("The query data:")
    for row in query_job:
        for val in row:
            print(val)
        # The `topic_path` method creates a fully qualified identifier
        # in the form `projects/{project_id}/topics/{topic_id}`
        print(row)
        data = {"ticker1": row[0], "ticker2": row[1]}
        data = json.dumps(data)
        print(data)
        # Data must be a bytestring
        data = data.encode("utf-8")
        # When you publish a message, the client returns a future.
        future = publisher.publish(topic_path, data)
        print(future.result())
    return ('', 204)


@app.route('/predict', methods=['POST'])
def predict():
    envelope = request.get_json()
    if not envelope:
        msg = 'no Pub/Sub message received'
        print(f'error: {msg}')
        return f'Bad Request: {msg}', 400

    if not isinstance(envelope, dict) or 'message' not in envelope:
        msg = 'invalid Pub/Sub message format'
        print(f'error: {msg}')
        return f'Bad Request: {msg}', 400

    pubsub_message = envelope['message']

    name = 'World'
    if isinstance(pubsub_message, dict) and 'data' in pubsub_message:
        name = base64.b64decode(pubsub_message['data']).decode('utf-8').strip()
        print(f'Filter {name}!')
        req_json = json.loads(name)
        ticker1 = req_json['ticker1']
        ticker2 = req_json['ticker2']
        print(ticker1,ticker2)
        key = init_vars()
        download_files(ticker1,ticker2)
        #ticker1='ZM'
        #ticker2='LBTYK'
        win = 5
        past = 6
        bot = TradeBot(ticker1,ticker2,win,past,key)
        result = bot.predictPrice()
        print(result)
        bot.buy(ticker2,result)
        return 'Hello {}!'.format(str(result))
    return ('', 204)

@app.route('/trainall', methods=['GET'])
def trainAll():
    # Construct a BigQuery client object.
    project_id = "iwasnothing-self-learning"
    topic_id = "stock_train"
    publisher = pubsub_v1.PublisherClient()
    topic_path = publisher.topic_path(project_id, topic_id)
    client = bigquery.Client()

    #today = date.today()
    #todstr = today.strftime("%Y-%m-%d")
    query_job = client.query(Q1)  # Make an API request.

    print("The query data:")
    for row in query_job:
        for val in row:
            print(val)
        # The `topic_path` method creates a fully qualified identifier
        # in the form `projects/{project_id}/topics/{topic_id}`
        print(row[0])
        print(row[1])
        data = {"ticker1": row[0], "ticker2": row[1]}
        data = json.dumps(data)
        print(data)
        # Data must be a bytestring
        data = data.encode("utf-8")
        # When you publish a message, the client returns a future.
        future = publisher.publish(topic_path, data)
        print(future.result())
    return ('', 204)



@app.route('/train', methods=['POST'])
def train():
    envelope = request.get_json()
    if not envelope:
        msg = 'no Pub/Sub message received'
        print(f'error: {msg}')
        return f'Bad Request: {msg}', 400

    if not isinstance(envelope, dict) or 'message' not in envelope:
        msg = 'invalid Pub/Sub message format'
        print(f'error: {msg}')
        return f'Bad Request: {msg}', 400

    pubsub_message = envelope['message']

    name = 'World'
    if isinstance(pubsub_message, dict) and 'data' in pubsub_message:
        name = base64.b64decode(pubsub_message['data']).decode('utf-8').strip()
        print(f'Filter {name}!')
        req_json = json.loads(name)
        print(req_json)
        ticker1 = req_json['ticker1']
        ticker2 = req_json['ticker2']
        print(ticker1,ticker2)
        key = init_vars()
        #ticker1='ZM'
        #ticker2='LBTYK'
        win = 5
        past = 6
        bot = TradeBot(ticker1,ticker2,win,past, key)
        result = bot.pair_loss()
        print(result)
        prefix = ticker1 + "-" + ticker2 + "-"
        zip_model(prefix + 'regression')
        upload_files(ticker1,ticker2)
        return 'Hello {}!'.format(str(result))
    return ('', 204)

def place_order(ticker,sperad):
    key = init_vars()
    api = tradeapi.REST(key['APCA_API_KEY_ID'], key['APCA_API_SECRET_KEY'], key['APCA_API_BASE_URL'], 'v2')
    #account = api.get_account()
    ticker_bars = api.get_barset(ticker, 'minute', 1).df.iloc[0]
    ticker_price = ticker_bars[ticker]['close']
    print(ticker_price)

    # We could buy a position and add a stop-loss and a take-profit of 5 %
    try:
        api.submit_order(
            symbol=ticker,
            qty=1,
            side='buy',
            type='market',
            time_in_force='gtc',
            order_class='bracket',
            stop_loss={'stop_price': ticker_price * (1 - spread),
                        'limit_price': ticker_price * (1 - spread) * 0.95},
            take_profit={'limit_price': ticker_price * (1 + spread)}
        )
    except Exception as e:
        print(e)


@app.route('/trade', methods=['POST'])
def trade():
    envelope = request.get_json()
    if not envelope:
        msg = 'no Pub/Sub message received'
        print(f'error: {msg}')
        return f'Bad Request: {msg}', 400

    if not isinstance(envelope, dict) or 'message' not in envelope:
        msg = 'invalid Pub/Sub message format'
        print(f'error: {msg}')
        return f'Bad Request: {msg}', 400

    pubsub_message = envelope['message']

    name = 'World'
    if isinstance(pubsub_message, dict) and 'data' in pubsub_message:
        name = base64.b64decode(pubsub_message['data']).decode('utf-8').strip()
        print(f'Filter {name}!')
        msg = json.loads(name)
        ticker = msg["ticker"]
        spread = msg['spread']
        print(ticker)
        print(spread)
        place_order(ticker,spread)
    return ('', 204)


    

def download_blob(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    # bucket_name = "your-bucket-name"
    # source_blob_name = "storage-object-name"
    # destination_file_name = "local/path/to/file"

    storage_client = storage.Client()
    print("dowbload_blob")
    print(bucket_name, source_blob_name, destination_file_name)
    if os.path.exists(destination_file_name):
        os.remove(destination_file_name)
    else:
        print(destination_file_name + " doesn't exists")
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    if blob.exists():
        try:
            ret = blob.download_to_filename(destination_file_name)
            print("download ",destination_file_name,ret)
            if ret is not None:
                print("Blob {} downloaded to {}.".format(source_blob_name, destination_file_name))
                return True
            else:
                dirs = os.listdir("/app")
                for file in dirs:
                    print(file)
        except:
            return False
    else:
        print("blob not exist")
    return False


def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    # bucket_name = "your-bucket-name"
    # source_file_name = "local/path/to/file"
    # destination_blob_name = "storage-object-name"

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)

    print(
        "File {} uploaded to {}.".format(
            source_file_name, destination_blob_name
        )
    )

@app.route('/dayend')
def sellAll():
    key = init_vars()
    api = tradeapi.REST(key['APCA_API_KEY_ID'], key['APCA_API_SECRET_KEY'], key['APCA_API_BASE_URL'], 'v2')
    api.cancel_all_orders()
    portfolio = api.list_positions()

    # Print the quantity of shares for each position.
    for position in portfolio:
        print("current position is {}".format(position.qty))
        try:
            api.submit_order(
                symbol=position.symbol ,
                qty=position.qty,
                side='sell',
                type='market',
                time_in_force='day'
            )
        except Exception as e:
            print(e)
    return ('', 204)

@app.route('/newspredict')
def news_train_predict():
    download_list()
    key = init_vars()
    newspredict = NewsPredictor(key=key['NEWS_API_KEY'],list="/app/list.txt")
    newspredict.training()
    upload_zip(newspredict.getModelLocation())
    download_zip(newspredict.getModelLocation())
    newspredict.predict()
    df = newspredict.getShortList()
    print(df)
    print(df.info())
    #client = bigquery.Client()
    # Prepares a reference to the dataset
    #dataset_ref = bigquery_client.dataset('stock_cor')
    #table_ref = dataset_ref.table('news_predict_short_list')
    #table = bigquery_client.get_table(table_ref)  # API call
    project_id = "iwasnothing-self-learning"
    table_id = project_id + ".stock_cor.news_predict_short_list"
    client = bigquery.Client()
    today = date.today()
    todstr = today.strftime("%Y-%m-%d")
    df['create_dt'] = todstr
    job_config = bigquery.LoadJobConfig(
        schema=[
            bigquery.SchemaField("stock", bigquery.enums.SqlTypeNames.STRING),
            bigquery.SchemaField("avg", bigquery.enums.SqlTypeNames.FLOAT64),
            bigquery.SchemaField("count", bigquery.enums.SqlTypeNames.INT64),
            bigquery.SchemaField("create_dt", bigquery.enums.SqlTypeNames.STRING),
        ]
    )
    job = client.load_table_from_dataframe(
        df, table_id, job_config=job_config
    )  # Make an API request.
    job.result()
    return ('', 204)

@app.route('/mapredict')
def ma_train_predict():
    download_list()
    #key = init_vars()
    ma = MAPredictor("/app/list.txt")
    ma.trainAll()
    df = ma.getShortList()
    project_id = "iwasnothing-self-learning"
    table_id = project_id + ".stock_cor.ma_predict_short_list"
    client = bigquery.Client()
    today = date.today()
    todstr = today.strftime("%Y-%m-%d")
    df['create_dt'] = todstr
    job_config = bigquery.LoadJobConfig(
        schema=[
            bigquery.SchemaField("stock", bigquery.enums.SqlTypeNames.STRING),
            bigquery.SchemaField("accuracy", bigquery.enums.SqlTypeNames.FLOAT64),
            bigquery.SchemaField("prediction", bigquery.enums.SqlTypeNames.STRING),
            bigquery.SchemaField("create_dt", bigquery.enums.SqlTypeNames.STRING),
        ]
    )
    job = client.load_table_from_dataframe(
        df, table_id, job_config=job_config
    )  # Make an API request.
    job.result()
    return ('', 204)

@app.route('/buyperday', methods=['POST'])
def buyperday():
    envelope = request.get_json()
    # Construct a BigQuery client object.
    #project_id = "iwasnothing-self-learning"
    client = bigquery.Client()
    print(envelope)
    qstr = envelope['query']
    #today = date.today()
    #todstr = today.strftime("%Y-%m-%d")
    query_job = client.query(qstr)  # Make an API request.

    print("The query data:")
    for row in query_job:
        print(row[0])
        place_order(row[0],0.03)
    return ('', 204)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
