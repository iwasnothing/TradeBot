import os
from google.cloud import secretmanager_v1
from flask import Flask
from flask import request
from google.cloud import storage
from zipfile import ZipFile
from TradeBot import TradeBot
from os.path import basename
import json


app = Flask(__name__)

PRJID="139391369285"
def init_vars():
    client = secretmanager_v1.SecretManagerServiceClient()
    name = client.secret_version_path(PRJID,'APCA_API_KEY_ID',  'latest')
    print(name)
    #name = "projects/139391369285/secrets/APCA_API_KEY_ID/versions/1"
    response = client.access_secret_version(name)
    os.environ["APCA_API_KEY_ID"] = response.payload.data.decode('UTF-8')
    name = client.secret_version_path(PRJID,'APCA_API_SECRET_KEY',  'latest')
    print(name)
    response = client.access_secret_version(name)
    os.environ["APCA_API_SECRET_ID"] = response.payload.data.decode('UTF-8')
    #name = os.environ.get('NAME', 'World')

def download_files(ticker1,ticker2):
    bucket_name = "iwasnothing-cloudml-job-dir"
    wdir = "/app/"
    prefix = ticker1 + "-" + ticker2 + "-"
    filename = prefix + "regression.zip"
    download_blob(bucket_name,  filename, wdir + filename)
    with ZipFile(wdir+filename, 'r') as zipObj:
        zipObj.extractall()
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


@app.route('/predict', methods=['POST'])
def predict():
    req_json = request.get_json(force=True)
    ticker1 = req_json['ticker1']
    ticker2 = req_json['ticker2']
    print(ticker1,ticker2)
    init_vars()
    download_files(ticker1,ticker2)
    #ticker1='ZM'
    #ticker2='LBTYK'
    win = 5
    past = 6
    bot = TradeBot(ticker1,ticker2,win,past)
    result = bot.predictPrice()
    print(result)
    bot.buy(ticker2,result)
    return 'Hello {}!'.format(str(result))

@app.route('/train', methods=['POST'])
def train():
    req_json = request.get_json(force=True)
    ticker1 = req_json['ticker1']
    ticker2 = req_json['ticker2']
    print(ticker1,ticker2)
    init_vars()
    #ticker1='ZM'
    #ticker2='LBTYK'
    win = 5
    past = 6
    bot = TradeBot(ticker1,ticker2,win,past)
    result = bot.pair_loss()
    print(result)
    prefix = ticker1 + "-" + ticker2 + "-"
    zip_model(prefix + 'regression')
    upload_files(ticker1,ticker2)
    return 'Hello {}!'.format(str(result))


def download_blob(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    # bucket_name = "your-bucket-name"
    # source_blob_name = "storage-object-name"
    # destination_file_name = "local/path/to/file"

    storage_client = storage.Client()

    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)

    print(
        "Blob {} downloaded to {}.".format(
            source_blob_name, destination_file_name
        )
    )

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

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
