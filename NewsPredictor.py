import requests
import ssl
import pandas as pd
import yfinance as yf
import turicreate as tc
from datetime import datetime, timedelta, date
from newsapi import NewsApiClient
import turicreate.aggregate as agg


class NewsPredictor:
    def __init__(self,key=None,list="/app/list.txt"):
        requests.packages.urllib3.disable_warnings()
        try:
            _create_unverified_https_context = ssl._create_unverified_context
        except AttributeError:
            # Legacy Python that doesn't verify HTTPS certificates by default
            pass
        else:
            # Handle target environment that doesn't support HTTPS verification
            ssl._create_default_https_context = _create_unverified_https_context
        self.key = key
        self.shortlist = []
        self.list_loc = list
        self.model_loc = "newsapiprediction.model"


    def download_news(self,ticker,period=30):
        tod = date.today()
        startday = datetime.today() - timedelta(period)
        print(startday,tod)
        newsapi = NewsApiClient(api_key=self.key)
        all_articles = newsapi.get_everything(q=ticker+' price',
                                            from_param=startday,
                                            to=tod,
                                            language='en',
                                            sort_by='relevancy',
                                            page_size=30,
                                            page=1)
        result_list = []
        for news in all_articles['articles']:
            toks = news['publishedAt'].split('T')
            day = toks[0]
            d = datetime.strptime(day,'%Y-%m-%d')
            if news['title'] is None:
                news['title'] = " "
            if news['content'] is None:
                news['content'] = " "
            result = {'text': news['title'] + news['content'], 'datetime': d }
            result_list.append(result)
        if len(result_list) > 0:
            data = pd.DataFrame(result_list)
            data['stock'] = data['datetime'].apply(lambda x: ticker)
            print(data)
        else:
            data = None
        return data

    def load_price(self,tic,period):
        ticker = yf.Ticker(tic)
        hist = ticker.history(period=period)
        stock_data = hist.reset_index()
        stock_data['rise'] = stock_data[['Open','Close']].apply(lambda x: 1 if x['Close'] >= x['Open'] else 0, axis=1)
        #stock_data['month'] = stock_data['Date'].apply(lambda x: x.month)
        #stock_data['day'] = stock_data['Date'].apply(lambda x: x.day)
        stock_data['stock'] = stock_data['Date'].apply(lambda x: tic)
        return stock_data

    def training(self):
        period='1y'
        all = []
        price = []
        #for i in ['AAPL','FB','GOOG','AMZN','NFLX']:
        with open(self.list_loc, 'r') as fp:
            list = fp.read().splitlines()
            for i in list:
                df = self.download_news(i,30)
                all.append(df)
                df_tic = self.load_price(i,period)
                print(df_tic)
                price.append(df_tic)

        data = pd.concat(all, ignore_index=True)
        data['next-day'] = data['datetime'].apply(lambda x: x + timedelta(days=1) )
        #data['month'] = data['datetime'].apply(lambda x: x.month)
        #data['day'] = data['datetime'].apply(lambda x: x.day)

        data['text'] = data[['stock','text','next-day']].groupby(['next-day'])['text'].transform(lambda x: ','.join(x))
        text_data = data[['stock','text','next-day']].drop_duplicates().sort_values('next-day').reset_index()
        print(text_data)
        stock_data = pd.concat(price,ignore_index=True)
        print(stock_data)
        combined_df = pd.merge(left=text_data, right=stock_data, how='left', left_on=['stock','next-day'], right_on=['stock','Date']).dropna()
        print(combined_df)
        sf = tc.SFrame(combined_df)
        sf['label'] = sf.apply(lambda x: int(x['rise']))

        # Split the data into training and testing
        training_data, test_data = sf.random_split(0.8)

        # Create a model using higher max_iterations than default
        model = tc.text_classifier.create(training_data, 'label', features=['text'], max_iterations=100)
        # Save the model for later use in Turi Create
        model.save(self.model_loc)
        # Save predictions to an SArray
        predictions = model.predict(test_data)
        predictions.explore()
        # Make evaluation the model
        metrics = model.evaluate(test_data)
        print(metrics['accuracy'])

    def predict(self):
        all = []
        with open(self.list_loc, 'r') as fp:
            list = fp.read().splitlines()
            for i in list:
                df = self.download_news(i,1)
                all.append(df)

        data = pd.concat(all, ignore_index=True)
        print(data)
        sf = tc.SFrame(data)

        model = tc.load_model(self.model_loc)
        # Save predictions to an SArray
        predictions = model.predict(sf)
        sf['prediction'] = predictions
        #sf.explore()
        trade_list = sf.groupby(key_column_names='stock', operations={'avg': agg.MEAN('prediction'), 'count': agg.COUNT()})
        #trade_list['label'] = trade_list.apply(lambda x: 'rise' if (x['avg'] >= 0.8 and x['count'] >= 10) else 'drop')
        self.shortlist = trade_list.to_dataframe()

    def getShortList(self):
        return self.shortlist

    def getModelLocation(self):
        return self.model_loc










