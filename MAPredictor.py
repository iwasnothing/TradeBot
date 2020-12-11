import yfinance as yf
import math
import turicreate as tc
import pandas as pd
import requests
import ssl

class MAPredictor:
    def __init__(self,list="/app/list.txt"):
        requests.packages.urllib3.disable_warnings()
        try:
            _create_unverified_https_context = ssl._create_unverified_context
        except AttributeError:
            # Legacy Python that doesn't verify HTTPS certificates by default
            pass
        else:
            # Handle target environment that doesn't support HTTPS verification
            ssl._create_default_https_context = _create_unverified_https_context
        self.list_loc = list


    def load_price(self,tic,period):
        ticker = yf.Ticker(tic)
        hist = ticker.history(period=period)
        print(hist)
        stock_data = hist.reset_index()
        stock_data['rise'] = stock_data[['Open','Close']].apply(lambda x: 1 if x['Close'] >= x['Open'] else 0, axis=1).astype('int32')
        stock_data['hi-lo-spread'] = stock_data[['High','Low']].apply(lambda x: x['High'] - x['Low'] , axis=1)
        stock_data['log-vol'] = stock_data['Volume'].apply(lambda x: math.log(x+1))
        return stock_data

    def rolling_avg(self,df,win):
        return df['Close']-df['Close'].rolling(win).mean()

    def train_evaluate(self,tic):
        df = self.load_price(tic,'4y')
        df = df.dropna()
        df['return'] = df['Close'].diff()
        df['log-vol-diff'] = df['log-vol'].diff()
        for x in range(10,40,10):
            df['gt-' + str(x) + '-avg'] = self.rolling_avg(df,x)

        df['label'] = df['rise'].shift(-1)
        cols = ['hi-lo-spread','return','log-vol-diff','gt-10-avg','gt-20-avg','gt-30-avg','label']
        print(df[cols])
        last = df.tail(1)
        last = tc.SFrame(last[cols[:-1]])
        dataset = df[cols].dropna().reset_index(drop=True)
        dataset['label'] = dataset['label'].astype('int32').apply(lambda x: "rise" if x == 1 else "drop")
        data = dataset.sample(frac=0.8, random_state=786)
        data_unseen = dataset.drop(data.index)
        data.reset_index(inplace=True, drop=True)
        data_unseen.reset_index(inplace=True, drop=True)
        print('Data for Modeling: ' + str(data.shape))
        print('Unseen Data For Predictions: ' + str(data_unseen.shape))
        sf = tc.SFrame(data)  # Make a train-test split
        train_data, test_data = sf.random_split(0.8)
        # Automatically picks the right model based on your data.
        model = tc.classifier.create(train_data, target='label',
                                    features=cols[:-1])
        # Save predictions to an SArray
        results = model.evaluate(test_data)
        prediction = model.predict(last)
        print(prediction)
        results['prediction'] = prediction[0]
        return results

    def trainAll(self):
        result_list = []
        with open(self.list_loc, 'r') as fp:
            list = fp.read().splitlines()
            for i in list:
        #for i in ['AAPL', 'FB', 'GOOG', 'AMZN', 'NFLX', "SQ", "MTCH", "AYX", "ROKU", "TTD"]:
                results = self.train_evaluate(i)
                result_list.append({"stock":i,"accuracy":results['accuracy'],'prediction':results['prediction']})

        #print(result_list)
        df = pd.DataFrame(result_list).sort_values('accuracy',ascending=False)
        print(df)
        print(df.info())
        self.shortlist = df

    def getShortList(self):
        return self.shortlist
