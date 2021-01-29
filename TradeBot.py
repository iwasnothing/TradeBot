import alpaca_trade_api as tradeapi
import requests
import ssl
import numpy as np
import pandas as pd
import turicreate as tc
from sklearn.preprocessing import MinMaxScaler
import os
import time
import joblib

class TradeBot:
    def __init__(self,ticker1='ZM',ticker2='LBTYK',win=5,past=6,API_KEY=None,API_SECRET=None):
        requests.packages.urllib3.disable_warnings()
        try:
            _create_unverified_https_context = ssl._create_unverified_context
        except AttributeError:
            # Legacy Python that doesn't verify HTTPS certificates by default
            pass
        else:
            # Handle target environment that doesn't support HTTPS verification
            ssl._create_default_https_context = _create_unverified_https_context
        self.past = past
        self.ticker1 = ticker1
        self.ticker2 = ticker2
        self.win = win
        self.period = past * win
        self.API_KEY = API_KEY
        self.API_SECRET = API_SECRET

    def accuracy(self, x_valid, rnn_forecast):
        a = []
        b = []
        n = min(len(x_valid), len(rnn_forecast))
        for i in range(1, n):
            a.append(x_valid[i] - x_valid[i - 1])
            b.append(rnn_forecast[i] - rnn_forecast[i - 1])
        a = np.array([1 if x > 0 else -1 for x in a])
        b = np.array([1 if x > 0 else -1 for x in b])
        c = a * b
        acc = sum([1 if x > 0 else 0 for x in c])
        acc = float(acc) / n

        return acc

    def get_data_x(self, tickX):
        # tickerX = yf.Ticker(tickX)
        # histX = tickerX.history(period=period)
        histX = self.load_data(tickX, 30*self.period)
        df = pd.DataFrame(histX)
        for i in range(self.win, self.past * self.win):
            df['lag-' + str(i)] = df['Close'].shift(i)
        df = df.drop(columns=['Close'])
        return df.dropna()

    def get_data_y(self, tickY):
        # tickerY = yf.Ticker(tickY)
        # histY = tickerY.history(period=period)
        histY = self.load_data(tickY, 30*self.period)
        df = pd.DataFrame(histY)
        df['avg'] = df['Close'].rolling(self.win).mean()
        df = df.drop(columns=['Close'])
        return df.dropna()

    def pair_loss(self):
        # win = 5
        # period = '2y'
        # ticker1 = 'AAPL'
        # ticker2 = 'FB'
        scalerX = MinMaxScaler()
        scalerY = MinMaxScaler()
        x = self.get_data_x(self.ticker1)
        y = self.get_data_y(self.ticker2)
        #print(x)
        # print(y)
        # print(y.values)
        #print(x.values.shape)
        x = pd.DataFrame(scalerX.fit_transform(x.values), columns=x.columns, index=x.index)
        y = pd.DataFrame(scalerY.fit_transform(y.values), columns=y.columns, index=y.index)
        joblib.dump(scalerX, self.ticker1 + "-" + self.ticker2 + '-scalerX.gz')
        joblib.dump(scalerY, self.ticker1 + "-" + self.ticker2 + '-scalerY.gz')
        # print(x)
        # print(y)
        # df = x.merge(y, on='Date').dropna()
        df = x.merge(y, left_index=True, right_index=True).dropna()
        data = tc.SFrame(df)  # Make a train-test split
        train_data, test_data = data.random_split(0.8)
        myFeatures = []
        for i in range(self.win, self.past * self.win):
            myFeatures.append('lag-' + str(i))
        # Automatically picks the right model based on your data.
        model = tc.regression.create(train_data, target='avg',
                                     features=myFeatures)
        # Save predictions to an SArray
        results = model.evaluate(test_data)
        model.save(self.ticker1 + "-" + self.ticker2 + '-regression')
        predictions = model.predict(test_data)
        predictions = pd.DataFrame(predictions, columns=['X1']).rolling(self.win).mean()
        chart = tc.SFrame(predictions)
        chart['actual'] = test_data['avg']
        acc = self.accuracy(chart['actual'], chart['X1'])
        print(acc)
        results['accuracy'] = acc
        return results

    def predictPrice(self):
        scalerX = joblib.load(self.ticker1 + "-" + self.ticker2 + '-scalerX.gz')
        scalerY = joblib.load(self.ticker1 + "-" + self.ticker2 + '-scalerY.gz')
        myFeatures = []
        for i in range(self.win,self.past*self.win):
            myFeatures.append('lag-' + str(i))
        # Automatically picks the right model based on your data.
        model = tc.load_model(self.ticker1 + "-" + self.ticker2 + '-regression')
        hist = self.load_data(self.ticker1,self.period)
        current = np.array(hist.iloc[-1*(self.past-1)*self.win:].values).reshape(1,-1)
        #print(current)
        current = scalerX.transform(current)
        current = tc.SFrame(current)
        future = model.predict(current)
        #print(hist)
        price = hist['Close'].values
        current = tc.SFrame(price[-1*(self.past-1)*self.win:])
        future = model.predict(current)
        future = np.expand_dims(future, axis=0)
        print(future[0][0])
        future = scalerY.inverse_transform(future)
        print("predicted price after transform is {}".format(future[0][0]))
        return future[0][0]

    def load_data(self,ticker,period):
        APCA_API_BASE_URL = "https://paper-api.alpaca.markets"
        api = tradeapi.REST(self.API_KEY, self.API_SECRET, APCA_API_BASE_URL, 'v2')

        # Get daily price data for AAPL over the last 5 trading days.
        barset = api.get_barset(ticker, 'minute', limit=period)
        bars = barset[ticker]
        df = pd.DataFrame( [ b.c for b in bars], columns=['Close'] )
        #df['Date'] = range(len(bars))
        #df.set_index('Date')
        # See how much AAPL moved in that timeframe.
        #week_open = aapl_bars[0].o
        #week_close = aapl_bars[-1].c
        #percent_change = (week_close - week_open) / week_open * 100
        #print('AAPL moved {}% over the last 5 minutes'.format(percent_change))
        #print(df)
        return df

    def buy(self,symbol,price):
        APCA_API_BASE_URL = "https://paper-api.alpaca.markets"
        api = tradeapi.REST(self.API_KEY, self.API_SECRET, APCA_API_BASE_URL, 'v2')
        symbol_price = api.get_last_quote(symbol)
        symbol_price = symbol_price.askprice
        spread = (price - symbol_price) / symbol_price
        
        print("predicted spread is {}".format(spread))
        toBuy = False
        if spread < 0 :
            print("No profit")
            toBuy = False
            return
            #position = api.get_position(symbol)
        # Get a list of all of our positions.
        toBuy = True
        portfolio = api.list_positions()

        # Print the quantity of shares for each position.
        for position in portfolio:
            if position.symbol == symbol:
                toBuy = False
                print("current position is {}".format(position.qty))
                if (int(position.qty) <= 3 and spread > 0):
                    print("place order")
                    spread = 0.4
                    api.submit_order(
                        symbol=symbol,
                        qty=1,
                        side='buy',
                        type='market',
                        time_in_force='gtc',
                        order_class='bracket',
                        stop_loss={'stop_price': symbol_price * (1-spread),
                                   'limit_price': symbol_price * (1-spread)*0.95},
                        take_profit={'limit_price': symbol_price * (1+spread)}
                    )
        if toBuy == True:
            spread = 0.4
            api.submit_order(
                symbol=symbol,
                qty=1,
                side='buy',
                type='market',
                time_in_force='gtc',
                order_class='bracket',
                stop_loss={'stop_price': symbol_price * (1 - spread),
                           'limit_price': symbol_price * (1 - spread) * 0.95},
                take_profit={'limit_price': symbol_price * (1 + spread)}
            )

