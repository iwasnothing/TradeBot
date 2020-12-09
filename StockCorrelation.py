import matplotlib.pyplot as plt
import yfinance as yf
import requests
import ssl
import numpy as np
import pandas as pd
import turicreate as tc
from sklearn.preprocessing import MinMaxScaler
import os
import time
import numpy as np
from numpy import genfromtxt


class StockCorrelation:
    def __init__(self):
        requests.packages.urllib3.disable_warnings()
        try:
            _create_unverified_https_context = ssl._create_unverified_context
        except AttributeError:
            # Legacy Python that doesn't verify HTTPS certificates by default
            pass
        else:
            # Handle target environment that doesn't support HTTPS verification
            ssl._create_default_https_context = _create_unverified_https_context
        self.plot_chart = False

    def accuracy(self,x_valid,rnn_forecast):
        a = []
        b = []
        n = min(len(x_valid),len(rnn_forecast))
        for i in range(1,n):
            a.append(x_valid[i]-x_valid[i-1])
            b.append(rnn_forecast[i]-rnn_forecast[i-1])
        a = np.array([1 if x > 0 else -1 for x in a])
        b = np.array([1 if x > 0 else -1 for x in b])
        c = a*b
        acc = sum ([ 1 if x > 0 else 0 for x in c])
        acc = float(acc) / n

        return acc

    def load_data(self,tic,period):
        n = 0
        while n == 0:
            ticker = yf.Ticker(tic)
            hist = ticker.history(period=period)
            n = hist.shape[0]
            time.sleep(1)
        return hist

    def get_data_x(self,tickX, period,win):
        #tickerX = yf.Ticker(tickX)
        #histX = tickerX.history(period=period)
        histX = self.load_data(tickX,period)
        df = pd.DataFrame(histX['Close'])
        for i in range(win , 3 * win ):
            df['lag-' + str(i)] = df['Close'].shift(i)
        df = df.drop(columns=['Close'])
        return df.dropna()

    def get_data_y(self,tickY, period, win):
        #tickerY = yf.Ticker(tickY)
        #histY = tickerY.history(period=period)
        histY = self.load_data(tickY,period)
        df = pd.DataFrame(histY['Close'])
        df['avg'] = df['Close'].rolling(win).mean()
        df = df.drop(columns=['Close'])
        return df.dropna()

    def pair_loss(self,ticker1,ticker2,period,win):
        #win = 5
        #period = '2y'
        #ticker1 = 'AAPL'
        #ticker2 = 'FB'
        scalerX = MinMaxScaler()
        scalerY = MinMaxScaler()
        x = self.get_data_x(ticker1,period,win)
        y = self.get_data_y(ticker2,period,win)
        #print(x)
        #print(y)
        #print(y.values)
        x = pd.DataFrame(scalerX.fit_transform(x.values), columns=x.columns, index=x.index)
        y = pd.DataFrame(scalerY.fit_transform(y.values), columns=y.columns, index=y.index)
        df = x.merge(y, on='Date').dropna()
        data = tc.SFrame(df)# Make a train-test split
        train_data, test_data = data.random_split(0.8)
        myFeatures = []
        for i in range(win,3*win):
            myFeatures.append('lag-' + str(i))
        # Automatically picks the right model based on your data.
        model = tc.regression.create(train_data, target='avg',
                                        features = myFeatures)
        # Save predictions to an SArray
        results = model.evaluate(test_data)
        predictions = model.predict(data)
        chart = tc.SFrame(predictions)
        chart['actual'] = data['avg']
        # Evaluate the model and save the results into a dictionary
        acc = self.accuracy(chart['actual'], chart['X1'])
        print(acc)
        results['accuracy'] = acc
        #tickerX = yf.Ticker(ticker1)
        #hist = tickerX.history(period='1mo')
        #price = hist['Close'].values
        #current = tc.SFrame(price[:2*win])
        hist = self.load_data(ticker1,period)
        price = hist['Close'].values
        current = tc.SFrame(price[-2*win:])
        future = model.predict(current)
        future = np.expand_dims(future, axis=0)
        print(future)
        future = scalerY.inverse_transform(future)
        print(future)
        results['future'] = future[0][0]
        results['ticker1'] = ticker1
        results['ticker2'] = ticker2
        #if plot_chart == True:
        if self.plot_chart == True and results['rmse'] <= 0.06:
            #print(chart)
            plt.clf()
            plt.plot(range(len(chart['actual'])), chart['X1'], "g-")
            plt.plot(range(len(chart['actual'])), chart['actual'], "y-")
            plt.xlabel("Time")
            plt.ylabel("Value")
            plt.grid(True)
            #plt.show()
            plt.savefig(ticker2 + "-prediction.png")
            #chart.show()
        return results

    def del_list_files(self,list,period):
        for tic in list:
            filename = tic + "-" + period + ".csv"
            os.remove(filename)

    def getAllCorrelation(self):
        #final_table = []
        period = '2y'
        win = 5
        with open('list.txt','r') as fp:
            list = fp.read().splitlines()
            tickerMap = {}
            for idx,ticker in enumerate(list):
                tickerMap[ticker] = idx
            n = len(list)
            rmse_matrix = np.zeros((n,n))
            acc_matrix = np.zeros((n,n))
            pred_matrix = np.zeros((n,n))
            print(list)
            for ticker1 in list:
                for ticker2 in list:
                    if ticker2 != ticker1:
                        result = self.pair_loss(ticker1, ticker2, period, win)
                        #print(result)
                        row = {"ticker1": ticker1, "ticker2": ticker2 ,
                               "rmse": result['rmse'], "future":result['future'], "accuracy":result['accuracy']}
                        x = tickerMap[ticker1]
                        y = tickerMap[ticker2]
                        rmse_matrix[x,y] = row['rmse']
                        acc_matrix[x,y] = row['accuracy']
                        pred_matrix[x,y] = row['future']
            np.savetxt('rmse.csv',rmse_matrix,delimiter=',')
            np.savetxt('acc.csv',acc_matrix,delimiter=',')
            np.savetxt('pred.csv',pred_matrix,delimiter=',')
            self.del_list_files(list,period)
            ind = np.unravel_index(np.argmax(acc_matrix, axis=None), acc_matrix.shape)
            print(ind)
            print(list[ind[0]],list[ind[1]])
            print(acc_matrix[ind])
            n = acc_matrix.shape[0]
            m = acc_matrix.shape[1]
            for i in range(n):
                for j in range(m):
                    if acc_matrix[i,j] >= 0.7:
                        print(i,j,list[i],list[j],rmse_matrix[i,j],acc_matrix[i,j], pred_matrix[i,j])

            n = rmse_matrix.shape[0]
            for i in range(n):
                rmse_matrix[i,i] = 1000
            ind = np.unravel_index(np.argmin(rmse_matrix, axis=None), rmse_matrix.shape)
            print(ind)
            print(list[ind[0]],list[ind[1]])
            print(rmse_matrix[ind])

            n = rmse_matrix.shape[0]
            m = rmse_matrix.shape[1]
            for i in range(n):
                for j in range(m):
                    if rmse_matrix[i,j] <= 0.04 and acc_matrix[i,j] > 0.6:
                        print(list[i],list[j],rmse_matrix[i,j],acc_matrix[i,j],pred_matrix[i,j])

