



import os 
import sys
import time
import json
import random 
import uuid
import h5py

import time 

from keras.optimizers import SGD

## pandas for dataframes manipulation 
import pandas as pd 
from datetime import datetime 
import dateparser
from clint.textui import puts , indent , colored 
## for protecting API keys  
from keys import pub_key , sec_key
## importing binance for assets_info fetching
from binance.client import *
from binance.enums import * 
##from Fancy_mails import contact_mail
import gc






from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout





##  for working with decimal numbers 
from decimal import Decimal

## importing tensorflow 
import tensorflow as tf

from tensorflow import keras


import matplotlib.dates as mdates

import matplotlib.pyplot as plt 

import matplotlib.transforms as mtransforms 
import numpy as np 




## coins to be traded   
 
     # ,'BTCZAR','BNBBUSD','BTCBUSD','BTCNGN', 'BCHBNB',  'XMRBNB'  ,  'XRPBNB' ,'ATOMBNB' , 'XRPBTC' , 'XMRBNB' , 'ZECBNB'

my_coins = ['BNBBTC']#,'BNBBUSD']#'BNBETH']#, 'BNBBUSD', 'BCHBNB',  'XMRBNB'  ,  'XRPBNB' ,'ATOMBNB' , 'XRPBTC' , 'XMRBNB' , 'ZECBNB']#,'ETHBTC' ,'BCHBTC']




## define the client to talk to binance server 
client = Client(pub_key , sec_key)


class_list = [] 

results_lst = [] 







#### define model layout 

#opt = keras.optimizers.Adam(lr=0.1, decay=1e-6)

model = Sequential()
model.add(LSTM(units = 100, return_sequences = True, input_shape = (None, 1)))
model.add(Dropout(0.2))

model.add(LSTM(units=100 , return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=1000 , return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=1000))
model.add(Dropout(0.2))
model.add(Dense(units=1 ))
model.compile(optimizer="adam" , loss='mean_squared_error' , metrics=[tf.keras.metrics.RootMeanSquaredError()])

### this the model used for training the data 
### the first 400 data points inside the dataframe goes to training and the last 100 for testing 























def graph_df(df):
    ''' this function graphs a simple df with out any facy maths '''
    time_stamps = list(df.date_time)
    open_postions = list(df.open)
    high_postions = list(df.high)
    low_postions = list(df.low) 
    close_postions = list(df.close)

#    print (len(time_stamps), len(open_postions), len(high_postions), len(low_postions), len(close_postions))

   # print (open_postions)

    # open is blue
    # high is green 
    # low is red
    # close is yellow
    plt.rcParams['axes.facecolor']='black'


    plt.matplotlib.pyplot.title("BNBBTC")
    plt.plot(time_stamps , open_postions, color='blue')
    plt.plot(time_stamps , high_postions, color='green')
    plt.plot(time_stamps , low_postions, color='red')
    plt.plot(time_stamps , close_postions, color='yellow')

    plt.show()

    return



#######################              define class               ########################


class fetch_coin:



    def __init__ (self, coin ):
        ''' init an object that holds all coin info as values any value can be cald using 'print (self.value_name)' '''
        coin_info = client.get_symbol_info(coin)
        


        self.balance = client.get_asset_balance(asset=coin[:3])

            ## filters used by binance 


        #print (coin_info)

  #      print (type(coin_info))
        sub_dict3 = (coin_info['filters'])[3]    
        sub_dict2 = (coin_info['filters'])[2]
        sub_dict  = (coin_info['filters'])[0]


     


        

        self.coin = coin 
        ## avg price over 5 mis period
        self.coin_price = client.get_symbol_ticker(symbol=coin)['price']
        ## get local time for your system 
        self.current_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(float(time.time())))





    ## important info about asset trading restrictions 
        ## MIN_NOT is the order value 
        ## for example : "if MIN_QTY * buy/sell price < MIN_NOT " 
        #  then the order is invalied  
        self.MIN_NOT = float(sub_dict3['minNotional'])

 #       print (sub_dict3)



        self.MAX_QTY = float(sub_dict2['maxQty'])
        self.MIN_QTY = float(sub_dict2['minQty'])
        ## precision is  the coin final_format if the len(coin_price) == precision or not 
        self.precision = int(coin_info['baseAssetPrecision'])
        self.precision2 = int(coin_info['quotePrecision'])

        self.model = model 



    ################################################################
    ################################################################

    def check_balance(self):
        ''' a simple method to check base asset balance '''

        if str(self.balance) == "None":
            print ("       NO FUNDS LEFT !!!! ")
        else:
            print ( "  YOU HAVE " + str(self.balance['free']) +" "+ str(self.balance['asset']))

        try:

            return str(self.balance['free'])
        
        except: 
            pass
        

    ################################################################
    ################################################################

    ## a method to check coin_pair performance over 1min  
 
    def check_performance(self):
        ''' a simple method to check for performance with time.sleep(180)'''

        coin__price = client.get_symbol_ticker(symbol=self.coin)
        
        
        time.sleep(180)

        coin__price2 = client.get_symbol_ticker(symbol=self.coin)

        ###  coin price differnce ratio over 1 min time interval 
        print ("P1 -> " + str(coin__price['price']))
        print ("P2 -> " + str(coin__price2['price']))
        ### the ratio between the asset prices
       # print ("price2 / price " + str(float(coin__price2['price'])/float(coin__price['price'])))


    ###  ( P2 - P1 / P1 ) * 100 <--- change in price over 1 min 


        deff = float(coin__price2['price']) - float (coin__price['price'])

        ratio = (float(deff) / float (coin__price['price']) )*100

        print ( "change over one min   "+ str("{:.3f}".format(ratio)))


        self.ratio = "{:.3f}".format(ratio)

        print (self.ratio)

    ################################################################
    ################################################################


    def test_order(self , order_type , predicted_price  ):
        ''' a test_order method to check for order approval 
        if this method returns an empty dictionary then order is valied '''
  
        print ("Creating a test with order "+order_type+" type")

     #   print (int(self.precision))
      #  print (int(self.precision2))

    #    print (str(float(self.coin_price)), " <-#-#-#-#-#-#-#")

        amt = float(self.coin_price)*0.003

        target_price = float(self.coin_price) - amt

    #    print (str(float(target_price)))

        tmp = str(self.coin_price).replace("." , " ")

        tmp = tmp.split()

     #  print (str(self.MIN_QTY) , "   this is MIN_QTY value !!!!! ")
     #   print (str(self.MAX_QTY) , "   this is MAX_QTY value !!!!! ")
     #   print (str(self.MIN_NOT) , "   this is MIN_NOT value !!!!! ")
      #  quant =  self.MIN_QTY *  10

        

        quant = self.MIN_QTY * 10 
        



    #    print (str(quant) +" <---------- trade amt ")


    #    print ( len(str(float(self.coin_price)) ),  len (str(target_price)))

      #  print ( " --------------> " , target_price)

        
                                                        ##########################
        target_price = "{:0.0{}f}".format(target_price ,len(tmp[1].rstrip("0"))  )
                                                        ##########################

     #   print ( " --------------> "  , target_price)

        print( "min_not = price * quant " , float(target_price) * float(quant))

       # print (predicted_price)


     #   print (quant)
        ## choosing amount to trade with 
       # if float(target_price) * float(quant) < self.MIN_NOT:
       #     quant = quant * 10 
     #   print (quant)






        test_order = client.create_test_order(
            symbol=str(self.coin),
            side=order_type,
            type="LIMIT",   
            timeInForce="GTC",
            quantity=quant,
            price=predicted_price
        )

        print (test_order)
        return quant

        #    target_price = target_price.rstrip("0").rstrip(".") if '.' is target_price else target_price

        ##### last thing to add to this method is a try/except statment to check if min_notational is trigged 


    ################################################################
    ################################################################


    def forecast_coin(self):
        ''' this method reads the market depth and prints out then average ask price and the average bid price
        then we get to the data that is used for the bot to train with 500 datapoints which gets split 
        into 400 for training and 100 for evalutation for the model, and this method adds the 5_period_sma , 10_period_sma 
        for data used ,also this method can output a graph of the data '''





        market_depth = client.get_order_book(symbol=str(self.coin))
        asks_list = list (market_depth['asks'])
        bids_list = list (market_depth['bids'])
        
        ask_sum = 0
        bid_sum = 0 


        for item in asks_list:
            ask_sum += float(item[0])
        for item in bids_list:
            bid_sum += float(item[0])
        
        self.avg_ask_price = (round(float(ask_sum)/ len (asks_list),7))
        self.avg_bid_price = ((round(float(bid_sum)/ len (bids_list),7)))




        #### max allowed by binance is 500 data-points 
        #### but you can change the "KLINE_INTERVAL_1MINUTE" check binance docs 
        #### https://python-binance.readthedocs.io/en/latest/binance.html
        candles = client.get_klines(symbol=str(self.coin),  interval=client.KLINE_INTERVAL_1MINUTE , limit=210)
        
        date_time = []
  
    
        open_lst = []
        high_lst = []
        low_lst = []
        close_lst = [] 
        volume_lst = []
        for item in candles:
            #print (item)
            t_time = float(item[0])/1000
            #print (t_time)
            dt_obj = datetime.fromtimestamp(t_time)

           # date_time.append(t_time)

            date_time.append(datetime.fromtimestamp(t_time))
            open_lst.append(float(item[1]))
            high_lst.append(float(item[2]))
            low_lst.append(float(item[3]))
            close_lst.append(float(item[4]))
            volume_lst.append(float(item[5]))
       
            
        ## creating data frame 
        coin_data_frame = {
            'date_time' : date_time,
            'open'  : open_lst,
            'high'  : high_lst,
            'low'   : low_lst,
            'close' : close_lst,
            'volume': volume_lst,
        }
        df = pd.DataFrame(coin_data_frame , columns = [ 'date_time' , 'open' , 'high' , 'low' , 'close','volume' ])
##########################################
       # df = df.tail(210)#################
##########################################




        rolling_mean = df['close'].rolling(window=5,  center=True ).mean()
        rolling_mean2 = df['close'].rolling(window=10, center=True).mean()
     

        df['5_sma'] = rolling_mean
        df['10_sma'] = rolling_mean2
      

        print("Number of questions: ", df.shape[0])

       

        plt.rcParams['axes.facecolor'] = 'grey'
        
        plt.grid(color='black')
        plt.title(label=self.coin)
        plt.matplotlib.pyplot.title(self.coin)
        plt.plot(date_time , open_lst, color='green' , label='open_prices')
        plt.plot(date_time, close_lst , color='red', label='close_prices')
        plt.plot( date_time, rolling_mean ,color='blue' , label='simple moving average window of 5')
        plt.plot( date_time,  rolling_mean2 ,color='yellow' , label='simple moving average window of 10')
        plt.legend(loc='lower right')
        plt.xticks(rotation=30)
        plt.ylabel("Price")
        plt.xlabel("Time")
        plt.savefig( self.coin+'_performance.png', dpi=200 )
        #plt.show()
        plt.clf()
        self.df = df



  
    ################################################################
    ################################################################

    def check_orders(self):
        ''' check order for current base coin '''

        ## get a list of open orders for current selected coin 

        open_orders_lst  = client.get_open_orders(symbol=self.coin)

        return (len(open_orders_lst))





    ################################################################
    ################################################################

    def make_data(self ):
        ''' here we take the data-frame from forcast_coin and we split the data for trainin , testing 
        then we fit the model with the training data this is a regression problem so we expect to output a number 
        as the prdicted price , we normalize data using MinMaxScaler with a feature range of (0,1) remember that 
        LSTM network take 3D array as the input , default batchs are 50 , epochs are 10 '''

        if not os.path.exists(str(self.coin)):
            os.makedirs(str(self.coin))


        checkpoint_filepath =str(self.coin)+"/check_point"
        weights_checkpoint = str(self.coin)


        checkpoint_dir = os.path.dirname(checkpoint_filepath)

        # a callback function to save model progress 

        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_weights_only=True,
           # monitor=[tf.keras.metrics.RootMeanSquaredError()],
            mode='max',
            save_best_only=True,
            verbose=1)



        dataset_train = self.df.head(150)
        training_set = dataset_train.iloc[:, 1:2].values

        print (dataset_train.tail(5)) 

        ## define the scaler for data normalization 
        sc = MinMaxScaler(feature_range=(0,1))
        training_set_scaled = sc.fit_transform(training_set)

        #print (self.df.tail(100))
      #  print (training_set_scaled)

        X_train = []
        y_train = []
        for i in range(10, 150):
            X_train.append(training_set_scaled[i-10:i, 0])
            y_train.append(training_set_scaled[i, 0])
        X_train, y_train = np.array(X_train), np.array(y_train)
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))


        ST = time.time() 

        try:
            model.load_weights(checkpoint_filepath)
            #model.compile(optimizer=opt ,  loss='mean_squared_error' , metrics=[tf.keras.metrics.RootMeanSquaredError()])
            print ("Weights loaded successfully $$$$$$$ ")
        except:
            print ("No Weights Found !!! ")



        model.fit(X_train,y_train,epochs=50,batch_size=10, callbacks=[model_checkpoint_callback])

        #model.summary()

        ### saving model conf and weights 

        #try:
           # model.save("model/")
        model.save_weights(filepath=checkpoint_filepath)
        model.save(checkpoint_dir+"/model")

        print ("Saving weights and model done   : #### #### #### #### #### #### #### #### #### ####")

        #except OSError as no_model:
        #    print ("Error saving weights and model !!!!!!!!!!!! ")

        
        x_time = round((time.time() - ST), 2)
        print ("_______________________________________________________")
        print ("Learning time = = "+str(x_time))


        self.model = model 


        

    ################################################################
    ################################################################


    def predict_symbol(self, balance):
        ''' this method takes the testing data from the (make_data) method and tests the model with 100 data-points 
        and the outputs the predicted price, you can also view the output data with matplotlib kust uncomment 
        the plt lines below ''' 
  
        model = self.model
      #  dataset_train = pd.DataFrame(self.df.tail(300))
        dataset_train = self.df.head(150)

        dataset_test  = self.df.tail(50)

        real_stock_price = dataset_test .iloc[:, 1:2].values


        #print (len(dataset_train),len(dataset_test))

       # training_set_scaled = sc.fit_transform(training_set)

        dataset_total = pd.concat((dataset_train['open'], dataset_test['open']), axis = 0)

        inputs = dataset_total[len(dataset_total) - len(dataset_test)  : ].values

        print ( len(inputs))
        
        inputs = inputs.reshape(-1,1)

      

        sc = MinMaxScaler(feature_range=(0,1))

        inputs = sc.fit_transform(inputs)

        X_test = []
        y_test = []
        for i in range(10, 50):
            X_test.append(inputs[i-10:i, 0])
            y_test.append(inputs[i , 0 ])
        X_test = np.array(X_test)
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

        predicted_stock_price = model.predict(X_test )
        predicted_stock_price = sc.inverse_transform(predicted_stock_price)


        #print (predicted_stock_price)
        #print (type(predicted_stock_price))


        predicted_stock_price_lst = predicted_stock_price.tolist()


        date_time_lst = list(self.df['date_time'])
        stock_price_lst =  list(self.df['open'])
        plt.grid(color='black')
        plt.title(label=self.coin)
        plt.rcParams['axes.facecolor'] = 'grey'
        plt.plot(date_time_lst , stock_price_lst , label='stock_prices' , color='red')
        plt.plot(date_time_lst[170:] , predicted_stock_price_lst, label="predicted_prices" , color="blue")
        plt.legend(loc='upper left')
        plt.xticks(rotation=30)
        plt.savefig(self.coin+"_prediction_stock.png", dpi=400 )
       # plt.show()
        plt.clf()



        coin_ticker = client.get_symbol_ticker(symbol=self.coin)['price']

        t_price =  (predicted_stock_price_lst[len(predicted_stock_price_lst)-1])


        tmp = str(coin_ticker).replace("." , " ")
        tmp = tmp.split()

        c = len(coin_ticker)
     
      #  t_price = "{:0.0{}f}".format(round(t_price[0] , self.precision) , len(tmp[1])  )

        t_price = round(t_price[0] , 7 )
        print ("Current_coin_price :> "+str(float(coin_ticker)))
        print ("Predicted_coin_price :> " +str(float(t_price)))


        

        plt.matplotlib.pyplot.title(self.coin)
        plt.grid(color='cyan')
        
        print(len(date_time_lst),  len(predicted_stock_price_lst))
       
        
        
        plt.plot(date_time_lst[170:],  predicted_stock_price_lst, color='blue'  , label='predicted coin price' )
        plt.title(label=self.coin)
        plt.plot(date_time_lst,  stock_price_lst, color='red', label='stock coin price ') 
        #plt.plot(x, g, color='red')
        plt.ylabel("Price")
        plt.xlabel("Time")
        plt.legend(loc='lower right')
        plt.xticks(rotation=30)
        plt.savefig(self.coin+'_predicted_graph.png', dpi=200 )
        #plt.show()
        plt.clf()


        print (len(X_test))

        result = sc.inverse_transform(X_test[len(X_test)-1])

     

        # Evaluate the model
        loss, loss_percentage = model.evaluate(X_test, y_test , verbose=2)
        print ("*****************************************************")

        print("Untrained model, LOSS_PERCENTAGE: {:5.2f}%".format(100*loss_percentage))
        print ("*****************************************************")


        print ("^^^^^^^^^^^^^^^^^^^^^^^")
        LOSS_PERCENT = "{:5.2f}".format(100*loss_percentage)

        print ("#<><><><><><><><><><><><><><><><><><><>#")
        print ("avg_ask price : "+str(self.avg_ask_price))
        print ("avg_bid price : " +str(self.avg_bid_price))
        time.sleep(15)
        results_lst.append([coin_ticker, t_price])

        

        return coin_ticker , t_price , LOSS_PERCENT



       # return coin_ticker, t_price 


    ### this part prints the technical analysis for the current coin

    #    technical_selenium(COIN=self.coin)


    ################################################################
    ################################################################


    def market_BUY(self ):
        ''' the bot executes a BUY with the market price at that moment '''
        buy_order = client.create_order(symbol=self.coin ,type="MARKET" , quantity="0.2", side="BUY" )
        




        pass
    
    ################################################################
    ################################################################


    def market_SELL(self):
        ''' the bot executes a SELL with the market price at that moment '''
        buy_order = client.create_order(symbol=self.coin ,type="MARKET" , quantity="0.2", side="SELL")

        
        pass












    ################################################################
    ################################################################

    def limit_sell_order(self , sell_price):
        ''' a method to create a limit order of type sell this order will be open until the 
        current coin price reaches the limit then the order is excuted ''' 


        quant =  self.MIN_QTY *  int (random.randrange(10, 20)) *10

        order = client.order_limit_sell(
            symbol=self.coin,
            quantity="0.15",
            price=sell_price)

        return order 




    ################################################################
    ################################################################

    def limit_buy_order(self , buy_price):
        ''' a method to create a limit order of type buy this order will be open until the 
        current coin price reaches the limit then the order is excuted '''
        quant =  self.MIN_QTY *  int (random.randrange(10, 20)) *10

        order = client.order_limit_buy(
            symbol=self.coin,
            quantity="0.15",
            price=buy_price)

        return order



    ################################################################
    ################################################################

    def cancel_order_by_id(self , order_id ):

        ''' a method to cancel order if the current price is too far away from the order buy/sell price ''' 



        pass




    def lst_orders(self):
        orders = client.get_open_orders(symbol=self.coin)
        for item in orders:
            #print (item)
            pass


        return orders





    def pass_to_gui(self ):
        coin_ticker_ = client.get_symbol_ticker(symbol=self.coin)

        pwd = os.getcwd()

        tmp_dict = {}
        data = {}
        data['tickers'] = coin_ticker_

        tmp_dict2 = {}
        for n , item in enumerate(self.lst_orders()):

            print (item['symbol'] ,  item['orderId'],  item['price'],  item['origQty'], item['executedQty'], item['status'] , item['time'] , item['updateTime'])

            tmp_dict2[item['orderId']] =  [item['symbol'] ,  item['orderId'],  item['price'],  item['origQty'], item['executedQty'], item['status'] , item['time'] , item['updateTime']]

        data['open_orders'] = tmp_dict2

        os.system("cd web-gui")



        with open(self.coin+'orders.json', 'w') as outfile:
            json.dump(data, outfile)












#   n3w_obj = fetch_coin("BNBBTC")
#   n3w_obj.forecast_coin()
#   n3w_obj.make_data()
#   balance = n3w_obj.check_balance() 
#   n3w_obj.predict_symbol(balance=balance)

#try:
#for item in my_coins:

#    n3w_coin = fetch_coin(coin=item)
    #print (str(n3w_coin.coin) +"  "+ str(n3w_coin.coin_price))
#    balance_ = n3w_coin.check_balance()


    
#    n3w_coin.forecast_coin()
    
    
#    n3w_coin.predict_symbol(balance=balance_)
    
    
#    n3w_coin.pass_to_gui()

#except:
#    print ("error.......")



#while True:
#for item in my_coins:

 #   obj = fetch_coin(coin=item)
  #  obj.forecast_coin()
#    obj.make_data()
#    balance_ = obj.check_balance()
#    obj.predict_symbol(balance=balance_)
#    time.sleep(15)
#    obj.pass_to_gui()