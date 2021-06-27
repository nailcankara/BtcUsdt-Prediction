import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import warnings 
from binance.client import Client
from tslearn.clustering import KShape
import pickle
from datetime import datetime
from sklearn.preprocessing import StandardScaler
import random
warnings.filterwarnings('ignore')

ksModel = KShape(n_clusters=100, n_init=1, random_state=0)
ksModel = ksModel.from_pickle("ksModel.pkl")

infile = open('arrayClusters.pkl','rb')
arrayClusters = pickle.load(infile)
infile.close()

infile = open('CloseListPred.pkl','rb')
CloseListPred = pickle.load(infile)
infile.close()


chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz1234567890"
api_key = "".join(random.choice(chars) for i in range(64))
api_secret = "".join(random.choice(chars) for i in range(64))
kripto = "BTCUSDT"

columns = ["Date","Open","High","Low","Close","VolXmr","gereksiz1","Volume","Islem","Hacim1","Hacim2","gereksiz5"]
after_columns = ["Open","High","Low","Close","VolBtc"]
gereksiz = ["VolXmr","gereksiz1","Islem","Hacim1","Hacim2","gereksiz5"]

client = Client(api_key, api_secret , {"verify": True, "timeout": 20})
kriptoVerileri = client.get_historical_klines(kripto, Client.KLINE_INTERVAL_5MINUTE, "6 hours ago UTC")    #veri çek
kriptoVerileri = pd.DataFrame(kriptoVerileri , columns=columns ).drop(columns=gereksiz)
kriptoVerileri.Date = (kriptoVerileri.Date/1000).apply(datetime.fromtimestamp)
kriptoVerileri = kriptoVerileri.astype(float , errors = 'ignore')
kriptoVerileriValue = kriptoVerileri.Close.values


sc = StandardScaler()
guncel = sc.fit_transform(kriptoVerileriValue.reshape(-1,1)).reshape(1,72,1)

sonuc = ksModel.predict(guncel)[0]
filteredArrays = arrayClusters[sonuc]
targetArrays = CloseListPred[filteredArrays]


ksModel2 = KShape(n_clusters=3, n_init=1, random_state=0).fit(targetArrays)

labels = np.unique(ksModel2.labels_, return_counts=True)[1] / np.unique(ksModel2.labels_, return_counts=True)[1].sum()
predDraws = ksModel2.cluster_centers_.reshape(3,-1)


beforeTrans0 = np.array(list(list(predDraws[0]))
beforeTrans1 = np.array(list(list(predDraws[1]))
beforeTrans2 = np.array(list(list(predDraws[2]))
predFinal0 = sc.inverse_transform(beforeTrans0)
predFinal1 = sc.inverse_transform(beforeTrans1)
predFinal2 = sc.inverse_transform(beforeTrans2)


dataFinal= pd.DataFrame((predFinal0,predFinal1,predFinal2))
#fark = kriptoVerileri.Close.iloc[-2] - dataFinal.iloc[:,0]
#fark = labels[0]*fark[0] + labels[1]*fark[1] + labels[2]*fark[2]


#dataFinal.iloc[0,:] = dataFinal.iloc[0,:]+fark
#dataFinal.iloc[1,:] = dataFinal.iloc[1,:]+fark
#dataFinal.iloc[2,:] = dataFinal.iloc[2,:]+fark


st.title('BTC-USDT PRICE PREDICTION')
st.button("Refresh")

fig = make_subplots()


figCandle = go.Candlestick(x=kriptoVerileri['Date'],
                     open=kriptoVerileri['Open'],
                     high=kriptoVerileri['High'],
                     low=kriptoVerileri['Low'],
                     close=kriptoVerileri['Close'])



forLine0 = pd.DataFrame(dataFinal.iloc[0,:].values , columns=["Close"])
forLine1 = pd.DataFrame(dataFinal.iloc[1,:].values , columns=["Close"])
forLine2 = pd.DataFrame(dataFinal.iloc[2,:].values , columns=["Close"])

timestamp = datetime.timestamp(kriptoVerileri.Date.iloc[0])
timeRange = [datetime.fromtimestamp(timestamp + 300*i) for i in range(1,85)]

forLine0["Date"] = timeRange
forLine1["Date"] = timeRange
forLine2["Date"] = timeRange

figLine0 = go.Scatter(x=forLine0["Date"] , y=forLine0["Close"] , name='%'+str(round(labels[0]*100)) + " Probability")
figLine1 = go.Scatter(x=forLine1["Date"] , y=forLine1["Close"] , name='%'+str(round(labels[1]*100)) + " Probability")
figLine2 = go.Scatter(x=forLine2["Date"] , y=forLine2["Close"] , name='%'+str(round(labels[2]*100)) + " Probability")


fig.add_trace(figCandle)

fig.add_trace(figLine0)
fig.add_trace(figLine1)
fig.add_trace(figLine2)


fig.update_layout(xaxis_rangeslider_visible=False)
st.write(fig)

st.subheader("For any question Twitter : @deepgrad")

