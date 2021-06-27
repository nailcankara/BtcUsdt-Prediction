import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import warnings 
from binance.client import Client
warnings.filterwarnings('ignore')


#%%
api_key = "5LkWukqCVPCjESnHdhODvZEcnJHDVm0u98Q5ydMkhsuhN9wMltfHuFUnNTmf8Zpq"
api_secret = "tCQsOLG5LsFKHReoRiSHL0gXpRCDXdaYWPQRks7FeiSYDEd7pre2QLJNXRCg2iJB"
kripto = "BTCUSDT"
columns = ["Date","Open","High","Low","Close","VolXmr","gereksiz1","Volume","Islem","Hacim1","Hacim2","gereksiz5"]
after_columns = ["Open","High","Low","Close","VolBtc"]
gereksiz = ["VolXmr","gereksiz1","Islem","Hacim1","Hacim2","gereksiz5"]

client = Client(api_key, api_secret , {"verify": True, "timeout": 20})
kriptoVerileri = client.get_historical_klines(kripto, Client.KLINE_INTERVAL_5MINUTE, "360 day ago UTC")    #veri çek
kriptoVerileri = pd.DataFrame(kriptoVerileri , columns=columns ).drop(columns=gereksiz)
kriptoVerileri.Date = (kriptoVerileri.Date/1000).apply(datetime.fromtimestamp)
kriptoVerileri = kriptoVerileri.set_index(kriptoVerileri.Date).drop(columns=["Date"])
kriptoVerileri = kriptoVerileri.astype(float)
kriptoVerileri = kriptoVerileri.iloc[:-1,:]


#%%

kriptoVerileri.to_csv("BTCUSDT.csv")

#%%

kriptoVerileri = pd.read_csv("BTCUSDT.csv")
kriptoVerileri = kriptoVerileri.set_index(kriptoVerileri.Date).drop(columns=["Date"])
kriptoVerileri = kriptoVerileri.astype(float)
Close = kriptoVerileri.Close


#%%
from sklearn.preprocessing import StandardScaler
winSize = 12*6
winSizePred = winSize + 12

CloseList = []
CloseListPred = []

for i in range(0,len(Close),12):

    values = Close.iloc[i:i+winSize].values.reshape(-1,1)
    valuesPred = Close.iloc[i:i+winSizePred].values.reshape(-1,1)

    if len(values) == winSize and len(valuesPred) == winSizePred:
        sc = StandardScaler()
        values = sc.fit_transform(values)
        valuesPred = sc.transform(valuesPred)
        
        CloseList.append(values)
        CloseListPred.append(valuesPred)
    else:
        pass

CloseList = np.array(CloseList)
CloseListPred = np.array(CloseListPred)

#%%
from tslearn.clustering import KShape
#ksModel = KShape(n_clusters=3, n_init=1, random_state=0).fit(CloseList)

#%%

ksModel = KShape(n_clusters=100, n_init=1, random_state=0).fit(CloseList)
preds = ksModel.predict(CloseList)
ksModel.to_pickle("ksModel.pkl")
print("Tamamlandı")

#%%
ksModel = KShape(n_clusters=100, n_init=1, random_state=0)
ksModel = ksModel.from_pickle("ksModel.pkl")

#%%
arrayClusters = {}

for i in range(100):
    arrayClusters[i] = np.where(preds==i)[0]

#%%
import pickle
filename = 'arrayClusters.pkl'
outfile = open(filename,'wb')
pickle.dump(arrayClusters,outfile)
outfile.close()

filename = 'CloseListPred.pkl'
outfile = open(filename,'wb')
pickle.dump(CloseListPred,outfile)
outfile.close()
#%%
infile = open('arrayClusters.pkl','rb')
arrayClusters = pickle.load(infile)
infile.close()

infile = open('CloseListPred.pkl','rb')
CloseListPred = pickle.load(infile)
infile.close()
#%%
xd = Close.iloc[1111+72:1111+72+12].values
pd.DataFrame(xd).plot()
#%%
deneme1 = Close.iloc[1111:1111+72].values.reshape(72,1)

sc = StandardScaler()
deneme1_1 = sc.fit_transform(deneme1).reshape(1,72,1)
#%%

deneme = deneme1_1#CloseList[1111].reshape(1,72,1)
sonuc = ksModel.predict(deneme)[0]
print(sonuc)

#%%

arrays = arrayClusters[sonuc]
targetArrays = CloseListPred[arrays]


#%%

clustSize = 3#round(np.sqrt(targetArrays.shape[0]))

ksModel2 = KShape(n_clusters=clustSize, n_init=1, random_state=0).fit(targetArrays)

#%%

for i in range(clustSize):
    #print(len(ksModel2.cluster_centers_[i]))
    pd.DataFrame(ksModel2.cluster_centers_[i]).plot()

#%%
labels = np.unique(ksModel2.labels_, return_counts=True)[1] / np.unique(ksModel2.labels_, return_counts=True)[1].sum()
predDraws = ksModel2.cluster_centers_[:,72:].reshape(3,12)
print(labels,predDraws)

#%%
beforeTrans0 = np.array(list(np.zeros(60)) + list(predDraws[0]))
beforeTrans1 = np.array(list(np.zeros(60)) + list(predDraws[1]))
beforeTrans2 = np.array(list(np.zeros(60)) + list(predDraws[2]))
predFinal0 = sc.inverse_transform(beforeTrans0)[-12:]
predFinal1 = sc.inverse_transform(beforeTrans1)[-12:]
predFinal2 = sc.inverse_transform(beforeTrans2)[-12:]

dataFinal= pd.DataFrame((predFinal0,predFinal1,predFinal2))
#%%


