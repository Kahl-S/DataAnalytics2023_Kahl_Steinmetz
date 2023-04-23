import aspects as asp
from sentence_transformers import SentenceTransformer
from datetime import datetime
import pandas as pd

es_uri = "https://localhost:8080/elasticsearch/"
embedding_type = 'sbert'

begin = datetime(2022,10,1)
end = datetime(2022,11,1)

date_range = [begin,end]
es_index = "ukraine-data-lite-mar23"
embedding_model = SentenceTransformer("all-MiniLM-L12-v2")
query = "US Economy"

runner = asp.run_query(es_uri, es_index, embedding_type, embedding_model, query, date_range, max_results=500)
df = (pd.DataFrame(runner)).T
print(df.info())
print("Tweets Fetched")
#%%
df=df.rename(columns={0:'Date',1:'Sbert_Score',2:'Is_Verified',3:'Relevancy',4:'Tweet_IDs'})
print('Columns Named')
#%%

df['Date'] = pd.to_datetime(df['Date'])
df['Date'] = df['Date'].apply(lambda x: x.date)
df['Date'] = pd.to_datetime(df['Date'])
print('Date Column Reformatted')

df = df.sort_values(by='Date')
df.reset_index(inplace=True)
df = df.drop(columns=['index'])

print('Sorted by Date')
#%%
pd.set_option('display.max_columns', 10)
pd.set_option('display.width', 1000)
#print(df.head())
#%%
SP_500=pd.read_csv('SP_500.csv')
SP_500 = pd.DataFrame(SP_500)
SP_500['Date'] = pd.to_datetime(SP_500['Date'])
SP_500['Date'] = SP_500['Date'].dt.floor('d')
SP_500 = SP_500.sort_values(by='Date')
SP_500.reset_index(inplace=True)
SP_500 = SP_500.drop(columns=['index'])
#SP_500['Date'] = SP_500['Date'].apply(lambda x: )
#print(SP_500.head())
print("S&P 500 Fetched")
SP_500 = SP_500.rename(columns={' Open':'Open',' High':'High',' Low':'Low',' Close':'Close'})
#%%
SP_500['Mkt_Change'] = SP_500['Open']-SP_500['Close']
SP_500['Mkt_Behavior'] = SP_500['Mkt_Change'].apply(lambda x: 'Increase' if x >= 0 else 'Decrease')
#%%
mega =df.merge(SP_500,on='Date')
print(mega.head())
#%%

from sklearn.model_selection import train_test_split
y = mega['Mkt_Behavior']
y= y.map({'Decrease':0,'Increase':1})
x = mega.drop(columns=['Open','Close','High','Low','Mkt_Behavior','Date','Is_Verified','Tweet_IDs','Mkt_Change'])
#x['Is_Verified'] = x['Is_Verified'].map({False:0,True:1})
x = x.astype(float)
y = y.astype(int)
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2)
#%%
print(y.head())
print(y.unique())
print(x_train.info())
#%%
import numpy as np
import statsmodels.api as sm
lr = sm.Logit(y_train,x_train).fit()
y_pred = lr.predict(x_test)
y_pred = list(map(round, y_pred))
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score

print(r2_score(y_test,y_pred))
print(accuracy_score(y_test,y_pred))
print(lr.summary())
#%%
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(max_depth=10,random_state=0)
rf.fit(x_train,y_train)
print(rf.feature_importances_)

rf_pred=rf.predict(x_test)
print(r2_score(y_test,rf_pred))
print(accuracy_score(y_test,rf_pred))
#%%
import matplotlib.pyplot as plt

feature_names = [f"feature {i}" for i in range(x_train.shape[1])]
forest_importances = pd.Series(rf.feature_importances_, index=feature_names)
std = np.std([rf.feature_importances_ for rf in rf.estimators_], axis=0)
fig, ax = plt.subplots()
forest_importances.plot.bar(yerr=std, ax=ax)
ax.set_title("Feature importances using MDI")
ax.set_ylabel("Mean decrease in impurity")
fig.tight_layout()

#%%
avg_df = mega['Sbert_Score'].groupby([mega['Date'],mega['Tweet_IDs']]).mean().mean(level=0)
avg_df = pd.DataFrame(avg_df).reset_index(False)
print(avg_df.head())
#%%
avg_merged = SP_500.merge(avg_df,on='Date')
print(avg_merged.head())
print(avg_merged.info())
index = np.arange(0,len(avg_merged['Date']))
plt.plot(index,avg_merged['Sbert_Score'])
#%%
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(-1,1))
scaled_avg_merged = avg_merged
col_scale = ['Mkt_Change','Sbert_Score','Close']
scaled_avg_merged[col_scale] = scaler.fit_transform(scaled_avg_merged[col_scale])

print(scaled_avg_merged.head())
#%%
import matplotlib.dates as mdates
index = scaled_avg_merged['Date'].dt.strftime('%m/%d')
df_plot = pd.DataFrame(index)
df_plot['Sbert_Score'] = scaled_avg_merged['Sbert_Score']
ax = df_plot.plot()
# set monthly locator
ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
# set formatter
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
# set font and rotation for date tick labels
plt.gcf().autofmt_xdate()

plt.show()


#%%
index = np.arange(0,len(scaled_avg_merged['Date']))
#plt.plot(index,scaled_avg_merged['Mkt_Change'])
plt.plot(index,scaled_avg_merged['Sbert_Score'])
plt.plot(index,scaled_avg_merged['Close'])
plt.ylabel('Scaled Values')
plt.xlabel('Date')
plt.show()






