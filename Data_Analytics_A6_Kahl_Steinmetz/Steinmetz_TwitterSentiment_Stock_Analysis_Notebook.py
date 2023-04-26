import aspects as asp
from sentence_transformers import SentenceTransformer
from datetime import datetime
import pandas as pd

#Create query to run elastisearch query using 'run_query' function from 'Twitter-NLP.aspect_modeling' repo
#Twitter-NLP Github Repo: https://github.com/TheRensselaerIDEA/twitter-nlp.git


#Manually construct necessary query variables to satisfy 'aspects.run_query' function
#Set elastisearch location
es_uri = "https://localhost:8080/elasticsearch/"
#Set embedding type
embedding_type = 'sbert'

#Create date range (YYYY,MM,D)
begin = datetime(2022,10,1)
end = datetime(2022,11,1)
date_range = [begin,end]

#Select data subset to pull from
es_index = "ukraine-data-lite-mar23"
#More query variables
embedding_model = SentenceTransformer("all-MiniLM-L12-v2")

#Text string to query (returns tweets elastisearch deems most similar to string)
query = "US Economy"

#Run and assign 'run_query' results to dataframe
runner = asp.run_query(es_uri, es_index, embedding_type, embedding_model, query, date_range, max_results=500)
df = (pd.DataFrame(runner)).T
print(df.info())
print("Tweets Fetched")
#%%
#Apply column names to tweet dataframe

df=df.rename(columns={0:'Date',1:'Sbert_Score',2:'Is_Verified',3:'Relevancy',4:'Tweet_IDs'})
print('Columns Named')
#%%
#Reformat 'Date' column, sort dataframe by date

df['Date'] = pd.to_datetime(df['Date'])
df['Date'] = df['Date'].apply(lambda x: x.date)
df['Date'] = pd.to_datetime(df['Date'])
print('Date Column Reformatted')

df = df.sort_values(by='Date')
df.reset_index(inplace=True)
df = df.drop(columns=['index'])

print('Sorted by Date')
#%%
#Resize output window so whole dataframe heads can be viewed

pd.set_option('display.max_columns', 10)
pd.set_option('display.width', 1000)
#print(df.head())
#%%
#Import S&P_500 Data
#Rename columns and convert date to datetime

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
#Create Mkt_Change and Mkt_Behavior Features
#Mkt_Change is the difference between a day's open and close
#Mkt_Behavior is the categorical representation of Mkt_Change,
#	If Mkt Change == (some positive), then Mkt_Behavior = 'Increase', vice versa

SP_500['Mkt_Change'] = SP_500['Open']-SP_500['Close']
SP_500['Mkt_Behavior'] = SP_500['Mkt_Change'].apply(lambda x: 'Increase' if x >= 0 else 'Decrease')
#%%
#Merge S&P 500 dataset and tweet dataset on 'Date'

mega =df.merge(SP_500,on='Date')
print(mega.info())
#%%
#Perform train test split on merged dataset

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
#Train and test logistic regression (prints summary and performance metrics)

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
#Train and test random forest model (prints performance metrics)

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(max_depth=10,random_state=0)
rf.fit(x_train,y_train)
print(rf.feature_importances_)

rf_pred=rf.predict(x_test)
print(r2_score(y_test,rf_pred))
print(accuracy_score(y_test,rf_pred))
#%%
#Plot random forest feature importances

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
#Create a dataframe of average daily sentiment values

avg_df = mega['Sbert_Score'].groupby([mega['Date'],mega['Tweet_IDs']]).mean().mean(level=0)
avg_df = pd.DataFrame(avg_df).reset_index(False)
print(avg_df.head())
#%%
# Merge S&P 500 dataset with the averaged daily sentiment values

avg_merged = SP_500.merge(avg_df,on='Date')
print(avg_merged.head())
print(avg_merged.info())
index = np.arange(0,len(avg_merged['Date']))
plt.plot(index,avg_merged['Sbert_Score'])
#%%
#Scale Target features for plotting so they fit on the same y-axis

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(-1,1))
scaled_avg_merged = avg_merged
col_scale = ['Mkt_Change','Sbert_Score','Close']
scaled_avg_merged[col_scale] = scaler.fit_transform(scaled_avg_merged[col_scale])

print(scaled_avg_merged.head())

#%%
# Plotting Market Close and Daily Average Sentiment

index = scaled_avg_merged['Date'].dt.strftime('%m/%d')
plt.figure(figsize=(10,6))
plt.plot(index,scaled_avg_merged['Sbert_Score'])
plt.plot(index,scaled_avg_merged['Close'])
plt.ylabel('Scaled Values')
plt.xlabel('Date')
plt.legend(['Daily Sentiment Avg','Market Close'])
plt.gcf().autofmt_xdate()
plt.title('Market Close and Daily Average Sentiment')
plt.show()

#%%
# Plotting Market Change and Daily Average Sentiment

plt.figure(figsize=(10,6))
plt.plot(index,scaled_avg_merged['Sbert_Score'])
plt.plot(index,scaled_avg_merged['Mkt_Change'])
plt.ylabel('Scaled Values')
plt.xlabel('Date')
plt.legend(['Daily Sentiment Avg','Market Daily Change'])
plt.title('Daily Market Change and Daily Average Sentiment')

plt.gcf().autofmt_xdate()
plt.show()

#%%
# Plotting Market Change and Daily Average Sentiment SHIFTED 3 DAYS

plt.figure(figsize=(10,6))
plt.plot(range(3,len(scaled_avg_merged['Mkt_Change'])+3),scaled_avg_merged['Sbert_Score'])
plt.plot(index,scaled_avg_merged['Mkt_Change'])
plt.ylabel('Scaled Values')
plt.xlabel('Date')
plt.legend(['Daily Sentiment Avg (3 Day Shift)','Market Daily Change'])
plt.title('Daily Market Change and Daily Average Sentiment (Shifted 3 Days)')
plt.xlim(["10/06","10/31"])
plt.gcf().autofmt_xdate()
plt.show()
#%%
#Train Test Split Averaged Data

y = avg_merged['Mkt_Change']

x = avg_merged.drop(columns=['Open','Close','High','Low','Mkt_Behavior','Date','Mkt_Change'])


x_train2, x_test2, y_train2, y_test2 = train_test_split(x,y,test_size = 0.3)



#%%
#Train and Test Random Forest Regressor

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

rf = RandomForestRegressor(max_depth=10,random_state=0)
rf.fit(x_train2,y_train2)


rf_pred2=rf.predict(x_test2)
print(r2_score(y_test2,rf_pred2))
print(mean_squared_error(y_test2,rf_pred2))

#%%
#Train and Test Linear Regression
from sklearn.linear_model import LinearRegression
lr = LinearRegression().fit(x_train2,y_train2)
y_pred2 = lr.predict(x_test2)


print(r2_score(y_test2,y_pred2))
print(mean_squared_error(y_test2,y_pred2))

#%%
#Create Predicted/True Scatter Plot for Linear Regression Model

plt.figure(figsize=(10,6))

plt.scatter(range(0,len(y_pred2)),y_pred2)
plt.scatter(range(0,len(y_pred2)),y_test2)
plt.axhline(y=avg_merged['Mkt_Change'].mean(),color='r')
plt.ylabel('Market Change')
plt.xlabel('Instance')
plt.grid()
plt.legend(['Predicted','Actual','Mean Mkt Change'])
plt.title('Linear Regression Predicted and Actual Test Data')

plt.show()

#%%
#Create Histogram of tweet counts by date

plt.figure(figsize=(10,6))
plt.hist(df['Date'],20)

plt.ylabel('Count of Tweets')
plt.xlabel('Date')

plt.title('Histogram of Tweets by Date')
#plt.xlim(["10/06","10/31"])
plt.gcf().autofmt_xdate()
plt.show()

#%%
#Plot stock market data 

plt.figure(figsize=(10,6))
  
col1 = 'green'
col2 = 'red'

  
# Setting width of candlestick elements
width1 = .3
width2 = .03

increase = SP_500[SP_500['Mkt_Behavior']=='Increase']
decrease = SP_500[SP_500['Mkt_Behavior']=='Decrease']
  
# Plotting stock increases
plt.bar(increase['Date'], increase['Close']-increase['Open'], width1, bottom=increase['Open'], color=col1)
plt.bar(increase['Date'], increase['High']-increase['Close'], width2, bottom=increase['Close'], color=col1)
plt.bar(increase['Date'], increase['Low']-increase['Open'], width2, bottom=increase['Open'], color=col1)
  
# Plotting stock decreases
plt.bar(decrease['Date'], decrease['Close']-decrease['Open'], width1, bottom=decrease['Open'], color=col2)
plt.bar(decrease['Date'], decrease['High']-decrease['Open'], width2, bottom=decrease['Open'], color=col2)
plt.bar(decrease['Date'], decrease['Low']-decrease['Close'], width2, bottom=decrease['Close'], color=col2)
  


plt.title('S&P 500 Performance')
plt.ylabel('S&P 500 Index')
plt.xlabel('Date')
plt.gcf().autofmt_xdate()
plt.grid()
plt.show()

#%%
#Plot histogram of S&P 500 target feature

plt.figure(figsize=(10,6))
plt.scatter(SP_500['Date'],SP_500['Mkt_Change'])
plt.ylabel('Daily Index Change')
plt.xlabel('Date')
plt.title('S&P 500 Daily Change')
plt.grid()
plt.gcf().autofmt_xdate()
plt.show()



