import pandas as pd
import yfinance as yf
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
def main():
   df = yf.download("GC=F",period="10y",interval="1d")
   df["return"] = df["Close"].pct_change()
   df["MA_10"] = df["Close"].rolling(10).mean()
   df["MA_20"] = df["Close"].rolling(20).mean()
   df["MA_5"] = df["Close"].rolling(5).mean()

   df["Volatility"] = df["Close"].rolling(10).std()
   df["Target"] = df["Close"].shift(-1)

   df.dropna(inplace=True)
   return df
df = main()

def modelfitting():
    
  X = df[["MA_5","Volatility","MA_20","MA_10","return"]]
  y = df["Target"]
  model = RandomForestRegressor(n_estimators=300,random_state=42,max_depth=10)
  model.fit(X,y)
  return model
model = modelfitting()
def extraction():
   latest =  yf.download("GC=F",period="30d",interval="1d")
   latest["return"] = latest["Close"].pct_change()
   latest["MA_10"] = latest["Close"].rolling(10).mean()
   latest["MA_20"] = latest["Close"].rolling(20).mean()
   latest["MA_5"] = latest["Close"].rolling(5).mean()
   latest["Volatility"] = latest["Close"].rolling(10).std()

   latest["Target"] = latest["Close"].shift(-1)
   latest.dropna()
   X_latest = latest[["MA_5","Volatility","MA_20","MA_10","return"]].tail(1)
   predictions = model.predict(X_latest)
   return predictions,latest
predictions,latest = extraction()
st.title("🪙Gold Price Prediction🪙")
def visualization():
   figure, ax = plt.subplots(figsize=(15, 10))
   ax.plot(df.index, df["Close"], label="Historical trend of Gold Price")
   fig, ax = plt.subplots(figsize=(15, 10))
   ax.plot(latest.index, latest["Close"], label="Actual Price")
   return figure,fig
figure,fig = visualization()
st.subheader("Historical Trend of Gold Price for 10 years")
st.pyplot(figure)
st.subheader("Latest Trend of Gold Price for 30 days")
st.pyplot(fig)
st.info("If you are predicting the gold price during the day time then you will get the Close price of that day as the predicted price. If you are predicting the gold price after the market closes then you will get the predicted price for the next day.")

if st.button("Predict Gold Price"):
     st.write("The predicted price of gold is:")
     st.write(f"${predictions[0]}")
     st.info("The predicted priced is for 1 Troy Ounce of Gold.")




    
