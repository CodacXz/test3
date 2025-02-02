import streamlit as st
import pandas as pd
import requests
from datetime import datetime, timedelta
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import yfinance as yf
import plotly.graph_objects as go
from ta.trend import MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
import re

# Page configuration
st.set_page_config(page_title="Saudi Market News Analyzer", layout="wide")

# API Configuration
NEWS_API_URL = "https://api.marketaux.com/v1/news/all"

# Get API key from secrets
try:
    API_TOKEN = st.secrets["general"]["MARKETAUX_API_KEY"]
except Exception as e:
    st.error("Error loading API key. Please check your secrets.toml file.")
    st.stop()

def test_api_key():
    try:
        response = requests.get(f"{NEWS_API_URL}?api_token={API_TOKEN}&countries=sa&limit=1")
        response.raise_for_status()
        st.success("API key is valid and working.")
    except requests.exceptions.RequestException as e:
        st.error("Error validating API key. Please check your API key and try again.")
        with st.expander("See error details"):
            st.write(f"Error type: {type(e).__name__}")
            st.write(f"Error message: {str(e)}")
            if hasattr(e, 'response'):
                st.write(f"Response status code: {e.response.status_code}")
                st.write(f"Response content: {e.response.text}")
        st.stop()

@st.cache_data
def load_company_data(uploaded_file=None):
    """Load company data from uploaded file or default GitHub URL"""
    try:
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
        else:
            github_url = "https://raw.githubusercontent.com/CodacXz/Test/main/saudi_companies.csv?raw=true"
            df = pd.read_csv(github_url)
        
        # Clean and prepare data
        df['Company_Name'] = df['Company_Name'].str.strip()
        df['Company_Code'] = df['Company_Code'].astype(str).str.zfill(4)
        
        return df
    except Exception as e:
        st.error(f"Error loading company data: {str(e)}")
        return pd.DataFrame()

def generate_name_variations(name):
    """Generate common variations of company names"""
    variations = [name]
    variations.append(name.replace(" ", ""))
    variations.append(name.replace("&", "and"))
    variations.append(name.replace("and", "&"))
    if "company" in name.lower():
        variations.append(name.lower().replace("company", "co"))
    if "corporation" in name.lower():
        variations.append(name.lower().replace("corporation", "corp"))
    return list(set(variations))

def find_companies_in_text(text, companies_df):
    """Find unique companies mentioned in the text"""
    if not text or companies_df.empty:
        return []
    
    text = text.lower()
    seen_companies = set()
    mentioned_companies = []
    
    for _, row in companies_df.iterrows():
        company_name = str(row['Company_Name'])
        company_code = str(row['Company_Code'])
        
        name_variations = generate_name_variations(company_name)
        
        for variation in name_variations:
            variation_clean = variation.lower().replace("'s", "").replace("'", "")
            if (re.search(r'\b' + re.escape(variation_clean) + r'\b', text) or 
                re.search(r'\b' + re.escape(company_code) + r'\b', text)) and company_code not in seen_companies:
                seen_companies.add(company_code)
                mentioned_companies.append({
                    'name': row['Company_Name'],
                    'code': company_code,
                    'symbol': f"{company_code}.SA"
                })
                break
    
    return mentioned_companies

def analyze_sentiment(text):
    """Analyze sentiment of text"""
    analyzer = SentimentIntensityAnalyzer()
    scores = analyzer.polarity_scores(text)
    
    compound_score = scores['compound']
    if compound_score >= 0.05:
        sentiment = 'Positive'
    elif compound_score <= -0.05:
        sentiment = 'Negative'
    else:
        sentiment = 'Neutral'
    
    return {
        'compound': compound_score,
        'positive': scores['pos'],
        'negative': scores['neg'],
        'neutral': scores['neu'],
        'sentiment': sentiment
    }

@st.cache_data(ttl=3600)
def fetch_news(days=7, limit=100):
    """Fetch news articles from the API"""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    params = {
        'api_token': API_TOKEN,
        'countries': 'sa',
        'limit': limit,
        'published_after': start_date.strftime('%Y-%m-%d'),
        'published_before': end_date.strftime('%Y-%m-%d')
    }
    
    try:
        response = requests.get(NEWS_API_URL, params=params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching news: {str(e)}")
        return None

def fetch_stock_data(symbol, period='1y'):
    """Fetch stock data using yfinance"""
    try:
        stock = yf.Ticker(symbol)
        df = stock.history(period=period)
        return df
    except Exception as e:
        st.error(f"Error fetching stock data for {symbol}: {str(e)}")
        return pd.DataFrame()

def calculate_technical_indicators(df):
    """Calculate technical indicators for the stock data"""
    if df.empty:
        return df
    
    # MACD
    macd = MACD(close=df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    
    # RSI
    rsi = RSIIndicator(close=df['Close'])
    df['RSI'] = rsi.rsi()
    
    # Bollinger Bands
    bb = BollingerBands(close=df['Close'])
    df['BB_High'] = bb.bollinger_hband()
    df['BB_Low'] = bb.bollinger_lband()
    df['BB_Mid'] = bb.bollinger_mavg()
    
    return df

def plot_stock_data(df, company_name):
    """Create interactive stock chart with technical indicators"""
    fig = go.Figure()
    
    # Candlestick chart
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='OHLC'
    ))
    
    # Bollinger Bands
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_High'], name='BB Upper', line=dict(color='gray', dash='dash')))
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_Low'], name='BB Lower', line=dict(color='gray', dash='dash')))
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_Mid'], name='BB Middle', line=dict(color='gray')))
    
    fig.update_layout(
        title=f'{company_name} Stock Price and Bollinger Bands',
        yaxis_title='Price',
        xaxis_title='Date',
        template='plotly_dark'
    )
    
    return fig

def main():
    st.title("Saudi Market News Analyzer")
    
    # Load company data
    companies_df = load_company_data()
    if companies_df.empty:
        st.error("Failed to load company data.")
        return
    
    # Sidebar controls
    st.sidebar.header("Settings")
    days_back = st.sidebar.slider("Days of news to analyze", 1, 30, 7)
    news_limit = st.sidebar.slider("Number of news articles", 10, 100, 50)
    
    # Fetch news
    with st.spinner("Fetching news articles..."):
        news_data = fetch_news(days=days_back, limit=news_limit)
    
    if not news_data or 'data' not in news_data:
        st.error("No news data available.")
        return
    
    # Process news articles
    all_articles = []
    for article in news_data['data']:
        companies = find_companies_in_text(article['title'] + ' ' + article['description'], companies_df)
        if companies:
            sentiment = analyze_sentiment(article['title'] + ' ' + article['description'])
            for company in companies:
                all_articles.append({
                    'date': article['published_at'],
                    'title': article['title'],
                    'description': article['description'],
                    'company_name': company['name'],
                    'company_code': company['code'],
                    'symbol': company['symbol'],
                    'sentiment': sentiment['sentiment'],
                    'compound_score': sentiment['compound']
                })
    
    # Display results
    if all_articles:
        df_articles = pd.DataFrame(all_articles)
        
        # Group by company and show sentiment statistics
        company_stats = df_articles.groupby('company_name').agg({
            'compound_score': ['mean', 'count'],
            'sentiment': lambda x: x.value_counts().index[0]
        }).round(3)
        
        st.header("Company Sentiment Analysis")
        st.dataframe(company_stats)
        
        # Show detailed news for selected company
        selected_company = st.selectbox(
            "Select a company to view detailed news and analysis",
            options=df_articles['company_name'].unique()
        )
        
        company_news = df_articles[df_articles['company_name'] == selected_company]
        symbol = company_news.iloc[0]['symbol']
        
        # Fetch and display stock data
        with st.spinner("Fetching stock data..."):
            stock_data = fetch_stock_data(symbol)
            if not stock_data.empty:
                stock_data = calculate_technical_indicators(stock_data)
                st.plotly_chart(plot_stock_data(stock_data, selected_company))
        
        # Display news articles
        st.subheader(f"Recent News for {selected_company}")
        for _, article in company_news.iterrows():
            with st.expander(f"{article['date']} - {article['title']} ({article['sentiment']})"):
                st.write(article['description'])
    else:
        st.info("No company-related news found in the specified time period.")

if __name__ == "__main__":
    main()
