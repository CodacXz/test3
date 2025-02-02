I apologize for the confusion. It seems that the previous response was mistakenly included in the code block. Let me provide you with the correct, full code for your `app.py` file without any extra text:

**File: c:\Users\DELL\Downloads\Test-main (2)\Test-main\app.py**
```python
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
    variations.append(name.replace(" ", ""))  # Remove spaces
    variations.append(name.replace("&", "and"))  # Replace & with and
    variations.append(name.replace("and", "&"))  # Replace and with &
    if "company" in name.lower():
        variations.append(name.lower().replace("company", "co"))  # Replace Company with Co
    if "corporation" in name.lower():
        variations.append(name.lower().replace("corporation", "corp"))  # Replace Corporation with Corp
    return list(set(variations))  # Remove duplicates

def find_companies_in_text(text, companies_df):
    """Find unique companies mentioned in the text"""
    if not text or companies_df.empty:
        return []
    
    text = text.lower()
    seen_companies = set()  # Track unique companies
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
                break  # Stop checking variations once we find a match
    
    return mentioned_companies

def analyze_sentiment(text):
    """Analyze sentiment of text"""
    analyzer = SentimentIntensityAnalyzer()
    scores = analyzer.polarity_scores(text)
    compound = scores['compound']
    confidence = (abs(compound) * 100)  # Convert to percentage
    
    if compound >= 0.05:
        sentiment = "🟢 Positive"
    elif compound <= -0.05:
        sentiment = "🔴 Negative"
    else:
        sentiment = "⚪ Neutral"
    
    return sentiment, confidence

def fetch_news(published_after, limit=3):
    """Fetch news articles"""
    params = {
        "api_token": API_TOKEN,
        "countries": "sa",
        "filter_entities": "true",
        "limit": limit,
        "published_after": published_after,
        "language": "en",
        "must_have_entities": "true",  # Only get articles with entities
        "group_similar": "true"  # Group similar articles to save API calls
    }
    try:
        response = requests.get(NEWS_API_URL, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        return data.get("data", [])
    except Exception as e:
        st.error(f"Error fetching news: {str(e)}")
        return []

@st.cache_data(ttl=3600)  # Cache stock data for 1 hour
def get_stock_data(symbol, period='1mo'):
    """Fetch stock data and calculate technical indicators"""
    try:
        # For Saudi stocks, use the format XXXX.SA
        if symbol.endswith('.SR'):
            symbol = symbol[:-3] + '.SA'
        
        stock = yf.Ticker(symbol)
        df = stock.history(period=period)
        
        if df.empty:
            return None, f"No stock data available for {symbol}"
        
        # Calculate indicators
        macd = MACD(df['Close'])
        df['MACD'] = macd.macd()
        df['MACD_Signal'] = macd.macd_signal()
        
        rsi = RSIIndicator(df['Close'])
        df['RSI'] = rsi.rsi()
        
        bb = BollingerBands(df['Close'])
        df['BB_upper'] = bb.bollinger_hband()
        df['BB_lower'] = bb.bollinger_lband()
        
        return df, None
    except Exception as e:
        return None, f"Error fetching data for {symbol}: {str(e)}"

def analyze_company(company, idx):
    """Analyze a single company"""
    try:
        symbol = company.get('symbol')
        df, error = get_stock_data(symbol)
        
        if error:
            st.warning(error)
            return
        
        if df is None or df.empty:
            st.warning(f"No data available for {company.get('name')} ({symbol})")
            return
        
        # Display stock chart
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=df.index,
                                     open=df['Open'],
                                     high=df['High'],
                                     low=df['Low'],
                                     close=df['Close'],
                                     name='Price'))
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_upper'], name='BB Upper'))
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_lower'], name='BB Lower'))
        fig.update_layout(title=f"{company.get('name')} Stock Price", xaxis_title="Date", yaxis_title="Price")
        st.plotly_chart(fig, use_container_width=True, key=f"price_chart_{idx}")
        
        # Display technical indicators
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("MACD")
            fig_macd = go.Figure()
            fig_macd.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='MACD'))
            fig_macd.add_trace(go.Scatter(x=df.index, y=df['MACD_Signal'], name='Signal'))
            st.plotly_chart(fig_macd, use_container_width=True, key=f"macd_chart_{idx}")
        
        with col2:
            st.subheader("RSI")
            fig_rsi = go.Figure()
            fig_rsi.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI'))
            fig_rsi.add_hline(y=70, line_dash="dash", line_color="red")
            fig_rsi.add_hline(y=30, line_dash="dash", line_color="green")
            st.plotly_chart(fig_rsi, use_container_width=True, key=f"rsi_chart_{idx}")
        
        # Display summary statistics
        st.subheader("Summary Statistics")
        summary = df['Close'].describe()
        st.write(summary)
    except Exception as e:
        st.error(f"Error analyzing {company.get('name')}: {str(e)}")

def main():
    st.set_page_config(page_title="Saudi Stock Market News", page_icon="📈", layout="wide")
    st.title("Saudi Stock Market News")
    st.write("Real-time news analysis for Saudi stock market")

    # Test API key
    test_api_key()

    # Sidebar
    st.sidebar.title("Settings")
    
    # File uploader for company data
    uploaded_file = st.sidebar.file_uploader("Upload companies file (optional)", type="csv")
    
    # Date input for news
    days_ago = st.sidebar.slider("Show news published after:", 1, 30, 7)
    published_after = (datetime.now() - timedelta(days=days_ago)).strftime("%Y-%m-%d")
    
    # Number of articles to fetch
    article_limit = st.sidebar.number_input("Number of articles", min_value=1, max_value=100, value=3)

    # Load company data
    companies_df = load_company_data(uploaded_file)

    if companies_df.empty:
        st.error("Failed to load company data. Please check your CSV file or internet connection.")
        return

    # Fetch and display news
    news_articles = fetch_news(published_after, article_limit)

    for idx, article in enumerate(news_articles, 1):
        st.subheader(f"Article {idx}: {article['title']}")
        st.write(f"Published: {article['published_at']}")
        st.write(f"Source: {article['source']}")

        # Analyze sentiment
        sentiment, confidence = analyze_sentiment(article['title'] + " " + article['description'])
        st.write(f"Sentiment: {sentiment} (Confidence: {confidence:.2f}%)")

        # Find mentioned companies
        mentioned_companies = find_companies_in_text(article['title'] + " " + article['description'], companies_df)
        if mentioned_companies:
            st.write("Mentioned Companies:")
            for company in mentioned_companies:
                st.write(f"{company['name']} ({company['code']})")
        else:
            st.write("No companies mentioned.")

        st.write("Article Description:")
        st.write(article['description'])
        
        if st.button(f"Read full article {idx}", key=
