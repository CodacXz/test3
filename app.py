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

# API Configuration
NEWS_API_URL = "https://api.marketaux.com/v1/news/all"

# Get API key from secrets with fallback
try:
    API_TOKEN = st.secrets["general"]["MARKETAUX_API_KEY"]
except Exception as e:
    st.error("Error loading API key. Please check your secrets.toml file.")
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

def find_companies_in_text(text, companies_df):
    """Find unique companies mentioned in the text"""
    if not text or companies_df.empty:
        return []
    
    text = text.lower()
    seen_companies = set()  # Track unique companies
    mentioned_companies = []
    
    for _, row in companies_df.iterrows():
        company_name = str(row['Company_Name']).lower()
        company_code = str(row['Company_Code'])
        
        # Only add each company once
        if (company_name in text or company_code in text) and company_code not in seen_companies:
            seen_companies.add(company_code)
            mentioned_companies.append({
                'name': row['Company_Name'],
                'code': company_code,
                'symbol': f"{company_code}.SR"
            })
    
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
        df, error = get_stock_data(f"{symbol}.SR")
        
        if error:
            st.error(error)
            return
            
        if df is not None and not df.empty:
            latest_price = df['Close'][-1]
            price_change = ((latest_price - df['Close'][-2])/df['Close'][-2]*100)
            
            # Metrics
            cols = st.columns(3)
            with cols[0]:
                st.metric(
                    "Current Price",
                    f"{latest_price:.2f} SAR",
                    f"{price_change:.2f}%",
                    key=f"price_{company['symbol']}_{idx}"
                )
            with cols[1]:
                st.metric(
                    "Day High",
                    f"{df['High'][-1]:.2f} SAR",
                    key=f"high_{company['symbol']}_{idx}"
                )
            with cols[2]:
                st.metric(
                    "Day Low",
                    f"{df['Low'][-1]:.2f} SAR",
                    key=f"low_{company['symbol']}_{idx}"
                )
            
            # Technical Analysis
            signals = []
            
            # MACD
            macd_signal = "BULLISH" if df['MACD'][-1] > df['MACD_Signal'][-1] else "BEARISH"
            signals.append({
                'Indicator': 'MACD',
                'Signal': macd_signal,
                'Reason': f"MACD line {'above' if macd_signal == 'BULLISH' else 'below'} signal line"
            })
            
            # RSI
            rsi = df['RSI'][-1]
            if rsi > 70:
                signals.append({
                    'Indicator': 'RSI',
                    'Signal': 'BEARISH',
                    'Reason': 'Overbought condition (RSI > 70)'
                })
            elif rsi < 30:
                signals.append({
                    'Indicator': 'RSI',
                    'Signal': 'BULLISH',
                    'Reason': 'Oversold condition (RSI < 30)'
                })
            else:
                signals.append({
                    'Indicator': 'RSI',
                    'Signal': 'NEUTRAL',
                    'Reason': 'RSI in neutral zone'
                })
            
            # Bollinger Bands
            if df['Close'][-1] > df['BB_upper'][-1]:
                signals.append({
                    'Indicator': 'Bollinger Bands',
                    'Signal': 'BEARISH',
                    'Reason': 'Price above upper band'
                })
            elif df['Close'][-1] < df['BB_lower'][-1]:
                signals.append({
                    'Indicator': 'Bollinger Bands',
                    'Signal': 'BULLISH',
                    'Reason': 'Price below lower band'
                })
            else:
                signals.append({
                    'Indicator': 'Bollinger Bands',
                    'Signal': 'NEUTRAL',
                    'Reason': 'Price within bands'
                })
            
            # Display signals
            st.write("### Technical Analysis")
            signals_df = pd.DataFrame(signals)
            st.dataframe(signals_df, key=f"signals_{company['symbol']}_{idx}")
            
            # Stock chart
            fig = go.Figure()
            fig.add_trace(go.Candlestick(
                x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                name='Price'
            ))
            
            fig.update_layout(
                title=None,
                yaxis_title='Price (SAR)',
                xaxis_title='Date',
                template='plotly_dark',
                height=400,
                margin=dict(t=0)
            )
            
            st.plotly_chart(fig, key=f"chart_{company['symbol']}_{idx}", use_container_width=True)
            
    except Exception as e:
        st.error(f"Error analyzing {company.get('name')}: {str(e)}")

def check_api_credits():
    """Check remaining API credits"""
    try:
        params = {
            "api_token": API_TOKEN
        }
        response = requests.get("https://api.marketaux.com/v1/usage", params=params)
        response.raise_for_status()
        data = response.json()
        return data.get("credits", {})
    except Exception as e:
        st.error(f"Error checking API credits: {str(e)}")
        return None

def main():
    st.title("Saudi Stock Market News")
    st.write("Real-time news analysis for Saudi stock market")
    
    # Initialize session state
    if 'api_calls_today' not in st.session_state:
        st.session_state.api_calls_today = 0
    
    # Sidebar
    with st.sidebar:
        st.header("Settings")
        
        # API Credits
        credits = check_api_credits()
        if credits:
            st.write("### API Credits")
            st.write(f"Used: {credits.get('used', 'N/A')}")
            st.write(f"Remaining: {credits.get('remaining', 'N/A')}")
            st.write(f"Limit: {credits.get('limit', 'N/A')}")
        
        # Reset button
        if st.button("🔄 Reset Session"):
            for key in list(st.session_state.keys()):
                if key.startswith('skip_') or key == 'api_calls_today':
                    del st.session_state[key]
            st.experimental_rerun()
        
        # Company data upload
        uploaded_file = st.file_uploader(
            "Upload companies file (optional)",
            type=['csv'],
            key="file_uploader"
        )
        
        companies_df = load_company_data(uploaded_file)
        if companies_df.empty:
            st.error("Failed to load company data")
            return
        
        # Date range
        days_ago = st.slider(
            "Show news published after:",
            min_value=1,
            max_value=30,
            value=1,
            key="days_slider"
        )
        
        # Number of articles
        article_limit = st.number_input(
            "Number of articles",
            min_value=1,
            max_value=3,
            value=3,
            key="article_limit"
        )
    
    published_after = (datetime.now() - timedelta(days=days_ago)).strftime("%Y-%m-%d")
    
    # Fetch news
    if st.button("Fetch News", use_container_width=True):
        news_data = fetch_news(published_after, limit=article_limit)
        
        if not news_data:
            st.error("No news articles found")
            return
        
        st.write(f"Found {len(news_data)} articles")
        
        # Keep track of analyzed companies to avoid duplicates
        analyzed_companies = set()
        
        # Process each article
        for article_idx, article in enumerate(news_data):
            with st.container():
                # Article header
                title = article.get('title', 'No title')
                description = article.get('description', 'No description')
                url = article.get('url', '#')
                source = article.get('source', 'Unknown')
                published_at = article.get('published_at', '')
                
                st.header(title, key=f"header_{article_idx}")
                st.write(f"Source: {source} | Published: {published_at[:16]}", key=f"source_{article_idx}")
                st.write(description, key=f"description_{article_idx}")
                
                # Sentiment Analysis
                text = f"{title} {description}"
                sentiment, confidence = analyze_sentiment(text)
                
                st.write("### Sentiment Analysis", key=f"sentiment_analysis_{article_idx}")
                st.write(f"**Sentiment:** {sentiment}", key=f"sentiment_{article_idx}")
                st.write(f"**Confidence:** {confidence:.2f}%", key=f"confidence_{article_idx}")
                
                # Company Analysis
                entities = article.get('entities', [])
                if entities:
                    unique_companies = []
                    for entity in entities:
                        symbol = entity.get('symbol')
                        if symbol and symbol not in analyzed_companies:
                            analyzed_companies.add(symbol)
                            unique_companies.append(entity)
                    
                    if unique_companies:
                        st.write("### Companies Mentioned", key=f"companies_mentioned_{article_idx}")
                        for company_idx, company in enumerate(unique_companies):
                            st.write(f"**{company.get('name')} ({company.get('symbol')})**", key=f"company_{article_idx}_{company_idx}")
                            
                            try:
                                symbol = company.get('symbol')
                                df, error = get_stock_data(f"{symbol}.SR")
                                
                                if error:
                                    st.error(error, key=f"error_{article_idx}_{company_idx}")
                                    continue
                                    
                                if df is not None and not df.empty:
                                    latest_price = df['Close'][-1]
                                    price_change = ((latest_price - df['Close'][-2])/df['Close'][-2]*100)
                                    
                                    cols = st.columns(3)
                                    with cols[0]:
                                        st.metric(
                                            "Current Price",
                                            f"{latest_price:.2f} SAR",
                                            f"{price_change:.2f}%",
                                            key=f"current_price_{article_idx}_{company_idx}"
                                        )
                                    with cols[1]:
                                        st.metric(
                                            "Day High",
                                            f"{df['High'][-1]:.2f} SAR",
                                            key=f"day_high_{article_idx}_{company_idx}"
                                        )
                                    with cols[2]:
                                        st.metric(
                                            "Day Low",
                                            f"{df['Low'][-1]:.2f} SAR",
                                            key=f"day_low_{article_idx}_{company_idx}"
                                        )
                                    
                                    # Technical Analysis
                                    signals = []
                                    
                                    # MACD
                                    macd_signal = "BULLISH" if df['MACD'][-1] > df['MACD_Signal'][-1] else "BEARISH"
                                    signals.append({
                                        'Indicator': 'MACD',
                                        'Signal': macd_signal,
                                        'Reason': f"MACD line {'above' if macd_signal == 'BULLISH' else 'below'} signal line"
                                    })
                                    
                                    # RSI
                                    rsi = df['RSI'][-1]
                                    if rsi > 70:
                                        signals.append({
                                            'Indicator': 'RSI',
                                            'Signal': 'BEARISH',
                                            'Reason': 'Overbought condition (RSI > 70)'
                                        })
                                    elif rsi < 30:
                                        signals.append({
                                            'Indicator': 'RSI',
                                            'Signal': 'BULLISH',
                                            'Reason': 'Oversold condition (RSI < 30)'
                                        })
                                    else:
                                        signals.append({
                                            'Indicator': 'RSI',
                                            'Signal': 'NEUTRAL',
                                            'Reason': 'RSI in neutral zone'
                                        })
                                    
                                    # Bollinger Bands
                                    if df['Close'][-1] > df['BB_upper'][-1]:
                                        signals.append({
                                            'Indicator': 'Bollinger Bands',
                                            'Signal': 'BEARISH',
                                            'Reason': 'Price above upper band'
                                        })
                                    elif df['Close'][-1] < df['BB_lower'][-1]:
                                        signals.append({
                                            'Indicator': 'Bollinger Bands',
                                            'Signal': 'BULLISH',
                                            'Reason': 'Price below lower band'
                                        })
                                    else:
                                        signals.append({
                                            'Indicator': 'Bollinger Bands',
                                            'Signal': 'NEUTRAL',
                                            'Reason': 'Price within bands'
                                        })
                                    
                                    st.write("### Technical Analysis", key=f"technical_analysis_{article_idx}_{company_idx}")
                                    signals_df = pd.DataFrame(signals)
                                    st.dataframe(signals_df, key=f"signals_df_{article_idx}_{company_idx}")
                                    
                                    # Stock chart
                                    fig = go.Figure()
                                    fig.add_trace(go.Candlestick(
                                        x=df.index,
                                        open=df['Open'],
                                        high=df['High'],
                                        low=df['Low'],
                                        close=df['Close'],
                                        name='Price'
                                    ))
                                    
                                    fig.update_layout(
                                        title=None,
                                        yaxis_title='Price (SAR)',
                                        xaxis_title='Date',
                                        template='plotly_dark',
                                        height=400,
                                        margin=dict(t=0)
                                    )
                                    
                                    st.plotly_chart(fig, use_container_width=True, key=f"chart_{article_idx}_{company_idx}")
                                    
                            except Exception as e:
                                st.error(f"Error analyzing {company.get('name')}: {str(e)}", key=f"exception_{article_idx}_{company_idx}")
                
                st.markdown(f"[Read full article]({url})", key=f"read_full_article_{article_idx}")
                st.markdown("---", key=f"divider_{article_idx}")

if __name__ == "__main__":
    main()
