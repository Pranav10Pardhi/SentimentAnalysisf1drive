import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import ssl

# Fix SSL certificate verification issues
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass

# Download required NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

class IntegratedAnalytics:
    def __init__(self):
        self.sentiment_model = None
        self.vectorizer = TfidfVectorizer(max_features=5000)
        self.kmeans = KMeans(n_clusters=5, random_state=42)
        self.scaler = StandardScaler()
        
    def train_sentiment_model(self, texts, labels):
        if not texts.empty and not labels.empty:
            processed_texts = [self.preprocess_text(text) for text in texts]
            X = self.vectorizer.fit_transform(processed_texts)
            self.sentiment_model = SVC(kernel='linear')
            self.sentiment_model.fit(X, labels)
            return True
        return False
        
    def preprocess_text(self, text):
        if pd.isna(text):
            return ""
        text = str(text).lower()
        tokens = word_tokenize(text)
        stop_words = set(stopwords.words('english'))
        tokens = [t for t in tokens if t not in stop_words]
        return ' '.join(tokens)
    
    def predict_sentiment(self, text):
        if self.sentiment_model is None:
            return None
        processed_text = self.preprocess_text(text)
        X = self.vectorizer.transform([processed_text])
        return self.sentiment_model.predict(X)[0]
    
    def segment_customers(self, data):
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        scaled_data = self.scaler.fit_transform(data[numeric_cols])
        return self.kmeans.fit_predict(scaled_data)

def main():
    st.set_page_config(layout="wide")
    st.title("Business Analytics Dashboard")
    
    analytics = IntegratedAnalytics()
    
    page = st.sidebar.selectbox("Choose Analysis", 
                               ["Sentiment Analysis", "Customer Segmentation", "Sales Dashboard"])
    
    if page == "Sentiment Analysis":
        st.header("Social Media Sentiment Analysis")
        
        uploaded_file = st.file_uploader("Upload training data (CSV with 'text' and 'sentiment' columns)")
        
        if uploaded_file:
            try:
                data = pd.read_csv(uploaded_file)
                if 'text' in data.columns and 'sentiment' in data.columns:
                    if analytics.train_sentiment_model(data['text'], data['sentiment']):
                        st.success("Model trained successfully!")
                else:
                    st.error("CSV must contain 'text' and 'sentiment' columns")
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
        
        user_text = st.text_area("Enter text for sentiment analysis:")
        if user_text and analytics.sentiment_model is not None:
            sentiment = analytics.predict_sentiment(user_text)
            if sentiment is not None:
                st.write(f"Predicted Sentiment: {'Positive' if sentiment == 1 else 'Negative'}")
    
    elif page == "Customer Segmentation":
        st.header("Customer Segmentation Analysis")
        
        uploaded_file = st.file_uploader("Upload customer data (CSV)")
        
        if uploaded_file:
            try:
                data = pd.read_csv(uploaded_file)
                if not data.empty:
                    segments = analytics.segment_customers(data)
                    data['Segment'] = segments
                    
                    st.subheader("Customer Segments Visualization")
                    numeric_cols = data.select_dtypes(include=[np.number]).columns
                    x_axis = st.selectbox("Select X-axis", numeric_cols)
                    y_axis = st.selectbox("Select Y-axis", numeric_cols)
                    
                    fig = px.scatter(data, x=x_axis, y=y_axis,
                                   color='Segment', title='Customer Segments')
                    st.plotly_chart(fig)
                    
                    st.subheader("Segment Statistics")
                    st.write(data.groupby('Segment').agg(['mean', 'count']))
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
    
    else:  # Sales Dashboard
        st.header("Real-Time Sales Dashboard")
        
        uploaded_file = st.file_uploader("Upload sales data (CSV)")
        
        if uploaded_file:
            try:
                data = pd.read_csv(uploaded_file)
                if 'Date' in data.columns:
                    data['Date'] = pd.to_datetime(data['Date'])
                    
                    date_range = st.date_input("Select Date Range",
                                             [data['Date'].min(), data['Date'].max()])
                    
                    filtered_data = data[(data['Date'].dt.date >= date_range[0]) &
                                       (data['Date'].dt.date <= date_range[1])]
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        total_sales = filtered_data['Sales'].sum()
                        st.metric("Total Sales", f"${total_sales:,.2f}")
                    with col2:
                        avg_orders = filtered_data['Orders'].mean()
                        st.metric("Average Daily Orders", f"{avg_orders:.0f}")
                    with col3:
                        roi = filtered_data['ROI'].mean()
                        st.metric("Average ROI", f"{roi:.1%}")
                    
                    fig = px.line(filtered_data, x='Date', y='Sales',
                                title='Sales Trend Over Time')
                    st.plotly_chart(fig)
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")

if __name__ == "__main__":
    main()