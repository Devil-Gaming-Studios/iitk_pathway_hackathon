import os
import streamlit as st
import pandas as pd
import altair as alt
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from transformers import pipeline
import time

# Risk detection labels and weights for zero-shot classification
# Used to identify misinformation patterns, conspiracy theories, and sensational content
candidate_labels = [
    "credible factual news",
    "sensational clickbait",
    "misleading or exaggerated news",
    "conspiracy theory",
    "paranormal or UFO claim", 
    "ufo",
    "alien",
    "alien invasion",
    "extraterrestrial",
    "area 51",
    "unidentified flying object",
    "alien contact",
    "predicted alien invasion",
    "shocking",
    "breaking",
    "viral",
    "you won't believe",
    "mind-blowing",
    "must watch",
    "sparks panic",
    "government cover-up",
    "secret files",
    "hidden agenda",
    "classified document",
    "leaked documents",
    "miracle cure",
    "secret formula",
    "instant cure",
    "doctors hate this"
]

# Weight values for each label indicating risk level (0.0 = safe, 1.0 = high risk)
risk_weights = {
    "credible factual news": 0.0,
    "sensational clickbait": 0.4,
    "misleading or exaggerated news": 0.6,
    "paranormal or UFO claim": 1,
    "conspiracy theory": 0.8,
    "ufo": 1.0,
    "alien": 1.0,
    "alien invasion": 1.0,
    "extraterrestrial": 1.0,
    "area 51": 1.0,
    "unidentified flying object": 1.0,
    "alien contact": 1.0,
    "predicted alien invasion": 1.0,
    "shocking": 0.1,
    "breaking": 0.1,
    "viral": 0.4,
    "you won't believe": 0.5,
    "mind-blowing": 0.1,
    "must watch": 0.1,
    "sparks panic": 0.5,
    "government cover-up": 1.0,
    "secret files": 0.8,
    "hidden agenda": 0.9,
    "classified document": 0.8,
    "leaked documents": 0.8,
    "miracle cure": 1.0,
    "secret formula": 1.0,
    "instant cure": 1.0,
    "doctors hate this": 0.6
}

# Calculates misinformation risk score for a given text using weighted label classification
def zero_shot_risk_score(text):
    classifier = get_classifier()
    result = classifier(
        text,
        candidate_labels=candidate_labels,
        hypothesis_template="This news article is {}."
    )
    risk = 0.0
    for label, score in zip(result["labels"], result["scores"]):
        risk += score * risk_weights[label]
    return round(risk * 100, 2)

# Maps risk score percentage to a categorical label with emoji indicator
def risk_category(risk_percentage):
    if risk_percentage <= 10:
        return "✅ Very Low Risk"
    elif risk_percentage <= 30:
        return "🟢 Low Risk"
    elif risk_percentage <= 50:
        return "🟡 Medium Risk"
    elif risk_percentage <= 70:
        return "⚠️ High Risk"
    else:
        return "🚨 Very High Risk"

# Loads and caches the transformer model for zero-shot classification
# Cached to avoid reloading the model on every interaction
@st.cache_resource
def get_classifier():
    return pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Safely loads JSONL files, returns empty DataFrame if file doesn't exist or is empty
def load_jsonl(path: str) -> pd.DataFrame:
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return pd.DataFrame()
    try:
        return pd.read_json(path, lines=True)
    except ValueError:
        return pd.DataFrame()

st.set_page_config(page_title="Misinformation Risk Dashboard", layout="wide", initial_sidebar_state="expanded")

# Define custom CSS styles for dashboard cards and headers
st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        margin-bottom: 20px;
    }
    .stat-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .metric-value {
        font-size: 32px;
        font-weight: bold;
        margin: 10px 0;
    }
    .metric-label {
        font-size: 14px;
        opacity: 0.9;
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown("""
    <div class="main-header">
        <h1>🔥 Misinformation Risk Dashboard</h1>
        <p>Real-time detection of trending fake news and misinformation</p>
    </div>
    """, unsafe_allow_html=True)

# Sidebar configuration and data file paths
st.sidebar.markdown("### ⚙️ Dashboard Settings")
st.sidebar.divider()

# Auto-refresh interval - 2 minutes default
refresh_interval = 600  # 2 minutes

st.sidebar.success(f"✅ Auto-refresh enabled: Every 5 minutes")

# Data file paths for velocity metrics and article details
VELOCITY_FILE = "output/velocity.jsonl"
ARTICLE_FILE = "output/articles.jsonl"

# Load data from JSONL files
velocity_df = load_jsonl(VELOCITY_FILE)
article_df = load_jsonl(ARTICLE_FILE)

# Velocity data is required; stop execution if not available
if velocity_df.empty:
    st.warning("Waiting for velocity data...")
    st.stop()

# Normalize time using _pw_window_start
velocity_df["_pw_window_start"] = pd.to_datetime(
    velocity_df["_pw_window_start"], errors="coerce"
)

# Keep only the most recent time window for each event to get current state
velocity_latest = (
    velocity_df
    .sort_values("_pw_window_start", ascending=False)
    .drop_duplicates(subset="event_id", keep="first")
)

# Sort events by article count in descending order to show trending events first
velocity_latest = velocity_latest.sort_values("article_count", ascending=False)

# Merge article details with velocity metrics (join article text and headline with velocity data)
if not article_df.empty and "event_id" in article_df.columns:
    article_latest = (
        article_df
        .sort_values("timestamp", ascending=False)
        .drop_duplicates(subset="event_id", keep="first")
        [["event_id", "headline", "text", "timestamp"]]
    )
    merged_df = velocity_latest.merge(
        article_latest, on="event_id", how="left"
    )
else:
    merged_df = velocity_latest.copy()
    merged_df["headline"] = "Headline not available"

# Classifies article emotion (fear, anger, sadness, joy, trust) and calculates risk score
# Results are cached to improve performance
@st.cache_data
def classify_emotions(df):
    if df.empty:
        return df
    classifier = get_classifier()
    emotions = []
    emotion_scores_list = []
    risk_scores = []
    for _, row in df.iterrows():
        text = row["text"]
        # Handle missing or empty text with neutral emotion scores
        if pd.isna(text) or text == "":
            emotions.append("unknown")
            emotion_scores_list.append({"fear": 0.2, "anger": 0.2, "sadness": 0.2, "joy": 0.2, "trust": 0.2})
            risk_scores.append(0.0)
        else:
            # Classify dominant emotion from text
            result = classifier(text, candidate_labels=["fear", "anger", "sadness", "joy", "trust"])
            emotions.append(result["labels"][0])
            # Store emotion scores as dictionary
            emotion_scores = {label: score for label, score in zip(result["labels"], result["scores"])}
            emotion_scores_list.append(emotion_scores)
            
            # Risk score calculation
            risk = 0.0
            risk_result = classifier(
                text,
                candidate_labels=candidate_labels,
                hypothesis_template="This news article is {}."
            )
            for label, score in zip(risk_result["labels"], risk_result["scores"]):
                risk += score * risk_weights[label]
            risk_scores.append(round(risk * 100, 2))
    df = df.copy()
    df["emotion"] = emotions
    df["emotion_scores"] = emotion_scores_list
    df["risk_score"] = risk_scores
    return df

if not merged_df.empty:
    merged_df = classify_emotions(merged_df)

# Sort merged data by article count (trending events first)
merged_df = merged_df.sort_values(
    "article_count", ascending=False
)

# Create tabs for different dashboard views
tab1, tab2 = st.tabs(["🏠 Home", "📊 News Trend Analytics"])

with tab1:
    # Check if user selected an event for detailed view
    if "selected_event" in st.session_state:
        selected = st.session_state["selected_event"]
        event_data = merged_df[merged_df["event_id"] == selected].iloc[0]
        rank = merged_df.index.get_loc(event_data.name) + 1
        st.header(f"Trending #{rank}")
        st.subheader(event_data["headline"])
        st.markdown("**Article Description:**")
        st.markdown(f'<p style="font-size: 16px; line-height: 1.6;">{event_data["text"]}</p>', unsafe_allow_html=True)
        
        col_left, col_right = st.columns([0.5, 0.5])
        
        with col_left:
            st.subheader("Emotion")
            emotion = event_data["emotion"]
            # Map emotions to color and emoji for visual representation
            if emotion == "joy":
                emoji = "😄"
                color = "green"
            elif emotion == "sadness":
                emoji = "😭"
                color = "blue"
            elif emotion == "anger":
                emoji = "😡"
                color = "red"
            elif emotion == "fear":
                emoji = "😱"
                color = "orange"
            elif emotion == "trust":
                emoji = "🤝"
                color = "green"
            else:
                emoji = "🤔"
                color = "gray"
            st.markdown(f'<div style="background-color: {color}; color: white; padding: 10px; border-radius: 5px; text-align: center; font-weight: bold; font-size: 18px;">{emoji}<br><small>{emotion.upper()}</small></div>', unsafe_allow_html=True)
            
            # Extract and prepare emotion scores for pie chart visualization
            emotion_scores = event_data["emotion_scores"]
            emotion_df = pd.DataFrame({
                "emotions": list(emotion_scores.keys()),
                "score": list(emotion_scores.values())
            })
            
            emotion_colors = {
                "joy": "#2ECC71",
                "sadness": "#3498DB",
                "anger": "#E74C3C",
                "fear": "#FF9800",
                "trust": "#9B59B6"
            }
            
            fig_emotion = px.pie(
                emotion_df,
                names="emotions",
                values="score",
                hole=0.4,
                color="emotions",
                color_discrete_map=emotion_colors
            )
            fig_emotion.update_layout(height=300, showlegend=True)
            st.plotly_chart(fig_emotion, use_container_width=True)
        
        with col_right:
            st.subheader("Misinformation Risk", anchor=False)
            st.write("")  # Add spacing
            risk_score = event_data["risk_score"]
            risk_cat = risk_category(risk_score)
            
            # Select color based on risk level for visual indicator
            if risk_score <= 10:
                risk_color = "#4CAF50"
            elif risk_score <= 30:
                risk_color = "#8BC34A"
            elif risk_score <= 50:
                risk_color = "#FFC107"
            elif risk_score <= 70:
                risk_color = "#FF9800"
            else:
                risk_color = "#F44336"
            
            st.markdown(f'''
            <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; padding: 20px; margin-top: 10px;">
                <svg width="120" height="120" style="transform: rotate(-90deg);">
                    <circle cx="60" cy="60" r="54" stroke="#E0E0E0" stroke-width="8" fill="none"></circle>
                    <circle cx="60" cy="60" r="54" stroke="{risk_color}" stroke-width="8" fill="none"
                        stroke-dasharray="{339.29 * risk_score / 100}" stroke-dashoffset="0"
                        style="stroke-linecap: round;"></circle>
                </svg>
                <div style="text-align: center; margin-top: -80px;">
                    <div style="font-size: 28px; font-weight: bold;">{risk_score:.0f}%</div>
                    <div style="font-size: 14px; color: #666; margin-top: 35px;">{risk_cat}</div>
                </div>
            </div>
            ''', unsafe_allow_html=True)
        
        st.divider()
        st.subheader("Article Velocity Trend")
        
        # Display time series chart showing article publication rate for selected event
        if not velocity_df.empty:
            event_velocity = velocity_df[velocity_df["event_id"] == selected]
            if not event_velocity.empty:
                # Sort by timestamp and convert to datetime
                event_velocity_sorted = event_velocity.sort_values('_pw_window_start')
                event_velocity_sorted['_pw_window_start'] = pd.to_datetime(event_velocity_sorted['_pw_window_start'], errors='coerce')
                trend_df = event_velocity_sorted[['_pw_window_start', 'article_count']].dropna()
                
                if not trend_df.empty:
                    st.line_chart(trend_df.set_index('_pw_window_start')['article_count'], height=250, use_container_width=True)
        
        if st.button("Back to Trending"):
            del st.session_state["selected_event"]
            st.rerun()
    else:
        # Display trending news list with summary statistics
        st.header("Trending News")
        
        # Display key statistics about trending events
        col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
        
        with col_stat1:
            st.markdown(f"""
            <div class="stat-card">
                <div class="metric-label">Total Trending Events</div>
                <div class="metric-value">{len(merged_df)}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_stat2:
            # Calculate average article count per event
            avg_articles = merged_df['article_count'].mean() if not merged_df.empty else 0
            st.markdown(f"""
            <div class="stat-card">
                <div class="metric-label">Avg Articles/Event</div>
                <div class="metric-value">{avg_articles:.0f}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_stat3:
            max_articles = merged_df['article_count'].max() if not merged_df.empty else 0
            st.markdown(f"""
            <div class="stat-card">
                <div class="metric-label">Peak Articles</div>
                <div class="metric-value">{max_articles:.0f}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_stat4:
            # Count events with high misinformation risk (>70%)
            high_risk = len(merged_df[merged_df['risk_score'] > 70]) if 'risk_score' in merged_df.columns else 0
            st.markdown(f"""
            <div class="stat-card">
                <div class="metric-label">High Risk Events</div>
                <div class="metric-value">{high_risk}</div>
            </div>
            """, unsafe_allow_html=True)
        
        
        st.divider()
        
        # Display column headers for trending news table
        col1, col2, col3, col4 = st.columns([0.05, 0.55, 0.2, 0.2])
        with col1:
            st.markdown("**Rank**")
        with col2:
            st.markdown("**Headline**")
        with col3:
            st.markdown("**Emotion**")
        with col4:
            st.markdown("**Risk**")
        st.divider()  # Optional separator
        for rank, (_, row) in enumerate(merged_df.iterrows(), start=1):
            col1, col2, col3, col4 = st.columns([0.05, 0.55, 0.2, 0.2])
            with col1:
                st.markdown(f'<div style="width: 30px; height: 30px; border-radius: 50%; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; text-align: center; line-height: 30px; font-weight: bold;">{rank}</div>', unsafe_allow_html=True)
            with col2:
                if st.button(row["headline"], key=row["event_id"]):
                    st.session_state["selected_event"] = row["event_id"]
                    st.rerun()
            with col3:
                # Display emotion badge with color and emoji
                emotion = row["emotion"]
                if emotion == "joy":
                    emoji = "😄"
                    color = "green"
                elif emotion == "sadness":
                    emoji = "😭"
                    color = "blue"
                elif emotion == "anger":
                    emoji = "😡"
                    color = "red"
                elif emotion == "fear":
                    emoji = "😱"
                    color = "orange"
                elif emotion == "trust":
                    emoji = "🤝"
                    color = "green"
                else:
                    emoji = "🤔"
                    color = "gray"
                st.markdown(f'<div style="background-color: {color}; color: white; padding: 5px; border-radius: 5px; text-align: center; font-weight: bold;">{emoji}<br><small>{emotion.upper()}</small></div>', unsafe_allow_html=True)
            with col4:
                # Calculate and display risk score for each article in list
                risk_score = zero_shot_risk_score(row["text"])
                if risk_score <= 10:
                    risk_color = "#4CAF50"
                elif risk_score <= 30:
                    risk_color = "#8BC34A"
                elif risk_score <= 50:
                    risk_color = "#FFC107"
                elif risk_score <= 70:
                    risk_color = "#FF9800"
                else:
                    risk_color = "#F44336"
                
                st.markdown(f'''
                <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; padding: 10px;">
                    <svg width="80" height="80" style="transform: rotate(-90deg);">
                        <circle cx="40" cy="40" r="36" stroke="#E0E0E0" stroke-width="6" fill="none"></circle>
                        <circle cx="40" cy="40" r="36" stroke="{risk_color}" stroke-width="6" fill="none"
                            stroke-dasharray="{226.19 * risk_score / 100}" stroke-dashoffset="0"
                            style="stroke-linecap: round;"></circle>
                    </svg>
                    <div style="text-align: center; margin-top: -60px;">
                        <div style="font-size: 18px; font-weight: bold;">{risk_score:.0f}%</div>
                    </div>
                </div>
                ''', unsafe_allow_html=True)
            st.divider()

# Analytics and trend visualization tab
with tab2:
    st.header("📊 News Trend Analytics")
    st.markdown("Visualize how many articles are being published over time for different events")
    
    st.divider()
    
    if not velocity_df.empty:
        # Sort velocity data by timestamp for time series analysis
        velocity_df_sorted = velocity_df.sort_values("_pw_window_start")
        
        # Display aggregate article volume across all events
        st.subheader("Total News Volume Over Time")
        total_trend = velocity_df_sorted.groupby("_pw_window_start")["article_count"].sum().reset_index()
        total_trend.columns = ["Time", "Total Articles"]
        
        if not total_trend.empty:
            fig_total = px.line(
                total_trend,
                x="Time",
                y="Total Articles",
                title="Total Articles Published Over Time",
                markers=True,
                template="plotly_dark"
            )
            fig_total.update_layout(
                hovermode="x unified",
                height=400,
                xaxis_title="Time",
                yaxis_title="Number of Articles"
            )
            st.plotly_chart(fig_total, use_container_width=True)
        
        st.divider()
        
        # Compare article trends across top 5 trending events
        st.subheader("Top 5 Events - Articles Over Time")
        
        # Identify top 5 events by maximum article count
        top_events = velocity_df.groupby("event_id")["article_count"].max().nlargest(5).index.tolist()
        
        # Create multi-line chart comparing top events
        fig_multi = go.Figure()
        
        # Add trace for each top event
        for event_id in top_events:
            event_data = velocity_df[velocity_df["event_id"] == event_id].sort_values("_pw_window_start")
            if not event_data.empty:
                fig_multi.add_trace(go.Scatter(
                    x=event_data["_pw_window_start"],
                    y=event_data["article_count"],
                    mode='lines+markers',
                    name=event_id,
                    hovertemplate='<b>%{fullData.name}</b><br>Time: %{x}<br>Articles: %{y}<extra></extra>'
                ))
        
        fig_multi.update_layout(
            title="Top 5 Events - Article Count Over Time",
            xaxis_title="Time",
            yaxis_title="Number of Articles",
            hovermode="x unified",
            height=450,
            template="plotly_dark",
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        )
        st.plotly_chart(fig_multi, use_container_width=True)
        
        st.divider()
        
        # Allow user to select individual event for detailed trend analysis
        st.subheader("Detailed Event Trend")
        selected_event_analytics = st.selectbox(
            "Select an event to view detailed trend",
            options=velocity_df["event_id"].unique(),
            key="event_selector"
        )
        
        if selected_event_analytics:
            event_trend = velocity_df[velocity_df["event_id"] == selected_event_analytics].sort_values("_pw_window_start")
            
            if not event_trend.empty:
                # Display key metrics for selected event
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    max_count = event_trend["article_count"].max()
                    st.metric("Peak Articles", int(max_count))
                
                with col2:
                    avg_count = event_trend["article_count"].mean()
                    st.metric("Average Articles", f"{avg_count:.1f}")
                
                with col3:
                    total_count = event_trend["article_count"].sum()
                    st.metric("Total Articles", int(total_count))
                
                st.markdown("")
                
                # Create bar chart showing article count trend over time
                fig_detail = px.bar(
                    event_trend,
                    x="_pw_window_start",
                    y="article_count",
                    title=f"Article Trend for Event: {selected_event_analytics}",
                    template="plotly_dark",
                    color="article_count",
                    color_continuous_scale="reds"
                )
                fig_detail.update_layout(
                    height=350,
                    xaxis_title="Time",
                    yaxis_title="Number of Articles",
                    hovermode="x unified"
                )
                st.plotly_chart(fig_detail, use_container_width=True)
    else:
        st.warning("No velocity data available yet. Please wait for the pipeline to generate data.")

# Auto-refresh page at specified interval to show latest data
if refresh_interval > 0:
    # Display countdown timer for next refresh
    refresh_placeholder = st.empty()
    
    # Count down and update display every second
    for remaining in range(refresh_interval, 0, -1):
        with refresh_placeholder.container():
            st.sidebar.info(f"⏱️ Next refresh in {remaining}s...")
        time.sleep(1)
    
    # Trigger page rerun to load latest data
    st.rerun()
