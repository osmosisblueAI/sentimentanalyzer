print("Starting Sentiment Analyzer...")
import tweepy
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import plotly.express as px
import plotly.graph_objects as go
import re
import pandas as pd
import random
from datetime import datetime, timedelta
from flask import Flask, render_template, request, send_file, redirect, url_for, flash, session
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, EmailField, SelectField, SubmitField
from wtforms.validators import DataRequired, Length, Email
from config import BEARER_TOKEN
import json
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from io import BytesIO
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import base64
import sqlite3
from collections import Counter
import spacy
from werkzeug.security import generate_password_hash, check_password_hash
import requests
from wordcloud import WordCloud
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
from email.mime.text import MIMEText
import numpy as np

print("Imports completed, initializing Flask...")
app = Flask(__name__)
app.secret_key = "super_secret_key_123"  # Replace with os.getenv("SECRET_KEY") for production

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"

analyzer = SentimentIntensityAnalyzer()
nlp = spacy.load("en_core_web_sm")

# Email config
EMAIL_ADDRESS = "laje6512@gmail.com"
EMAIL_PASSWORD = "wyzpappoouokrqrp"

class User(UserMixin):
    def __init__(self, id, username, password_hash):
        self.id = id
        self.username = username
        self.password_hash = password_hash

# Flask-WTF Forms
class AnalysisForm(FlaskForm):
    topics = StringField('Topics (comma-separated)', validators=[DataRequired()], default="Bitcoin")
    mode = SelectField('Mode', choices=[('simulated', 'Simulated'), ('live', 'Live (X API)')], default='simulated')
    very_positive = StringField('Very Positive', default="#2ecc71")
    positive = StringField('Positive', default="#66BB6A")
    neutral = StringField('Neutral', default="#FFCA28")
    negative = StringField('Negative', default="#EF5350")
    very_negative = StringField('Very Negative', default="#e74c3c")
    submit = SubmitField('Analyze')

class LoginForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired(), Length(min=3)])
    password = PasswordField('Password', validators=[DataRequired(), Length(min=6)])
    submit = SubmitField('Login')

class SignupForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired(), Length(min=3, max=50)])
    password = PasswordField('Password', validators=[DataRequired(), Length(min=6)])
    email = EmailField('Email', validators=[DataRequired(), Email()])
    submit = SubmitField('Sign Up')

def validate_json(data):
    try:
        json.dumps(data)
        return True
    except Exception as e:
        print(f"Invalid JSON: {e}")
        return False

def init_db():
    conn = sqlite3.connect("sentiment.db")
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS analyses 
                 (id INTEGER PRIMARY KEY, topic TEXT, mode TEXT, total_posts INTEGER, 
                  sentiments TEXT, avg_compound REAL, timestamp TEXT, user_id INTEGER)''')
    c.execute('''CREATE TABLE IF NOT EXISTS feedback 
                 (id INTEGER PRIMARY KEY, feedback TEXT, timestamp TEXT, user_id INTEGER)''')
    c.execute('''CREATE TABLE IF NOT EXISTS users 
                 (id INTEGER PRIMARY KEY, username TEXT UNIQUE, password_hash TEXT, email TEXT, preferences TEXT)''')
    try:
        c.execute("ALTER TABLE users ADD COLUMN preferences TEXT")
    except sqlite3.OperationalError:
        pass
    conn.commit()
    conn.close()

init_db()

@login_manager.user_loader
def load_user(user_id):
    conn = sqlite3.connect("sentiment.db")
    c = conn.cursor()
    c.execute("SELECT id, username, password_hash FROM users WHERE id = ?", (user_id,))
    user_data = c.fetchone()
    conn.close()
    if user_data:
        return User(user_data[0], user_data[1], user_data[2])
    return None

SOLANA_COINS = ["SOL", "JTO", "WIF", "BONK", "RAY", "ORCA", "SBR", "STEP", "MSOL", "KIN"]

def simulate_tweets(num_tweets=100, topic="Bitcoin"):
    print(f"Simulating {num_tweets} tweets for {topic}...")
    templates = [
        f"{topic} pumping hard! 🚀 Time to buy!",
        f"{topic} dumping—sell now or HODL? 😱",
        f"Just swapped some {topic}, market’s wild!",
        f"{topic} breaking resistance at $50k! 📈",
        f"{topic} wallet down again—FUD spreading 👎",
        f"{topic} to $100k by EOY! #HODL",
        f"{topic} trading flat, boring day 😐",
        f"Bullish on {topic}—new ATH soon? 😍",
        f"{topic} team needs to fix this dip 🤬",
        f"Smooth {topic} staking rewards today 👍"
    ]
    tweets = []
    now = datetime.now()
    for _ in range(num_tweets):
        text = random.choice(templates) + f" {random.randint(1, 100)}"
        timestamp = now - timedelta(days=random.uniform(0, 7), seconds=random.randint(0, 86400))
        cleaned_text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
        cleaned_text = re.sub(r"@\w+|\#", "", cleaned_text)
        tweets.append({"text": cleaned_text, "timestamp": timestamp.isoformat()})
    return tweets

def fetch_live_tweets(query, max_results=100):
    print(f"Fetching live tweets for {query}...")
    client = tweepy.Client(bearer_token=BEARER_TOKEN)
    try:
        response = client.search_recent_tweets(query=f"{query} lang:en", max_results=max_results,
                                              tweet_fields=["created_at"])
        posts = response.data if response.data else []
        return [{"text": post.text, "timestamp": post.created_at.isoformat()} for post in posts]
    except tweepy.TweepyException as e:
        print(f"Error fetching live posts: {e}")
        return []

def fetch_crypto_price(topic, days=7):
    print(f"Fetching price data for {topic}...")
    crypto_map = {
        "Bitcoin": "bitcoin",
        "Ethereum": "ethereum",
        "SOL": "solana",
        "JTO": "jito-governance-token",
        "WIF": "dogwifhat",
        "BONK": "bonk",
        "RAY": "raydium",
        "ORCA": "orca",
        "SBR": "saber",
        "STEP": "step-finance",
        "MSOL": "marinade-staked-sol",
        "KIN": "kin"
    }
    crypto_id = crypto_map.get(topic, topic.lower())
    url = f"https://api.coingecko.com/api/v3/coins/{crypto_id}/market_chart?vs_currency=usd&days={days}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        prices = pd.DataFrame(data["prices"], columns=["timestamp", "price"])
        prices["timestamp"] = pd.to_datetime(prices["timestamp"], unit="ms").dt.isoformat()
        return prices
    print(f"Error fetching price data: {response.status_code} - {response.text}")
    return pd.DataFrame()

def analyze_sentiment(tweets):
    print("Analyzing sentiment...")
    sentiments = {"Very Positive": 0, "Positive": 0, "Neutral": 0, "Negative": 0, "Very Negative": 0}
    sentiment_scores = []
    for tweet in tweets:
        vs = analyzer.polarity_scores(tweet["text"])
        compound = vs["compound"]
        if compound >= 0.5:
            sentiments["Very Positive"] += 1
            label = "Very Positive"
        elif compound >= 0.05:
            sentiments["Positive"] += 1
            label = "Positive"
        elif compound <= -0.5:
            sentiments["Very Negative"] += 1
            label = "Very Negative"
        elif compound <= -0.05:
            sentiments["Negative"] += 1
            label = "Negative"
        else:
            sentiments["Neutral"] += 1
            label = "Neutral"
        sentiment_scores.append({
            "text": tweet["text"],
            "timestamp": tweet["timestamp"],
            "compound": compound if not np.isnan(compound) else 0,
            "sentiment": label
        })
    return sentiments, sentiment_scores

def generate_insights(tweets):
    print("Generating NLP insights...")
    pos_words, neg_words = [], []
    for tweet in tweets:
        doc = nlp(tweet["text"])
        vs = analyzer.polarity_scores(tweet["text"])
        for token in doc:
            if token.pos_ in ["ADJ", "NOUN", "VERB"] and not token.is_stop:
                if vs["compound"] >= 0.05:
                    pos_words.append(token.text.lower())
                elif vs["compound"] <= -0.05:
                    neg_words.append(token.text.lower())
    top_pos = Counter(pos_words).most_common(5)
    top_neg = Counter(neg_words).most_common(5)
    return top_pos, top_neg

def generate_word_cloud(tweets, topic):
    print(f"Generating word cloud for {topic}...")
    text = " ".join(tweet["text"] for tweet in tweets)
    wordcloud = WordCloud(width=800, height=400, background_color="white", colormap="viridis").generate(text)
    plt.figure(figsize=(8, 4))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    img_buffer = BytesIO()
    plt.savefig(img_buffer, format="png", bbox_inches="tight")
    plt.close()
    img_buffer.seek(0)
    return base64.b64encode(img_buffer.getvalue()).decode("utf-8")

def generate_trend_chart(tweets, topic):
    print(f"Generating trend chart for {topic}...")
    df = pd.DataFrame([{"timestamp": t["timestamp"], "compound": analyzer.polarity_scores(t["text"])["compound"]} for t in tweets])
    df["date"] = pd.to_datetime(df["timestamp"]).dt.floor("h")
    trend = df.groupby("date")["compound"].mean().reset_index()
    trend_fig = px.line(trend, x="date", y="compound", title=f"Sentiment Trend: {topic}")
    trend_fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Avg Sentiment Score",
        yaxis=dict(showgrid=True, gridcolor="lightgray", griddash="dash"),
        xaxis=dict(rangeslider=dict(visible=True), type="date", autorange=True)
    )
    img_buffer = BytesIO()
    trend_fig.write_image(img_buffer, format="png")
    img_buffer.seek(0)
    return img_buffer

def generate_output(tweets, topic, colors, mode):
    print(f"Generating output for {topic}...")
    sentiments, sentiment_scores = analyze_sentiment(tweets)
    total_posts = sum(sentiments.values())
    percentages = [f"{(value / total_posts) * 100:.1f}%" for value in sentiments.values()]
    avg_compound = sum(score["compound"] for score in sentiment_scores) / total_posts if total_posts else 0
    top_pos, top_neg = generate_insights(tweets)

    color_map = {
        "Very Positive": colors.get("very_positive", "#2ecc71"),
        "Positive": colors.get("positive", "#66BB6A"),
        "Neutral": colors.get("neutral", "#FFCA28"),
        "Negative": colors.get("negative", "#EF5350"),
        "Very Negative": colors.get("very_negative", "#e74c3c")
    }

    # Bar Chart
    bar_fig = px.bar(x=list(sentiments.keys()), y=list(sentiments.values()),
                     color=list(sentiments.keys()), color_discrete_map=color_map,
                     labels={"x": "Sentiment", "y": "Number of Posts"},
                     title=f"Current Sentiment: {topic}")
    bar_fig.update_traces(text=percentages, textposition="auto")
    bar_fig.update_layout(showlegend=True, bargap=0.2, plot_bgcolor="white",
                          yaxis=dict(showgrid=True, gridcolor="lightgray", griddash="dash"))
    bar_json = bar_fig.to_json()
    if not validate_json(bar_json):
        print("Bar JSON invalid, using fallback:", bar_json)
        bar_json = json.dumps({"data": [{"type": "bar", "x": [], "y": []}], "layout": {}})

    # Pie Chart
    pie_fig = px.pie(values=list(sentiments.values()), names=list(sentiments.keys()),
                     title=f"Sentiment Distribution: {topic}", color_discrete_map=color_map)
    pie_json = pie_fig.to_json()
    if not validate_json(pie_json):
        print("Pie JSON invalid, using fallback:", pie_json)
        pie_json = json.dumps({"data": [{"type": "pie", "values": [], "labels": []}], "layout": {}})

    # Trend Chart
    df = pd.DataFrame(sentiment_scores)
    df["date"] = pd.to_datetime(df["timestamp"]).dt.floor("h").apply(lambda x: x.isoformat())
    trend = df.groupby("date")["compound"].mean().reset_index()
    trend_fig = px.line(trend, x="date", y="compound", title=f"Sentiment Trend (Hourly): {topic}")
    trend_fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Avg Sentiment Score",
        yaxis=dict(showgrid=True, gridcolor="lightgray", griddash="dash"),
        xaxis=dict(rangeslider=dict(visible=True), type="date", autorange=True)
    )
    trend_json = trend_fig.to_json()
    if not validate_json(trend_json):
        print("Trend JSON invalid, using fallback:", trend_json)
        trend_json = json.dumps({"data": [{"type": "scatter", "x": [], "y": []}], "layout": {}})

    # Histogram
    hist_fig = px.histogram(df, x="compound", color="sentiment", nbins=20,
                            title=f"Sentiment Score Distribution: {topic}",
                            color_discrete_map=color_map)
    hist_fig.update_layout(bargap=0.1, plot_bgcolor="white",
                           yaxis=dict(showgrid=True, gridcolor="lightgray", griddash="dash"))
    hist_json = hist_fig.to_json()
    if not validate_json(hist_json):
        print("Histogram JSON invalid, using fallback:", hist_json)
        hist_json = json.dumps({"data": [{"type": "histogram", "x": []}], "layout": {}})

    # Heatmap
    heatmap_fig = px.density_heatmap(df, x="date", y="sentiment", z="compound",
                                     title=f"Sentiment Density Heatmap: {topic}",
                                     color_continuous_scale="Viridis")
    heatmap_fig.update_layout(plot_bgcolor="white",
                              yaxis=dict(showgrid=True, gridcolor="lightgray", griddash="dash"))
    heatmap_json = heatmap_fig.to_json()
    if not validate_json(heatmap_json):
        print("Heatmap JSON invalid, using fallback:", heatmap_json)
        heatmap_json = json.dumps({"data": [{"type": "heatmap", "x": [], "y": [], "z": []}], "layout": {}})

    # Correlation Charts
    price_df = fetch_crypto_price(topic)
    correlation_insights = {}
    signals = []
    correlation_jsons = {}
    if not price_df.empty:
        sentiment_df = trend.rename(columns={"compound": "sentiment"})
        merged_df = pd.merge_asof(sentiment_df.sort_values("date"), price_df.sort_values("timestamp"),
                                  left_on="date", right_on="timestamp", direction="nearest")
        timeframes = {"1h": 1, "24h": 24, "7d": 168}
        for timeframe, hours in timeframes.items():
            merged_df[f"sentiment_shift_{timeframe}"] = merged_df["sentiment"].shift(hours)
            merged_df[f"price_change_{timeframe}"] = merged_df["price"].pct_change(periods=hours) * 100
            correlation = merged_df[[f"sentiment_shift_{timeframe}", f"price_change_{timeframe}"]].corr().iloc[0, 1]
            significant_moves = merged_df[merged_df[f"price_change_{timeframe}"].abs() > 5]
            if not significant_moves.empty:
                lead_count = len(significant_moves[significant_moves[f"sentiment_shift_{timeframe}"].abs() > 0.3])
                prob = (lead_count / len(significant_moves)) * 100 if len(significant_moves) > 0 else 0
                correlation_insights[timeframe] = f"Sentiment led {lead_count}/{len(significant_moves)} >5% moves (Prob: {prob:.0f}%). Corr: {correlation:.2f}"
                latest = merged_df.iloc[-1]
                if latest[f"sentiment_shift_{timeframe}"] > 0.3 and latest[f"price_change_{timeframe}"] > 0:
                    signals.append(f"{timeframe} Buy Signal: Sentiment +{latest[f'sentiment_shift_{timeframe}']:.2f} led {latest[f'price_change_{timeframe}']:.1f}% rise")
                elif latest[f"sentiment_shift_{timeframe}"] < -0.3 and latest[f"price_change_{timeframe}"] < 0:
                    signals.append(f"{timeframe} Sell Signal: Sentiment {latest[f'sentiment_shift_{timeframe}']:.2f} led {latest[f'price_change_{timeframe}']:.1f}% drop")
            else:
                correlation_insights[timeframe] = f"No significant moves (>5%) in {timeframe}. Corr: {correlation:.2f}"
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=merged_df["date"], y=merged_df["sentiment"], name="Sentiment", yaxis="y1"))
            fig.add_trace(go.Scatter(x=merged_df["date"], y=merged_df["price"], name="Price (USD)", yaxis="y2", line=dict(color="orange")))
            fig.update_layout(
                title=f"{timeframe} Sentiment vs. Price: {topic}",
                xaxis_title="Date",
                yaxis=dict(title="Sentiment Score", side="left"),
                yaxis2=dict(title="Price (USD)", side="right", overlaying="y"),
                plot_bgcolor="white",
                xaxis=dict(rangeslider=dict(visible=True), type="date", autorange=True)
            )
            correlation_jsons[timeframe] = fig.to_json()
            if not validate_json(correlation_jsons[timeframe]):
                print(f"{timeframe} Correlation JSON invalid, using fallback:", correlation_jsons[timeframe])
                correlation_jsons[timeframe] = json.dumps({"data": [{"type": "scatter", "x": [], "y": []}], "layout": {}})
    else:
        correlation_insights = {"1h": "Price data unavailable.", "24h": "Price data unavailable.", "7d": "Price data unavailable."}
        correlation_jsons = {}
        signals = ["Price data unavailable—no signals generated."]

    wordcloud_img = generate_word_cloud(tweets, topic)

    conn = sqlite3.connect("sentiment.db")
    c = conn.cursor()
    user_id = current_user.id if current_user.is_authenticated else None
    c.execute("INSERT INTO analyses (topic, mode, total_posts, sentiments, avg_compound, timestamp, user_id) VALUES (?, ?, ?, ?, ?, ?, ?)",
              (topic, "simulated" if mode == "simulated" else "live", total_posts, json.dumps(sentiments),
               avg_compound, datetime.now().isoformat(), user_id))
    conn.commit()
    conn.close()

    df.to_csv("static/sentiment_data.csv", index=False)

    return (sentiments, percentages, avg_compound, total_posts, bar_json, pie_json, trend_json,
            hist_json, heatmap_json, correlation_jsons, wordcloud_img, top_pos, top_neg, correlation_insights, signals)

def generate_weekly_report(mode="simulated"):
    print("Generating weekly report data...")
    report_data = []
    trend_images = {}
    for coin in SOLANA_COINS:
        tweets = fetch_live_tweets(coin) if mode == "live" else simulate_tweets(topic=coin)
        if not tweets:
            continue
        sentiment_score, sentiment_scores = analyze_sentiment(tweets)
        total_posts = len(tweets)
        avg_compound = sum(s["compound"] for s in sentiment_scores) / total_posts if total_posts else 0
        sentiment_score = ((avg_compound + 1) / 2) * 100
        price_df = fetch_crypto_price(coin)
        correlation_insight = "Price data unavailable."
        if not price_df.empty:
            df = pd.DataFrame(sentiment_scores)
            df["date"] = [t["timestamp"] for t in tweets]
            df["date"] = pd.to_datetime(df["date"]).dt.floor("h").apply(lambda x: x.isoformat())
            trend = df.groupby("date")["compound"].mean().reset_index()
            sentiment_df = trend.rename(columns={"compound": "sentiment"})
            merged_df = pd.merge_asof(sentiment_df.sort_values("date"), price_df.sort_values("timestamp"),
                                      left_on="date", right_on="timestamp", direction="nearest")
            if len(merged_df) > 24:
                merged_df["sentiment_shift_24h"] = merged_df["sentiment"].shift(24)
                merged_df["price_change_24h"] = merged_df["price"].pct_change(periods=24) * 100
                correlation = merged_df[["sentiment_shift_24h", "price_change_24h"]].corr().iloc[0, 1]
                significant_moves = merged_df[merged_df["price_change_24h"].abs() > 5]
                lead_count = len(significant_moves[significant_moves["sentiment_shift_24h"].abs() > 0.3])
                prob = (lead_count / len(significant_moves)) * 100 if len(significant_moves) > 0 else 0
                correlation_insight = f"Sentiment led {lead_count}/{len(significant_moves)} >5% moves (Prob: {prob:.0f}%). Corr: {correlation:.2f}"
        report_data.append({
            "coin": coin,
            "sentiment_score": sentiment_score,
            "total_posts": total_posts,
            "correlation_insight": correlation_insight
        })
        trend_images[coin] = generate_trend_chart(tweets, coin)
    report_data = sorted(report_data, key=lambda x: x["sentiment_score"], reverse=True)[:10]
    return report_data, trend_images

def generate_pdf_report(report_data, trend_images):
    print("Generating weekly PDF report...")
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=50, bottomMargin=50)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("Weekly Solana Ecosystem Sentiment Report", styles["Title"]))
    story.append(Spacer(1, 12))
    story.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d')}", styles["Normal"]))
    story.append(Spacer(1, 300))
    story.append(Paragraph("Powered by xAI", styles["Normal"]))
    story.append(PageBreak())

    table_data = [["Rank", "Coin", "Sentiment Score (0-100)", "Total Posts", "24h Price Correlation Insight"]]
    for i, data in enumerate(report_data, 1):
        table_data.append([
            str(i),
            data["coin"],
            f"{data['sentiment_score']:.1f}",
            str(data["total_posts"]),
            data["correlation_insight"]
        ])

    table = Table(table_data, colWidths=[50, 50, 100, 70, 250])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.white),
        ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 10),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey])
    ]))
    story.append(table)
    story.append(Spacer(1, 24))

    for data in report_data:
        story.append(Paragraph(f"{data['coin']} Sentiment Trend", styles["Heading2"]))
        trend_img = Image(trend_images[data["coin"]], width=500, height=250)
        story.append(trend_img)
        story.append(Spacer(1, 12))

    def add_footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.drawString(50, 30, f"Page {doc.page} - Powered by xAI")
        canvas.restoreState()

    doc.build(story, onFirstPage=add_footer, onLaterPages=add_footer)
    buffer.seek(0)
    return buffer

def send_email_report(pdf_buffer, recipient_emails):
    print("Sending weekly report via email...")
    msg = MIMEMultipart()
    msg["From"] = EMAIL_ADDRESS
    msg["To"] = ", ".join(recipient_emails)
    msg["Subject"] = "Weekly Solana Sentiment Report"

    body = "Attached is your weekly Solana Ecosystem Sentiment Report."
    msg.attach(MIMEText(body, "plain"))

    pdf_attachment = MIMEApplication(pdf_buffer.getvalue(), _subtype="pdf")
    pdf_attachment.add_header("Content-Disposition", "attachment", filename="solana_sentiment_report.pdf")
    msg.attach(pdf_attachment)

    with smtplib.SMTP("smtp.gmail.com", 587) as server:
        server.starttls()
        server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
        server.send_message(msg)

# Flask Routes
@app.route("/login", methods=["GET", "POST"])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        username = form.username.data
        password = form.password.data
        conn = sqlite3.connect("sentiment.db")
        c = conn.cursor()
        c.execute("SELECT id, username, password_hash FROM users WHERE username = ?", (username,))
        user_data = c.fetchone()
        conn.close()
        if user_data and check_password_hash(user_data[2], password):
            user = User(user_data[0], user_data[1], user_data[2])
            login_user(user)
            return redirect(url_for("index"))
        flash("Invalid username or password")
    return render_template("login.html", form=form)

@app.route("/signup", methods=["GET", "POST"])
def signup():
    form = SignupForm()
    if form.validate_on_submit():
        username = form.username.data
        password = form.password.data
        email = form.email.data
        password_hash = generate_password_hash(password)
        conn = sqlite3.connect("sentiment.db")
        c = conn.cursor()
        try:
            c.execute("INSERT INTO users (username, password_hash, email) VALUES (?, ?, ?)", (username, password_hash, email))
            conn.commit()
            flash("Account created! Please log in.")
            return redirect(url_for("login"))
        except sqlite3.IntegrityError:
            flash("Username already exists.")
        finally:
            conn.close()
    return render_template("signup.html", form=form)

@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("login"))

@app.route("/", methods=["GET", "POST"])
@login_required
def index():
    form = AnalysisForm()
    conn = sqlite3.connect("sentiment.db")
    c = conn.cursor()
    c.execute("SELECT preferences FROM users WHERE id = ?", (current_user.id,))
    prefs = c.fetchone()
    default_colors = {"very_positive": "#2ecc71", "positive": "#66BB6A", "neutral": "#FFCA28", "negative": "#EF5350", "very_negative": "#e74c3c"}
    if prefs and prefs[0]:
        form_data = json.loads(prefs[0])
        form.very_positive.data = form_data.get("very_positive", "#2ecc71")
        form.positive.data = form_data.get("positive", "#66BB6A")
        form.neutral.data = form_data.get("neutral", "#FFCA28")
        form.negative.data = form_data.get("negative", "#EF5350")
        form.very_negative.data = form_data.get("very_negative", "#e74c3c")
    if form.validate_on_submit():
        topics = form.topics.data.split(",")
        mode = form.mode.data
        colors = {
            "very_positive": form.very_positive.data,
            "positive": form.positive.data,
            "neutral": form.neutral.data,
            "negative": form.negative.data,
            "very_negative": form.very_negative.data
        }
        c.execute("UPDATE users SET preferences = ? WHERE id = ?", (json.dumps(colors), current_user.id))
        conn.commit()
        results = []
        for topic in topics:
            topic = topic.strip()
            tweets = fetch_live_tweets(topic) if mode == "live" else simulate_tweets(topic=topic)
            if not tweets:
                flash(f"No data available for {topic}.")
                return redirect(url_for("index"))
            output = generate_output(tweets, topic, colors, mode)
            results.append({
                "topic": topic, "mode": mode, "total_posts": output[3],
                "sentiments": output[0], "percentages": output[1], "avg_compound": output[2],
                "bar_json": output[4], "pie_json": output[5], "trend_json": output[6],
                "hist_json": output[7], "heatmap_json": output[8], "correlation_jsons": output[9],
                "wordcloud_img": output[10], "top_pos": output[11], "top_neg": output[12],
                "correlation_insights": output[13], "signals": output[14]
            })
        with open("static/report.json", "w") as f:
            json.dump(results, f)
        return render_template("results.html", results=results)
    conn.close()
    return render_template("index.html", form=form, coins=SOLANA_COINS + ["Bitcoin", "Ethereum"])

@app.route("/weekly_report", methods=["GET", "POST"])
@login_required
def weekly_report():
    report_data = []
    if request.method == "POST":
        mode = request.form["mode"]
        report_data, trend_images = generate_weekly_report(mode)
        pdf_buffer = generate_pdf_report(report_data, trend_images)
        with open("static/weekly_report.pdf", "wb") as f:
            f.write(pdf_buffer.getvalue())
        
        conn = sqlite3.connect("sentiment.db")
        c = conn.cursor()
        c.execute("SELECT email FROM users WHERE email IS NOT NULL")
        subscriber_emails = [row[0] for row in c.fetchall()]
        conn.close()
        
        if subscriber_emails:
            try:
                send_email_report(pdf_buffer, subscriber_emails)
                flash("Report generated and emailed to subscribers!")
            except smtplib.SMTPAuthenticationError:
                flash("Report generated, but email failed due to authentication error. Check credentials.")
        else:
            flash("Report generated, but no subscribers found.")
    return render_template("weekly_report.html", report_data=report_data)

@app.route("/download_pdf")
@login_required
def download_pdf():
    return send_file("static/weekly_report.pdf", as_attachment=True, download_name="solana_sentiment_report.pdf")

if __name__ == "__main__":
    print("Starting Flask server on port 5001...")
    app.run(port=5001, debug=True)