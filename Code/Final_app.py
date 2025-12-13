"""
Financial News Multi-Model Sentiment Analyzer

Advanced NLP application demonstrating comparative sentiment analysis across multiple
pre-trained transformer models with comprehensive evaluation metrics.

Course: Natural Language Processing in Data Science
Author: Reza Tashakkori
Date: December 2025

=== KEY FEATURES ===
1. Multi-Model Comparison:
   - FinBERT (ProsusAI) - Finance-specialized BERT
   - FinBERT-Tone (yiyanghkust) - Financial PhraseBank fine-tuned
   - BERT (nlptown) - Multilingual sentiment (5-star scale)
   - DistilBERT-SST - Stanford Sentiment Treebank fine-tuned

2. Comprehensive Evaluation:
   - Accuracy, Precision, Recall, F1-Score
   - Confusion matrices per model
   - Model agreement analysis
   - Confidence distribution visualization

3. Text Summarization:
   - BART (Facebook) - CNN/DailyMail fine-tuned
   - T5 (Google) - Text-to-Text Transfer Transformer
   - Pegasus (Google) - XSum fine-tuned
   - Performance metrics: length, abstractiveness, coverage

4. Interactive Visualizations:
   - Process flowchart of NLP pipeline
   - Sentiment distribution charts (bar, pie)
   - Box plots with statistical indicators
   - Agreement heatmaps
   - Stock price integration (demo)

5. Flexible Input Options:
   - Sample headlines for quick testing
   - Custom headline input
   - Labeled dataset upload (CSV/Excel) for evaluation
   - 10 pre-labeled samples included

=== TECHNICAL STACK ===
- Framework: Streamlit (interactive web UI)
- NLP: Hugging Face Transformers (7 models total)
- ML Backend: PyTorch
- Evaluation: scikit-learn metrics
- Visualization: Plotly (interactive charts)
- Financial Data: Yahoo Finance API

=== HOW TO RUN ===
From project root directory:
    python -m streamlit run Code\Final_app.py

Or with Python launcher:
    py -m streamlit run Code\Final_app.py

=== PREREQUISITES ===
Python 3.12+
All dependencies: pip install -r requirements.txt

=== REPRODUCIBILITY NOTES ===
- All models cached after first download (~8-12GB total)
- Cross-platform compatible (Windows/Linux/macOS)
- No API keys required for basic functionality
- Sample data included for immediate testing
"""

from __future__ import annotations

# =============================================================================
# CRITICAL: TensorFlow DLL Error Prevention (Windows Compatibility)
# =============================================================================
# Issue: Windows + Python 3.12 + transformers library attempts to import
#        TensorFlow by default, which has broken DLL dependencies on Windows.
#
# Solution: Force transformers to use PyTorch exclusively by setting environment
#           variables BEFORE any imports. These must be set prior to loading the
#           transformers library to prevent TensorFlow initialization.
#
# Variables:
#   TRANSFORMERS_NO_TF="1" - Disables TensorFlow backend in transformers
#   USE_TF="0"             - Secondary flag to prevent TF usage
#   TF_CPP_MIN_LOG_LEVEL="3" - Suppresses TensorFlow warning messages
#
# Result: Application runs with PyTorch only (stable on Windows)
# =============================================================================

import os
import sys

# Set environment variables before any other imports
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["USE_TF"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


from datetime import datetime
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import yfinance as yf
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import torch
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support

# -----------------------------------------------------------------------------
# Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
	page_title="Financial Sentiment Analyzer",
	page_icon="🎯",
	layout="wide",
	initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# Helper Functions: Model Loading
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_sentiment_models() -> Dict[str, Tuple]:
	"""Load multiple sentiment models for comparison."""
	models_dict = {}
	
	# Model 1: FinBERT (specialized for financial text)  maps 0=positive, 1=negative, 2=neutral.
	finbert_name = "ProsusAI/finbert"
	finbert_tok = AutoTokenizer.from_pretrained(finbert_name)
	finbert_model = AutoModelForSequenceClassification.from_pretrained(finbert_name)
	finbert_model.eval()
	models_dict["FinBERT"] = (finbert_tok, finbert_model, finbert_model.config.id2label)
	
	# Model 2: FinBERT-tone (alternative financial model) maps 0=neutral, 1=positive, 2=negative
	fintone_name = "yiyanghkust/finbert-tone"
	fintone_tok = AutoTokenizer.from_pretrained(fintone_name)
	fintone_model = AutoModelForSequenceClassification.from_pretrained(fintone_name)
	fintone_model.eval()
	models_dict["FinBERT-Tone"] = (fintone_tok, fintone_model, fintone_model.config.id2label)
	
	# Model 3: BERT (base general model) 5-class (1-5 stars);  4-5=positive, 1-2=negative, 3=neutral
	bert_name = "nlptown/bert-base-multilingual-uncased-sentiment"
	bert_tok = AutoTokenizer.from_pretrained(bert_name)
	bert_model = AutoModelForSequenceClassification.from_pretrained(bert_name)
	bert_model.eval()
	models_dict["BERT"] = (bert_tok, bert_model, bert_model.config.id2label)
	
	# Model 4: DistilBERT (general sentiment) - Binary {0: 'NEGATIVE', 1: 'POSITIVE'}; neutral if low confidence (probs balanced)
	distil_name = "distilbert-base-uncased-finetuned-sst-2-english"
	distil_tok = AutoTokenizer.from_pretrained(distil_name)
	distil_model = AutoModelForSequenceClassification.from_pretrained(distil_name)
	distil_model.eval()
	models_dict["DistilBERT-SST"] = (distil_tok, distil_model, distil_model.config.id2label)
	
	return models_dict

@st.cache_resource(show_spinner=False)
def load_summarizers():
	"""Load multiple summarization models for comparison."""
	summarizers = {}
	
	# Model 1: BART
	summarizers["BART"] = pipeline("summarization", model="facebook/bart-large-cnn")
	
	# Model 2: T5
	summarizers["T5"] = pipeline("summarization", model="t5-small")
	
	# Model 3: Pegasus
	summarizers["Pegasus"] = pipeline("summarization", model="google/pegasus-xsum")
	
	return summarizers

# -----------------------------------------------------------------------------
# Helper Functions: Sentiment Analysis
# -----------------------------------------------------------------------------
def analyze_single_text(
	text: str,
	tokenizer: AutoTokenizer,
	model: AutoModelForSequenceClassification,
	id2label: Dict[int, str]
) -> Dict[str, float]:
    
	"""Analyze sentiment of a single text."""
 
    # Handle empty input
	if not text or not text.strip():
		return {"positive": 0.0, "negative": 0.0, "neutral": 1.0, "confidence": 0.0, "label": "neutral"}
	
	inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True) # Tokenization
    # truncation and max_length added to avoid errors with long texts more than 512 tokens
    # padding to make sure inputs are of uniform length
	
    # Model Prediction
    # We're doing inference only (predicting), not training. 
    # Gradients are only needed for training to update model weights via backpropagation
	with torch.no_grad():   # Disable gradeint calculation for faster inference and save memory
		outputs = model(**inputs) # Forward pass - Feeds tokens into model
		probs = torch.nn.functional.softmax(outputs.logits, dim=-1).detach().cpu().numpy()[0]
        # Streamlit and Python libraries (like Plotly, Pandas) work with CPU data, 
        # so we move tensors to CPU and convert to numpy arrays.
	
	# Map probabilities to standard labels
	label_probs = {"positive": 0.0, "negative": 0.0, "neutral": 0.0}
	
	for i, prob in enumerate(probs):
		label = id2label.get(i, str(i)).lower()
		# Normalize labels (POSITIVE/positive -> positive)
  
		# For models with 5-star ratings (like BERT), aggregate appropriately
		if "5" in label or "4" in label or "pos" in label:    # (very good/ good -> positive)
			label_probs["positive"] += float(prob)
		elif "1" in label or "2" in label or "neg" in label:  # (very bad/ bad -> negative)
			label_probs["negative"] += float(prob)
		else:
			label_probs["neutral"] += float(prob) # (3-star/ neutral)
	
	# Normalize to ensure sum = 1.0 (important for BERT and similar models)
	total_prob = sum(label_probs.values())
	if total_prob > 0:
		label_probs = {k: v/total_prob for k, v in label_probs.items()}
	
	# Find dominant sentiment with highest probability
	dominant = max(label_probs, key=label_probs.get)
	confidence = label_probs[dominant]
	
	return {
		"positive": label_probs["positive"],
		"negative": label_probs["negative"],
		"neutral": label_probs["neutral"],
		"confidence": confidence,
		"label": dominant
	}
    
    # Example Output:
    # {
    # "positive": 0.67,
    # "negative": 0.19,
    # "neutral": 0.14,
    # "confidence": 0.67,
    # "label": "positive"
    # }


def analyze_with_all_models(headlines: List[str], models_dict: Dict) -> pd.DataFrame:
	"""Run all models on headlines and return comparative results."""
	all_results = []
	
	for idx, headline in enumerate(headlines):
		row = {"ID": idx + 1, "Headline": headline}
		
		for model_name, (tokenizer, model, id2label) in models_dict.items():
			result = analyze_single_text(headline, tokenizer, model, id2label)
			row[f"{model_name}_Label"] = result["label"]
			row[f"{model_name}_Confidence"] = result["confidence"]
			row[f"{model_name}_Pos"] = result["positive"]
			row[f"{model_name}_Neg"] = result["negative"]
			row[f"{model_name}_Neu"] = result["neutral"]
		
		all_results.append(row)
	
	return pd.DataFrame(all_results)

# -----------------------------------------------------------------------------
# Helper Functions: Visualizations
# -----------------------------------------------------------------------------
def create_process_flowchart():
	"""Create a flowchart showing the NLP pipeline."""
	fig = go.Figure()
	
	# Define flowchart nodes
	nodes = [
		{"text": "Input:\nFinancial Headlines", "x": 0.5, "y": 1.0, "color": "#3498db"},
		{"text": "Tokenization", "x": 0.5, "y": 0.88, "color": "#9b59b6"},
		{"text": "FinBERT", "x": 0.15, "y": 0.73, "color": "#2ecc71"},
		{"text": "FinBERT-Tone", "x": 0.38, "y": 0.73, "color": "#2ecc71"},
		{"text": "BERT", "x": 0.62, "y": 0.73, "color": "#2ecc71"},
		{"text": "DistilBERT", "x": 0.85, "y": 0.73, "color": "#2ecc71"},
		{"text": "Sentiment\nClassification", "x": 0.5, "y": 0.58, "color": "#e74c3c"},
		{"text": "Model Comparison\n& Agreement", "x": 0.5, "y": 0.43, "color": "#f39c12"},
		{"text": "Evaluation\nMetrics", "x": 0.35, "y": 0.28, "color": "#16a085"},
		{"text": "Summarization:\nBART, T5, Pegasus", "x": 0.65, "y": 0.28, "color": "#8e44ad"},
		{"text": "Dashboard\n& Insights", "x": 0.5, "y": 0.13, "color": "#34495e"}
	]
	
	# Add nodes
	for node in nodes:
		fig.add_shape(
			type="rect",
			x0=node["x"]-0.09, y0=node["y"]-0.04,
			x1=node["x"]+0.09, y1=node["y"]+0.04,
			fillcolor=node["color"],
			line=dict(color="white", width=2)
		)
		fig.add_annotation(
			x=node["x"], y=node["y"],
			text=node["text"],
			showarrow=False,
			font=dict(color="white", size=9, family="Arial", weight="bold"),
			align="center"
		)
	
	# Add arrows
	arrows = [
		(0.5, 0.96, 0.5, 0.92),   # Input -> Tokenization
		(0.5, 0.84, 0.15, 0.77),  # Tokenization -> FinBERT
		(0.5, 0.84, 0.38, 0.77),  # Tokenization -> FinBERT-Tone
		(0.5, 0.84, 0.62, 0.77),  # Tokenization -> BERT
		(0.5, 0.84, 0.85, 0.77),  # Tokenization -> DistilBERT
		(0.15, 0.69, 0.5, 0.62),  # FinBERT -> Classification
		(0.38, 0.69, 0.5, 0.62),  # FinBERT-Tone -> Classification
		(0.62, 0.69, 0.5, 0.62),  # BERT -> Classification
		(0.85, 0.69, 0.5, 0.62),  # DistilBERT -> Classification
		(0.5, 0.54, 0.5, 0.47),   # Classification -> Comparison
		(0.5, 0.39, 0.35, 0.32),  # Comparison -> Metrics
		(0.5, 0.39, 0.65, 0.32),  # Comparison -> Summarization
		(0.35, 0.24, 0.5, 0.17),  # Metrics -> Dashboard
		(0.65, 0.24, 0.5, 0.17),  # Summarization -> Dashboard
	]
	
	for x0, y0, x1, y1 in arrows:
		fig.add_annotation(
			x=x1, y=y1, ax=x0, ay=y0,
			xref="x", yref="y", axref="x", ayref="y",
			showarrow=True,
			arrowhead=2, arrowsize=1, arrowwidth=2, arrowcolor="#95a5a6"
		)
	
	fig.update_layout(
		title="NLP Pipeline Process Flow",
		xaxis=dict(visible=False, range=[0, 1]),
		yaxis=dict(visible=False, range=[0, 1.05]),
		height=650,
		showlegend=False,
		plot_bgcolor="rgba(0,0,0,0)",
		paper_bgcolor="rgba(0,0,0,0)"
	)
	
	return fig

def plot_model_comparison(df: pd.DataFrame, models_list: List[str]):
	"""Create comparison visualization for models with counts and percentages."""
	# Count sentiment predictions by model
	sentiment_counts = {}
	for model in models_list:
		col = f"{model}_Label"
		if col in df.columns:
			counts = df[col].value_counts()
			sentiment_counts[model] = counts
	
	# Create grouped bar chart with percentages
	sentiments = ["positive", "negative", "neutral"]
	fig = go.Figure()
	
	for model in models_list:
		counts = sentiment_counts.get(model, {})
		values = [counts.get(sent, 0) for sent in sentiments]
		total = sum(values)
		percentages = [(v/total*100) if total > 0 else 0 for v in values]
		
		# Create text labels with just the count
		text_labels = [str(v) for v in values]
		
		fig.add_trace(go.Bar(
			name=model, 
			x=sentiments, 
			y=percentages,  # Show percentage on y-axis
			text=text_labels,  # Show count as label
			textposition='outside'
		))
	
	fig.update_layout(
		title="Sentiment Distribution Across Models (Percentage)",
		xaxis_title="Sentiment",
		yaxis_title="Percentage (%)",
		barmode="group",
		height=400,
		yaxis=dict(range=[0, 110])  # Give room for labels
	)
	
	return fig

def plot_sentiment_pies(df: pd.DataFrame, models_list: List[str]):
	"""Create pie charts for each model's sentiment distribution."""
	num_models = len(models_list)
	cols_per_row = 2
	rows = (num_models + cols_per_row - 1) // cols_per_row
	
	from plotly.subplots import make_subplots
	
	fig = make_subplots(
		rows=rows, cols=cols_per_row,
		specs=[[{'type':'domain'}] * cols_per_row for _ in range(rows)],
		subplot_titles=[f"{m} Sentiment" for m in models_list]
	)
	
	colors = {"positive": "#2ecc71", "negative": "#e74c3c", "neutral": "#95a5a6"}
	
	for idx, model in enumerate(models_list):
		col = f"{model}_Label"
		if col in df.columns:
			counts = df[col].value_counts()
			labels = counts.index.tolist()
			values = counts.values.tolist()
			color_list = [colors.get(lbl, "#95a5a6") for lbl in labels]
			
			row = (idx // cols_per_row) + 1
			col_pos = (idx % cols_per_row) + 1
			
			fig.add_trace(
				go.Pie(
					labels=labels,
					values=values,
					marker=dict(colors=color_list),
					hole=0.3
				),
				row=row, col=col_pos
			)
	
	fig.update_layout(
		height=300 * rows,
		showlegend=True,
		title_text="Sentiment Distribution by Model"
	)
	
	return fig

def plot_confidence_distribution(df: pd.DataFrame, models_list: List[str]):
	"""Plot confidence score distribution with statistical indicators.
	
	Box plot components:
	- **Box**: Shows Q1 (25th percentile) to Q3 (75th percentile) - middle 50% of predictions
	- **Line in box**: Median (50th percentile) - typical confidence
	- **Diamond**: Mean confidence with standard deviation
	- **Whiskers (fences)**: Show data range excluding outliers
	  - Upper fence: Q3 + 1.5*IQR (detects overconfident predictions)
	  - Lower fence: Q1 - 1.5*IQR (detects uncertain predictions)
	- **Points beyond whiskers**: Outliers (unusually high/low confidence)
	
	Useful for:
	- Comparing model certainty levels
	- Identifying models that are consistently confident vs uncertain
	- Detecting outlier predictions that need review
	"""
	fig = go.Figure()
	
	for model in models_list:
		col = f"{model}_Confidence"
		if col in df.columns:
			fig.add_trace(go.Box(
				y=df[col], 
				name=model, 
				boxmean='sd',  # Show mean and standard deviation
				boxpoints='outliers'  # Show outlier points
			))
	
	fig.update_layout(
		title="Model Confidence Distribution (Box Plot with Statistical Indicators)",
		yaxis_title="Confidence Score",
		height=450
	)
	
	return fig

def plot_agreement_matrix(df: pd.DataFrame, models_list: List[str]):
	"""Create agreement heatmap between models."""
	if len(models_list) < 2:
		return None
	
	# Calculate pairwise agreement
	n = len(models_list)
	agreement = np.zeros((n, n))
	
	for i, model1 in enumerate(models_list):
		for j, model2 in enumerate(models_list):
			col1 = f"{model1}_Label"
			col2 = f"{model2}_Label"
			if col1 in df.columns and col2 in df.columns:
				agreement[i, j] = (df[col1] == df[col2]).mean() * 100
	
	fig = go.Figure(data=go.Heatmap(
		z=agreement,
		x=models_list,
		y=models_list,
		colorscale="Blues",
		text=np.round(agreement, 1),
		texttemplate="%{text}%",
		textfont={"size": 12},
		colorbar=dict(title="Agreement %")
	))
	
	fig.update_layout(
		title="Model Agreement Matrix (%)",
		height=400
	)
	
	return fig

# -----------------------------------------------------------------------------
# Helper Functions: Sample Data
# -----------------------------------------------------------------------------
def get_sample_headlines(ticker: str) -> List[str]:
	"""Generate sample headlines."""
	t = ticker.upper()
	return [
		f"{t} beats earnings expectations; revenue climbs on robust demand",
		f"{t} shares dip as guidance disappoints despite solid quarterly results",
		f"Analysts upgrade {t} citing strong balance sheet and margin resilience",
		f"{t} announces new partnership; premarket trading shows positive reaction",
		f"Regulatory concerns weigh on {t}; stock under pressure amid investigation",
		f"{t} expands buyback program and raises dividend amid healthy cash flows",
		f"Macro headwinds and supply chain issues challenge {t}'s near-term outlook",
		f"{t} unveils product roadmap; investors react to innovation pipeline",
		f"Strong quarterly performance boosts {t} investor confidence significantly",
		f"{t} faces challenges but maintains market position with innovation"
	]

def get_default_labeled_headlines() -> Tuple[List[str], List[str]]:
	"""Get default headlines with ground truth labels for evaluation."""
	headlines = [
		"Company reports record profits and exceeds analyst expectations",
		"Stock plunges after disappointing earnings and weak guidance",
		"Firm maintains steady performance amid market volatility",
		"Investors cheer new product launch and strategic partnership",
		"Regulatory investigation causes sharp decline in share price",
		"Quarterly results meet expectations with stable revenue growth",
		"Major acquisition drives stock to all-time high",
		"Company warns of headwinds and lowers full-year forecast",
		"Dividend increase signals management confidence in future",
		"Mixed signals from earnings report leave investors uncertain"
	]
	
	labels = [
		"positive", "negative", "neutral", "positive", "negative",
		"neutral", "positive", "negative", "positive", "neutral"
	]
	
	return headlines, labels

def load_user_labeled_dataset(uploaded_file) -> Tuple[List[str], List[str]]:
	"""Load user-uploaded labeled dataset from CSV or Excel file.
	
	Expected format:
	- Two columns: 'Headline' and 'Label'
	- Labels must be: 'positive', 'negative', or 'neutral' (case-insensitive)
	
	Args:
		uploaded_file: Streamlit UploadedFile object (CSV or Excel)
		
	Returns:
		Tuple of (headlines list, labels list)
	"""
	try:
		# Read file based on extension
		file_ext = uploaded_file.name.split('.')[-1].lower()
		
		if file_ext == 'csv':
			df = pd.read_csv(uploaded_file)
		elif file_ext in ['xlsx', 'xls']:
			df = pd.read_excel(uploaded_file)
		else:
			raise ValueError(f"Unsupported file format: {file_ext}. Please use CSV or Excel (.xlsx, .xls)")
		
		# Validate columns
		if 'Headline' not in df.columns or 'Label' not in df.columns:
			raise ValueError("File must contain 'Headline' and 'Label' columns")
		
		# Clean and validate data
		df = df[['Headline', 'Label']].dropna()
		df['Label'] = df['Label'].str.lower().str.strip()
		
		# Validate labels
		valid_labels = {'positive', 'negative', 'neutral'}
		invalid_labels = set(df['Label'].unique()) - valid_labels
		if invalid_labels:
			raise ValueError(f"Invalid labels found: {invalid_labels}. Labels must be 'positive', 'negative', or 'neutral'")
		
		if len(df) == 0:
			raise ValueError("No valid data found in file")
		
		headlines = df['Headline'].tolist()
		labels = df['Label'].tolist()
		
		return headlines, labels
		
	except Exception as e:
		raise Exception(f"Error loading file: {str(e)}")

# -----------------------------------------------------------------------------
# Fetch Stock Data (for Future Work section)
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def fetch_stock_data(ticker: str, period: str) -> pd.DataFrame:
	"""Fetch historical stock data."""
	try:
		stock = yf.Ticker(ticker)
		df = stock.history(period=period)
		return df if not df.empty else pd.DataFrame()
	except:
		return pd.DataFrame()

def plot_stock_chart(df: pd.DataFrame, ticker: str):
	"""Create stock price candlestick chart."""
	fig = go.Figure(data=[go.Candlestick(
		x=df.index,
		open=df["Open"],
		high=df["High"],
		low=df["Low"],
		close=df["Close"],
		name=ticker
	)])
	
	fig.update_layout(
		title=f"{ticker} Stock Price",
		xaxis_title="Date",
		yaxis_title="Price (USD)",
		height=400,
		xaxis_rangeslider_visible=False
	)
	
	return fig

# =============================================================================
# MAIN APPLICATION
# =============================================================================

st.title("🎯 Financial News Multi-Model Sentiment Analysis")
st.markdown("### Reza Tashakkori")
st.markdown("---")

# Sidebar Configuration
st.sidebar.header("⚙️ Configuration")

# Model selection
available_models = ["BERT", "DistilBERT-SST", "FinBERT", "FinBERT-Tone"]
selected_models = st.sidebar.multiselect(
	"Select Models to Compare",
	available_models,
	default=["FinBERT", "BERT"]
)

if len(selected_models) == 0:
	st.error("Please select at least one model.")
	st.stop()

# News source
news_source = st.sidebar.radio(
	"News Source",
	["Sample Headlines", "Custom Input", "Labeled Dataset (for Evaluation)"],
	help="**Sample Headlines**: Pre-generated headlines for quick testing\n\n**Custom Input**: Paste your own headlines\n\n**Labeled Dataset**: 10 pre-labeled headlines with ground truth labels to calculate accuracy, precision, recall, and F1-score"
)

ticker = ""
custom_text = ""
labeled_dataset_source = "Default"
uploaded_file = None

ticker = st.sidebar.text_input("Stock Ticker", value="TSLA").upper()

if news_source == "Custom Input":
	custom_text = st.sidebar.text_area(
		"Enter headlines (one per line)",
		height=250,
		placeholder="Headline 1\nHeadline 2\n..."
	)
elif news_source == "Labeled Dataset (for Evaluation)":
	# Sub-option for labeled dataset
	labeled_dataset_source = st.sidebar.radio(
		"Labeled Dataset Source",
		["Default (10 samples)", "Upload File (CSV/Excel)"],
		help="**Default**: Use 10 pre-labeled sample headlines\n\n**Upload File**: Upload your own CSV or Excel file with 'Headline' and 'Label' columns. Labels must be 'positive', 'negative', or 'neutral'."
	)
	
	if labeled_dataset_source == "Upload File (CSV/Excel)":
		uploaded_file = st.sidebar.file_uploader(
			"Upload Labeled Dataset",
			type=["csv", "xlsx", "xls"],
			help="File must contain two columns: 'Headline' and 'Label'. Example:\n\nHeadline,Label\nStock price surges,positive\nCompany reports loss,negative\nQuarterly results as expected,neutral"
		)
		
		if uploaded_file is not None:
			st.sidebar.success(f"✅ File uploaded: {uploaded_file.name}")
		else:
			st.sidebar.info("📁 Please upload a CSV or Excel file")

# Analysis options
st.sidebar.markdown("---")
st.sidebar.subheader("Analysis Options")
show_flowchart = st.sidebar.checkbox("Show Process Flowchart", value=True)
show_evaluation = st.sidebar.checkbox(
	"Show Evaluation Metrics", 
	value=True,
	help="Shows Accuracy, Precision, Recall, F1-Score, and Confusion Matrices. Only works with 'Labeled Dataset' news source which has ground truth labels."
)
enable_summarization = st.sidebar.checkbox("Generate Summary", value=False)

analyze_btn = st.sidebar.button("🚀 Analyze", type="primary", use_container_width=True)

# Load models
with st.spinner("Loading models..."):
	all_models = load_sentiment_models()
	selected_model_dict = {k: v for k, v in all_models.items() if k in selected_models}

# =============================================================================
# MAIN CONTENT
# =============================================================================

# Show flowchart independently of analyze button
if show_flowchart:
	st.markdown("---")
	st.subheader("📊 NLP Pipeline Process Flow")
	st.plotly_chart(create_process_flowchart(), use_container_width=True, key="flowchart_main")

if analyze_btn:
	
	# Prepare headlines
	if news_source == "Sample Headlines":
		headlines = get_sample_headlines(ticker)
		ground_truth = None
	elif news_source == "Custom Input":
		headlines = [h.strip() for h in custom_text.split("\n") if h.strip()]
		if not headlines:
			st.error("Please enter at least one headline.")
			st.stop()
		ground_truth = None
	else:  # Labeled Dataset
		if labeled_dataset_source == "Default (10 samples)":
			headlines, ground_truth = get_default_labeled_headlines()
		else:  # Upload File
			if uploaded_file is None:
				st.error("Please upload a CSV or Excel file with labeled data.")
				st.stop()
			try:
				headlines, ground_truth = load_user_labeled_dataset(uploaded_file)
				st.info(f"📊 Loaded {len(headlines)} labeled headlines from {uploaded_file.name}")
			except Exception as e:
				st.error(f"❌ {str(e)}")
				st.stop()
	
	st.success(f"✅ Analyzing {len(headlines)} headlines with {len(selected_models)} model(s)...")
	
	# Analyze with all selected models
	with st.spinner("Running sentiment analysis..."):
		results_df = analyze_with_all_models(headlines, selected_model_dict)
	
	# =============================================================================
	# SECTION 1: SENTIMENT ANALYSIS RESULTS
	# =============================================================================
	st.markdown("---")
	st.header("📰 1. Sentiment Analysis Results")
	
	# Model comparison charts
	col1, col2 = st.columns(2)
	
	with col1:
		st.plotly_chart(plot_model_comparison(results_df, selected_models), use_container_width=True)
	
	with col2:
		st.plotly_chart(plot_confidence_distribution(results_df, selected_models), use_container_width=True)
		st.caption("""
**Understanding the Box Plot:**
- **Box (IQR)**: Middle 50% of predictions (Q1 to Q3)
- **Line**: Median confidence (typical prediction certainty)
- **Diamond**: Mean ± Standard Deviation
- **Whiskers**: Data range excluding outliers (Q1-1.5×IQR to Q3+1.5×IQR)
- **Points**: Outliers (unusually high/low confidence predictions)

**Why useful?** Compares model reliability \n- tight boxes = consistent confidence \n- wide boxes = variable certainty
""")
	
	# Pie charts for sentiment distribution
	st.plotly_chart(plot_sentiment_pies(results_df, selected_models), use_container_width=True)
	
	# Agreement matrix (if multiple models)
	if len(selected_models) > 1:
		st.plotly_chart(plot_agreement_matrix(results_df, selected_models), use_container_width=True)
	
	# =============================================================================
	# SECTION 2: DETAILED RESULTS TABLE
	# =============================================================================
	st.markdown("---")
	st.header("📋 2. Detailed Analysis Table")
	
	st.markdown("""
	**How to use this table:**
	- Click column headers to sort
	- Use search box to filter headlines
	- Compare predictions across different models
	- Check confidence scores to assess prediction reliability
	""")
	
	# Prepare display dataframe with formatted predictions
	display_data = []
	for _, row in results_df.iterrows():
		display_row = {"ID": row["ID"], "Headline": row["Headline"]}
		
		for model in selected_models:
			pos = row[f"{model}_Pos"]
			neg = row[f"{model}_Neg"]
			neu = row[f"{model}_Neu"]
			
			# Find which is dominant
			max_val = max(pos, neg, neu)
			
			# Format with bold for dominant and determine color
			if pos == max_val:
				prediction_text = f"Pos: {100*pos:.2f}%  \nNeg: {100*neg:.2f}%  \nNeu: {100*neu:.2f}%"
				color = "🟢"  # Green indicator
			elif neg == max_val:
				prediction_text = f"Pos: {100*pos:.2f}%  \nNeg: {100*neg:.2f}%  \nNeu: {100*neu:.2f}%"
				color = "🔴"  # Red indicator
			else:
				prediction_text = f"Pos: {100*pos:.2f}%  \nNeg: {100*neg:.2f}%  \nNeu: {100*neu:.2f}%"
				color = "⚪"  # Gray indicator
			
			display_row[f"{model}"] = f"{color} {prediction_text}"
		
		display_data.append(display_row)
	
	display_df = pd.DataFrame(display_data)
	
	# Show dataframe
	st.dataframe(
		display_df,
		use_container_width=True,
		height=400,
		column_config={
			"Headline": st.column_config.TextColumn("Headline", width="large"),
			**{m: st.column_config.TextColumn(f"{m} Prediction", width="medium") for m in selected_models}
		},
		hide_index=True
	)
	
	# Download option
	# csv = display_df.to_csv(index=False)
	# st.download_button(
	# 	label="📥 Download Results as CSV",
	# 	data=csv,
	# 	file_name=f"sentiment_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
	# 	mime="text/csv"
	# )
	
	# Display total headline count
	st.markdown(f"\n**Total Number of Headlines = {len(headlines)}**")
	
	# =============================================================================
	# SECTION 3: EVALUATION METRICS
	# =============================================================================
	if show_evaluation and ground_truth is not None:
		st.markdown("---")
		st.header("📊 3. Model Evaluation Metrics")
		
		st.markdown("""
		**Evaluation Metrics Explained:**
		- **Accuracy**: % of correct predictions
		- **Precision**: Of predicted positive/negative, how many were correct?
		- **Recall**: Of actual positive/negative, how many did we find?
		- **F1-Score**: Harmonic mean of precision and recall (balance metric)
		""")
		
		# Calculate metrics for each model
		metrics_data = []
		
		for model in selected_models:
			pred_col = f"{model}_Label"
			predictions = results_df[pred_col].tolist()
			
			# Overall accuracy
			acc = accuracy_score(ground_truth, predictions)
			
			# Per-class metrics
			precision, recall, f1, support = precision_recall_fscore_support(
				ground_truth, predictions, average=None, labels=["positive", "negative", "neutral"]
			)
			
			# Weighted average
			precision_w, recall_w, f1_w, _ = precision_recall_fscore_support(
				ground_truth, predictions, average="weighted"
			)
			
			metrics_data.append({
				"Model": model,
				"Accuracy": f"{acc*100:.1f}%",
				"Precision (weighted)": f"{precision_w*100:.1f}%",
				"Recall (weighted)": f"{recall_w*100:.1f}%",
				"F1-Score (weighted)": f"{f1_w*100:.1f}%"
			})
		
		metrics_df = pd.DataFrame(metrics_data)
		st.dataframe(metrics_df, use_container_width=True, hide_index=True)
		
		# Confusion matrices
		st.subheader("Confusion Matrices")
		
		cols = st.columns(len(selected_models))
		for idx, model in enumerate(selected_models):
			with cols[idx]:
				pred_col = f"{model}_Label"
				predictions = results_df[pred_col].tolist()
				
				cm = confusion_matrix(ground_truth, predictions, labels=["positive", "negative", "neutral"])
				
				fig = go.Figure(data=go.Heatmap(
					z=cm,
					x=["Positive", "Negative", "Neutral"],
					y=["Positive", "Negative", "Neutral"],
					colorscale="Blues",
					text=cm,
					texttemplate="%{text}",
					textfont={"size": 14}
				))
				
				fig.update_layout(
					title=f"{model}",
					xaxis_title="Predicted",
					yaxis_title="Actual",
					height=350
				)
				
				st.plotly_chart(fig, use_container_width=True)
	
	# =============================================================================
	# SECTION 4: TEXT SUMMARIZATION WITH MODEL COMPARISON
	# =============================================================================
	if enable_summarization:
		st.markdown("---")
		st.header("📝 4. Text Summarization - Multi-Model Comparison")
		
		st.markdown("""
		**Summarization Models:**
		- **BART**: Facebook's BART fine-tuned on CNN/DailyMail
		- **T5**: Google's T5 (Text-to-Text Transfer Transformer)
		- **Pegasus**: Google's Pegasus fine-tuned on XSum
		
		**Evaluation Metrics:**
		- Length (tokens)
		- Abstractiveness (unique n-grams)
		- Coverage (how much original content is represented)
		""")
		
		with st.spinner("Generating summaries with 3 models..."):
			try:
				summarizers = load_summarizers()
				combined_text = " ".join(headlines)
				
				# Split if too long
				max_length = 1024
				if len(combined_text) > max_length:
					combined_text = combined_text[:max_length]
				
				summary_results = []
				
				for model_name, summarizer in summarizers.items():
					try:
						if model_name == "T5":
							# T5 needs special prompt - generate comprehensive summary
							prompt = f"summarize: {combined_text}"
							summary = summarizer(prompt, max_length=150, min_length=50, do_sample=False, num_beams=4)
						else:
							# Generate comprehensive summaries with better quality
							summary = summarizer(combined_text, max_length=200, min_length=60, do_sample=False, num_beams=4)
						
						summary_text = summary[0]['summary_text']
						
						# Calculate metrics
						token_count = len(summary_text.split())
						
						# Abstractiveness: ratio of unique bigrams
						words = summary_text.lower().split()
						bigrams = set([f"{words[i]} {words[i+1]}" for i in range(len(words)-1)])
						abstractiveness = len(bigrams) / max(len(words) - 1, 1)
						
						# Coverage: how many headline words appear in summary
						headline_words = set(combined_text.lower().split())
						summary_words = set(summary_text.lower().split())
						coverage = len(headline_words & summary_words) / len(headline_words) if headline_words else 0
						
						summary_results.append({
							"Model": model_name,
							"Summary": summary_text,
							"Length (tokens)": token_count,
							"Abstractiveness": f"{abstractiveness:.2f}",
							"Coverage": f"{coverage*100:.1f}%"
						})
					except Exception as e:
						summary_results.append({
							"Model": model_name,
							"Summary": f"Error: {str(e)[:50]}",
							"Length (tokens)": 0,
							"Abstractiveness": "N/A",
							"Coverage": "N/A"
						})
				
				# Display summaries
				for result in summary_results:
					if "Error" not in result["Summary"]:
						st.success(f"**{result['Model']}**: {result['Summary']}")
					else:
						st.warning(f"**{result['Model']}**: {result['Summary']}")
				
				# Show metrics comparison
				st.markdown("### Summarizer Performance Metrics")
				metrics_df = pd.DataFrame(summary_results)
				metrics_display = metrics_df[["Model", "Length (tokens)", "Abstractiveness", "Coverage"]]
				st.dataframe(metrics_display, use_container_width=True, hide_index=True)
				
				st.markdown("""
				**Metrics Interpretation:**
				- **Length**: Shorter summaries are more concise
				- **Abstractiveness**: Higher values indicate more paraphrasing (better)
				- **Coverage**: Higher percentages capture more original content
				""")
				
			except Exception as e:
				st.error(f"Summarization error: {e}")
	
	# =============================================================================
	# SECTION 5: INSIGHTS & RECOMMENDATIONS
	# =============================================================================
	st.markdown("---")
	st.header("💡 5. Key Insights")
	
	# Calculate overall sentiment (aggregate across ALL selected models)
	all_predictions = []
	for model in selected_models:
		all_predictions.extend(results_df[f"{model}_Label"].tolist())
	
	total = len(all_predictions)
	pos_count = all_predictions.count("positive")
	neg_count = all_predictions.count("negative")
	neu_count = all_predictions.count("neutral")
	
	pos_pct = (pos_count / total) * 100 if total > 0 else 0
	neg_pct = (neg_count / total) * 100 if total > 0 else 0
	neu_pct = (neu_count / total) * 100 if total > 0 else 0
	
	col1, col2, col3 = st.columns(3)
	col1.metric("Positive Sentiment", f"{pos_pct:.1f}%", delta=None)
	col2.metric("Negative Sentiment", f"{neg_pct:.1f}%", delta=None)
	col3.metric("Neutral Sentiment", f"{neu_pct:.1f}%", delta=None)
	
	# Interpretation
	if pos_pct > 50:
		st.success(f"🟢 **Overall Bullish Sentiment**: {pos_pct:.1f}% of headlines are positive.")
	elif neg_pct > 50:
		st.error(f"🔴 **Overall Bearish Sentiment**: {neg_pct:.1f}% of headlines are negative.")
	else:
		st.warning(f"🟡 **Mixed Sentiment**: No clear dominant sentiment detected.")
	
	# Model agreement insights
	if len(selected_models) > 1:
		st.markdown("### Model Agreement Analysis")
		
		# Check where models disagree
		label_cols = [f"{m}_Label" for m in selected_models]
		agreement = results_df[label_cols].apply(lambda row: len(set(row)) == 1, axis=1)
		agreement_pct = agreement.mean() * 100
		
		st.write(f"**Models agree on {agreement_pct:.2f}% of predictions**")
		
		if agreement_pct < 70:
			st.warning("⚠️ Significant disagreement (< 70%) between models suggests uncertain or ambiguous headlines.")
		else:
			st.success("✅ High agreement (> 70%) indicates consistent and reliable predictions.")
	
	# =============================================================================
	# SECTION 6: FUTURE WORK - PRICE INTEGRATION
	# =============================================================================
	st.markdown("---")
	st.header("🔮 6. Future Work: Stock Price Integration")
	
	st.markdown("""
	**Potential Future Enhancements:**
	
	1. **Automated News Pipeline**: Real-time fetching and analysis of financial headlines from news APIs (NewsAPI, Alpha Vantage) with scheduled batch processing
	
	2. **Sentiment-Price Correlation**: Quantitative analysis linking sentiment scores to same-day and next-day stock returns using regression models
	
	3. **Temporal Sentiment Tracking**: Time-series visualization of sentiment trends to identify market mood shifts and sentiment momentum indicators
	
	4. **Intelligent Alert System**: Automated notifications when sentiment dramatically changes (e.g., >20% shift in 24h) for proactive trading decisions
	
	5. **Multi-Language Support**: Extend analysis to international markets with multilingual sentiment models for global stock coverage
	""")
	
	# Show stock price chart if ticker available
	if True:
		with st.expander("📈 View Stock Price Chart (Example)"):
			stock_df = fetch_stock_data(ticker, "1mo")
			if not stock_df.empty:
				st.plotly_chart(plot_stock_chart(stock_df, ticker), use_container_width=True)
				
				# Calculate metrics
				latest_price = stock_df["Close"].iloc[-1]
				price_change = ((stock_df["Close"].iloc[-1] / stock_df["Close"].iloc[0]) - 1) * 100
				
				c1, c2 = st.columns(2)
				c1.metric("Current Price", f"${latest_price:.2f}")
				c2.metric("1-Month Change", f"{price_change:+.2f}%")
				
				st.caption("*This section demonstrates potential integration with market data for future analysis.*")
			else:
				st.info("Stock data not available for this ticker.")

else:
	# Initial state - show instructions
	st.info("👈 Configure your analysis in the sidebar and click **Analyze** to begin.")
	
	st.markdown("""
### About This Project

This application demonstrates **multi-model sentiment analysis** for financial news headlines.

**Key Features:**
1. ✅ **Multi-Model Comparison**: Compare predictions from 4 different models
2. ✅ **Evaluation Metrics**: Accuracy, Precision, Recall, F1-Score with confusion matrices
3. ✅ **Process Visualization**: Clear flowchart of the NLP pipeline
4. ✅ **Interactive Analysis**: Sortable, filterable results table
5. ✅ **Model Agreement**: Analyze where models agree/disagree

**Models Available:**
- **BERT**: General-purpose,  language model - not domain-specialized
- **DistilBERT-SST**: General sentiment (Stanford Sentiment Treebank) - fast general-purpose sentiment model
- **FinBERT**: Specialized for financial text (ProsusAI) - domain-adapted BERT for finance
- **FinBERT-Tone**: Financial sentiment (Financial PhraseBank) - fine-tuned FinBERT

**Quick Start:**
1. Select models to compare in the sidebar
2. Choose a news source (sample, custom, or labeled dataset)
3. Click "Analyze" to see results
4. Explore visualizations, metrics, and insights
""")

# Footer
st.markdown("---")
st.caption("Built with Streamlit • Transformers • PyTorch")

# Cache Management Info
with st.expander("💾 Cache Management & Storage Tips"):
	st.markdown("""
	### Managing Hugging Face Cache & Storage
	
	**Why does Drive C get full?**
	- Hugging Face downloads models to: `C:\\Users\\YourName\\.cache\\huggingface\\hub\\`
	- Each model can be 500MB - 2GB in size
	- With 4 sentiment models + 3 summarizers = ~7-10GB total
	
	**Solutions:**
	
	**Option 1: Clear Hugging Face Cache (Recommended)**
	```powershell
	# Clear all cached models
	Remove-Item -Recurse -Force "$env:USERPROFILE\\.cache\\huggingface\\hub"
	
	# Or clear specific model (replace MODEL_NAME)
	Remove-Item -Recurse -Force "$env:USERPROFILE\\.cache\\huggingface\\hub\\models--MODEL_NAME"
	```
	
	**Option 2: Move Cache to Another Drive**
	```powershell
	# Set environment variable to use D: drive instead
	$env:HF_HOME = "D:\\huggingface_cache"
	
	# Make it permanent (add to your PowerShell profile)
	[System.Environment]::SetEnvironmentVariable('HF_HOME', 'D:\\huggingface_cache', 'User')
	```
	
	**Option 3: Clear Streamlit Cache**
	```powershell
	# Clear Streamlit cache directory
	Remove-Item -Recurse -Force "$env:USERPROFILE\\.streamlit\\cache"
	```
	
	**Best Practice:**
	- After each project session, clear Hugging Face cache if not needed
	- Keep only models you frequently use
	- Use the environment variable to move cache to drive with more space
	
	**Check Your Cache Size:**
	```powershell
	# See cache folder size
	Get-ChildItem "$env:USERPROFILE\\.cache\\huggingface" -Recurse | Measure-Object -Property Length -Sum
	```
	""")

