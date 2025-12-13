# Financial News Multi-Model Sentiment Analyzer

An advanced NLP application demonstrating **multi-model sentiment analysis** with comprehensive evaluation metrics for financial news headlines. Built for reproducible research and educational purposes.

**Course**: Natural Language Processing in Data Science  
**Author**: Reza Tashakkori  
**Date**: December 2025  
**Purpose**: Demonstrate comparative analysis of transformer-based sentiment models with rigorous evaluation

## Key Features

### 🎯 Multi-Model Sentiment Analysis
Compare predictions across **4 transformer-based models**:
- **FinBERT** (ProsusAI) - Specialized for financial communication
- **FinBERT-Tone** (yiyanghkust) - Financial PhraseBank fine-tuned
- **BERT** (nlptown) - Multilingual 5-star sentiment model
- **DistilBERT-SST** - Stanford Sentiment Treebank fine-tuned

### 📊 Comprehensive Evaluation Metrics
- **Classification Metrics**: Accuracy, Precision, Recall, F1-Score (weighted)
- **Confusion Matrices**: Per-model visualization of prediction patterns
- **Model Agreement Analysis**: Pairwise agreement heatmap
- **Confidence Distribution**: Box plots with statistical indicators (median, IQR, outliers)

### 📝 Multi-Model Text Summarization
Compare summaries from **3 state-of-the-art models**:
- **BART** (facebook/bart-large-cnn) - Denoising autoencoder
- **T5** (t5-small) - Text-to-text transfer learning
- **Pegasus** (google/pegasus-xsum) - Pre-trained with gap-sentence generation

With performance metrics:
- Token length comparison
- Abstractiveness score (unique n-gram ratio)
- Coverage percentage (original content representation)

### 🔄 Interactive Process Visualization
- **NLP Pipeline Flowchart**: Visual representation of analysis workflow
- **Sentiment Distribution Charts**: Bar charts (with percentages) and pie charts
- **Confidence Box Plots**: Statistical distribution of model certainty
- **Agreement Heatmap**: Cross-model consistency analysis

### 📥 Flexible Input Options
1. **Sample Headlines**: Pre-generated financial news for quick testing
2. **Custom Input**: Paste your own headlines (one per line)
3. **Labeled Dataset**: Upload CSV/Excel with ground truth labels for evaluation
   - 10 pre-labeled samples included by default
   - Support for user-uploaded datasets

### 📈 Stock Price Integration (Demo)
- Yahoo Finance API integration
- Candlestick price charts
- Price change metrics
- Foundation for future sentiment-price correlation analysis

## Project Structure
```
Final Project/
  Code/
    app.py
  requirements.txt
  README.md
```

## Setup (Windows, Command Prompt - cmd)

1) Open Command Prompt (cmd) and go to the project root folder:
```cmd
cd /d "d:\Study\Semester 3\Natural Language Processing in Data Science\Final Project"
```

2) Create a virtual environment (Python 3.12) and activate it:
```cmd
python -m venv .venv
.venv\Scripts\activate
```

3) Upgrade pip and install dependencies:
```cmd
python -m pip install --upgrade pip
pip install -r requirements.txt
```
**Note**: First run will download ~8-12GB of models to `C:\Users\YourName\.cache\huggingface\`. See "Cache Management" section below.

4) Run the application:
```cmd
python -m streamlit run "Code\Final_app.py"
```

The app will automatically open in your default browser at `http://localhost:8501`

**Alternative run command** (if `python` command not found):
```cmd
py -m streamlit run "Code\Final_app.py"
```

## Usage

### 1. Configure Analysis (Sidebar)
- **Select Models**: Choose 1-4 models to compare (default: FinBERT + BERT)
- **News Source**: 
  - `Sample Headlines` - Pre-generated for ticker (default: TSLA)
  - `Custom Input` - Paste your own headlines (one per line)
  - `Labeled Dataset` - Use 10 default samples or upload CSV/Excel for evaluation
- **Stock Ticker**: Enter symbol (e.g., AAPL, GOOGL, MSFT, TSLA, AMZN)
- **Analysis Options**:
  - Show Process Flowchart (default: ON)
  - Show Evaluation Metrics (requires labeled dataset)
  - Generate Summary (compares BART, T5, Pegasus)

### 2. Run Analysis
Click **"🚀 Analyze"** button and wait for model inference (~5-30 seconds depending on models selected)

### 3. Explore Results

**Section 1: Sentiment Analysis Results**
- Sentiment distribution bar chart (with percentages)
- Confidence distribution box plots (statistical indicators)
- Per-model sentiment pie charts
- Model agreement heatmap (if 2+ models selected)

**Section 2: Detailed Analysis Table**
- Sortable, searchable table with per-headline predictions
- Color-coded indicators (🟢 positive, 🔴 negative, ⚪ neutral)
- Confidence scores and full probability distributions
- Compare predictions across selected models

**Section 3: Evaluation Metrics** (if labeled dataset used)
- Accuracy, Precision, Recall, F1-Score for each model
- Confusion matrices with prediction heatmaps
- Identifies where models succeed/fail

**Section 4: Text Summarization** (if enabled)
- Side-by-side comparison of 3 summarization models
- Performance metrics table (length, abstractiveness, coverage)
- Quality assessment of generated summaries

**Section 5: Key Insights**
- Overall sentiment percentages (aggregated across all models)
- Bullish/Bearish/Neutral interpretation
- Model agreement statistics

**Section 6: Future Work**
- Stock price chart demo (1-month candlestick)
- Discussion of potential enhancements

## Troubleshooting (cmd)
- "python is not recognized": install Python 3.12 and reopen cmd, or use `py`.
- "streamlit is not recognized": ensure the venv is active (prompt shows `(.venv)`) or use `python -m streamlit ...`.
- To deactivate the venv: `deactivate`

## Technical Details

### NLP Models Used (7 Total)

**Sentiment Analysis (4 models):**
1. `ProsusAI/finbert` - BERT fine-tuned on financial communication (TRC2-financial, corporate reports)
2. `yiyanghkust/finbert-tone` - FinBERT fine-tuned on Financial PhraseBank (4,840 annotated sentences)
3. `nlptown/bert-base-multilingual-uncased-sentiment` - Multilingual BERT with 5-star rating scale
4. `distilbert-base-uncased-finetuned-sst-2-english` - Distilled BERT on Stanford Sentiment Treebank v2

**Summarization (3 models):**
5. `facebook/bart-large-cnn` - BART (Bidirectional and Auto-Regressive Transformers) fine-tuned on CNN/DailyMail
6. `t5-small` - T5 (Text-to-Text Transfer Transformer) small variant
7. `google/pegasus-xsum` - PEGASUS pre-trained with gap-sentence generation on XSum dataset

### NLP Techniques Demonstrated
- **Multi-Model Ensemble**: Comparative analysis across architectures (BERT, DistilBERT, BART, T5, Pegasus)
- **Transfer Learning**: Leveraging pre-trained models fine-tuned for specific tasks
- **Evaluation Metrics**: Precision, Recall, F1-Score, Confusion Matrix analysis
- **Probabilistic Classification**: Softmax probability distributions for uncertainty quantification
- **Label Normalization**: Mapping diverse model outputs (binary, 3-class, 5-star) to unified sentiment labels
- **Statistical Analysis**: Box plots with IQR, outlier detection, agreement matrices
- **Abstractive Summarization**: Sequence-to-sequence generation with beam search
- **Performance Benchmarking**: Length, abstractiveness, and coverage metrics for summarization quality

### Data Sources
- **Stock Data**: Yahoo Finance API via `yfinance` library
- **News**: Sample headlines or user-provided text (no API keys required for basic version)

## Project Structure
```
Final Project/
├── Code/
│   ├── Final_app.py        # Main multi-model sentiment analyzer (USE THIS)
│   └── app.py              # Alternative: basic single-model version
├── requirements.txt         # Python dependencies (all 7 models)
└── README.md               # This documentation
```

**Primary Application**: `Code/Final_app.py` (recommended)
- Full-featured multi-model comparison
- Evaluation metrics with labeled datasets
- Text summarization with 3 models
- Interactive visualizations

## Dependencies
See `requirements.txt` for full list. Key packages:
- **streamlit** (≥1.32): Interactive web UI framework
- **transformers** (≥4.44): Hugging Face model hub access
- **torch** (≥2.3): PyTorch deep learning backend
- **scikit-learn** (≥1.4): Evaluation metrics (accuracy, precision, recall, F1, confusion matrix)
- **plotly** (≥5.22): Interactive charts (bar, pie, box, heatmap, candlestick)
- **pandas** (≥2.2) + **numpy** (≥1.26): Data manipulation
- **yfinance** (≥0.2): Yahoo Finance API for stock data
- **openpyxl** (≥3.1): Excel file support (optional, for labeled dataset uploads)

**Total Download Size**: ~8-12GB (all 7 models cached on first run)

## Cache Management

### Storage Requirements
- **First Run**: Downloads ~8-12GB of models to Hugging Face cache
- **Default Location**: `C:\Users\YourName\.cache\huggingface\hub\`
- **Subsequent Runs**: Loads from cache (fast)

### Clear Cache (if needed)
```powershell
# Clear all Hugging Face models
Remove-Item -Recurse -Force "$env:USERPROFILE\.cache\huggingface\hub"

# Check cache size
Get-ChildItem "$env:USERPROFILE\.cache\huggingface" -Recurse | Measure-Object -Property Length -Sum
```

### Move Cache to Another Drive
```powershell
# Permanently set cache to D: drive
[System.Environment]::SetEnvironmentVariable('HF_HOME', 'D:\huggingface_cache', 'User')
```

## Notes
- All models cached after first load (~5-10 min initial setup)
- Sample headlines and labeled dataset included (no API keys required)
- Designed for educational/research purposes
- Cross-platform compatible (Windows/Linux/macOS)
- Requires ~15GB free space on C: drive (or alternative via `HF_HOME`)

## Future Enhancements

### Phase 1: Data Pipeline
- [ ] Real-time news API integration (NewsAPI, Alpha Vantage, Finnhub)
- [ ] Automated headline scraping with scheduled batch processing
- [ ] Multi-ticker batch comparison dashboard

### Phase 2: Advanced Analysis
- [ ] Sentiment-price correlation quantitative analysis
- [ ] Temporal sentiment tracking with time-series visualization
- [ ] Confidence-weighted sentiment scoring
- [ ] Sentiment momentum indicators

### Phase 3: Production Features
- [ ] CSV/PDF report export with charts
- [ ] Email/SMS alert system for sentiment spikes
- [ ] REST API for programmatic access
- [ ] Model fine-tuning interface with custom datasets

### Phase 4: Global Expansion
- [ ] Multi-language support (multilingual models)
- [ ] International market coverage
- [ ] Cross-market sentiment comparison

## Citation

If you use this project in your research or work, please cite:

```bibtex
@misc{tashakkori2025finsentiment,
  author = {Tashakkori, Reza},
  title = {Financial News Multi-Model Sentiment Analyzer},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/yourusername/financial-sentiment-analyzer}}
}
```

## License

MIT License - see LICENSE file for details

## Acknowledgments

- **Hugging Face** for transformer model hosting and libraries
- **ProsusAI** for FinBERT model
- **yiyanghkust** for FinBERT-Tone model
- **Streamlit** for interactive web framework
- **Course Instructor** for project guidance
