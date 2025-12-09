# Feed Optimizer - LLM Version

## Overview
This version uses **OpenRouter API** with LLM for intelligent rule generation.

## Files
- `feed_optimizer_openrouter.py` - Main Streamlit application
- `feed_optimizer.py` - CLI version
- `.env` - API key configuration

## Setup

### Configure API Key
Edit `.env` file:
```
OPENROUTER_API_KEY=your-api-key-here
```

### Install Dependencies
```bash
pip install streamlit pandas requests openpyxl python-dotenv
```

### Run Application
```bash
python -m streamlit run feed_optimizer_openrouter.py --server.port 8502
```

## Usage
1. Upload **Raw Data** (CSV/Excel)
2. Upload **Client Instructions** (CSV/Excel) - Optional
3. Upload **Google Taxonomy** (TXT)
4. Enter **API Key** (or use .env)
5. Click **Generate Rules**
6. Download Product Type and GPC rules

## Features
- ✅ AI-powered rule generation
- ✅ Validates taxonomy against titles
- ✅ Handles 85K+ rows
- ✅ Pluralization
- ✅ Single quotes in output
- ✅ Dual column AND logic
- ✅ Discrepancy tracking

## Note
This version requires an OpenRouter API key and incurs API costs.
