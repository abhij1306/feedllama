# Feed Optimizer - Deterministic Version

## Overview
This version uses **pure keyword matching** with NO AI/LLM. Fast, deterministic, and free.

## Files
- `feed_optimizer_deterministic.py` - Main Streamlit application
- `create_keyword_classes.py` - One-time GPC processor
- `keyword_classes.csv` - Universal keyword mapping (pre-generated)

## Setup

### One-Time: Generate Keyword Classes
```bash
python create_keyword_classes.py --input google_taxonomy.txt
```
This creates `keyword_classes.csv` which can be reused forever.

### Run Application
```bash
python -m streamlit run feed_optimizer_deterministic.py --server.port 8503
```

## Usage
1. Upload **Client Instructions** (CSV/Excel with product_type hierarchies)
2. Upload **Raw Data** (CSV/Excel)
3. **Select columns** to search for keywords
4. Click **Generate Rules**
5. Download Product Type and GPC rules

## Features
- ✅ Zero AI costs
- ✅ Instant results
- ✅ 100% deterministic
- ✅ Offline capable
- ✅ Universal keyword file (reusable)
- ✅ Supports 2-5 tier hierarchies
- ✅ Automatic pluralization
