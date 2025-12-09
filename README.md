# Feed Optimizer

Two versions of the Feed Optimizer for generating Product Type & GPC rules.

## 📁 Project Structure

```
feedllama/
├── deterministic_version/    # No AI - Pure keyword matching
│   ├── feed_optimizer_deterministic.py
│   ├── create_keyword_classes.py
│   ├── create_product_type_repository.py
│   ├── keyword_classes.csv (generated)
│   └── product_type_repository.txt (generated)
│
├── llm_version/              # AI-powered with OpenRouter
│   ├── feed_optimizer_openrouter.py
│   ├── .env
│   └── google_taxonomy.txt
│
└── google_taxonomy.txt       # Shared GPC taxonomy file
```

## 🎯 Deterministic Version (Recommended)

**Features:**
- ✅ Zero AI costs
- ✅ Instant results
- ✅ 100% deterministic
- ✅ Offline capable
- ✅ Category-aware matching

**Setup:**
```bash
cd deterministic_version
python create_keyword_classes.py --input ../google_taxonomy.txt
python -m streamlit run feed_optimizer_deterministic.py --server.port 8503
```

**Usage:**
1. Upload Client Instructions (product_type hierarchies)
2. Upload Raw Data
3. Select columns to search
4. Generate rules instantly

## 🧠 LLM Version

**Features:**
- ✅ AI-powered rule generation
- ✅ Handles complex patterns
- ✅ Validates taxonomy
- ✅ Discrepancy tracking

**Setup:**
```bash
cd llm_version
# Add API key to .env: OPENROUTER_API_KEY=your-key
python -m streamlit run feed_optimizer_openrouter.py --server.port 8502
```

## 📊 Comparison

| Feature | Deterministic | LLM |
|---------|--------------|-----|
| Cost | Free | API costs |
| Speed | Instant | ~30-60s |
| Accuracy | High (exact match) | Very High |
| Offline | Yes | No |
| Setup | One-time keyword file | API key |

## 🚀 Quick Start

**First time:**
```bash
# Generate universal keyword file (one-time)
cd deterministic_version
python create_keyword_classes.py --input ../google_taxonomy.txt
```

**Daily use:**
```bash
# Deterministic (recommended)
cd deterministic_version
python -m streamlit run feed_optimizer_deterministic.py --server.port 8503

# Or LLM version
cd llm_version
python -m streamlit run feed_optimizer_openrouter.py --server.port 8502
```

## 📝 Notes

- Both versions support 2-5 tier product_type hierarchies
- Automatic pluralization of last tier
- GPC rules use exact keyword matching
- `contains_any` for efficient multi-keyword rules
