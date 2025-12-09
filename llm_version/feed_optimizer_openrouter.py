import streamlit as st
import pandas as pd
import requests
import json
import re
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ============================================================
# File loading helpers
# ============================================================

def load_table(uploaded_file):
    if uploaded_file is None:
        return None
    name = uploaded_file.name.lower()
    if name.endswith(".csv") or name.endswith(".txt"):
        return pd.read_csv(uploaded_file)
    elif name.endswith(".xls") or name.endswith(".xlsx"):
        return pd.read_excel(uploaded_file)
    else:
        raise ValueError(f"Unsupported file type: {name}")


def load_taxonomy(uploaded_file):
    if uploaded_file is None:
        return []
    content = uploaded_file.read().decode("utf-8")
    lines = [ln.strip() for ln in content.split("\n")]
    return [ln for ln in lines if ln]


# ============================================================
# Extract product types from existing taxonomy columns
# ============================================================

def extract_product_keywords(df, title_column='title', max_keywords=500):
    """
    Extract unique product types using pdxtstyletax and pdxtcat_testing columns.
    Preserves exact values as they appear in the data.
    """
    
    # Check for required columns
    has_styletax = 'pdxtstyletax' in df.columns
    has_cat_testing = 'pdxtcat_testing' in df.columns
    
    if not has_styletax and not has_cat_testing:
        st.error("Required columns 'pdxtstyletax' or 'pdxtcat_testing' not found in data!")
        st.info(f"Available columns: {', '.join(df.columns.tolist())}")
        raise ValueError("Missing required taxonomy columns")
    
    st.info(f"Using columns: pdxtstyletax={has_styletax}, pdxtcat_testing={has_cat_testing}")
    
    # Find title column
    if title_column not in df.columns:
        possible_cols = [col for col in df.columns if 'title' in col.lower() or 'name' in col.lower() or 'product' in col.lower()]
        if possible_cols:
            title_column = possible_cols[0]
        else:
            title_column = None
    
    # Extract unique values from taxonomy columns (preserve exact casing and format)
    product_types = set()
    
    # Primary source: pdxtstyletax
    if has_styletax:
        styletax_values = df['pdxtstyletax'].dropna().astype(str).unique()
        for val in styletax_values:
            val = val.strip()
            if val and val not in ['', 'nan', 'NaN', 'none', 'None', 'null', 'NULL']:
                product_types.add(val)  # Keep original casing
        st.info(f"Found {len(styletax_values)} unique values in pdxtstyletax")
    
    # Secondary source: pdxtcat_testing (for second tier)
    if has_cat_testing:
        cat_values = df['pdxtcat_testing'].dropna().astype(str).unique()
        for val in cat_values:
            val = val.strip()
            if val and val not in ['', 'nan', 'NaN', 'none', 'None', 'null', 'NULL']:
                product_types.add(val)  # Keep original casing
        st.info(f"Found {len(cat_values)} unique values in pdxtcat_testing")
    
    # Convert to list and sort
    unique_keywords = sorted(list(product_types))[:max_keywords]
    
    st.success(f"Extracted {len(unique_keywords)} unique product types from taxonomy columns")
    
    return unique_keywords, title_column if title_column else 'title'


def create_sample_with_keywords(df, keywords, title_column, samples_per_keyword=2):
    """
    Create a representative sample showing products for each keyword.
    Searches in pdxtstyletax and pdxtcat_testing columns for taxonomy keywords.
    """
    samples = []
    
    # Check which columns are available
    has_styletax = 'pdxtstyletax' in df.columns
    has_cat_testing = 'pdxtcat_testing' in df.columns
    
    for kw in keywords[:100]:  # Limit to top 100 keywords for the sample
        # Search in taxonomy columns if available
        if has_styletax:
            matching = df[df['pdxtstyletax'].str.contains(kw, case=False, na=False, regex=False)]
        elif has_cat_testing:
            matching = df[df['pdxtcat_testing'].str.contains(kw, case=False, na=False, regex=False)]
        else:
            # Fallback to title if taxonomy columns not found
            matching = df[df[title_column].str.contains(kw, case=False, na=False, regex=False)]
        
        if not matching.empty:
            samples.append(matching.head(samples_per_keyword))
    
    if samples:
        return pd.concat(samples).drop_duplicates()
    return df.head(100)


# ============================================================
# OpenRouter chat completion
# ============================================================

def call_openrouter(model, system_prompt, user_prompt, api_key):
    url = "https://openrouter.ai/api/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://localhost",
        "X-Title": "FeedOptimizer"
    }

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    }

    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=600)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    except requests.exceptions.HTTPError as e:
        error_detail = ""
        try:
            error_detail = resp.json()
        except:
            error_detail = resp.text
        raise Exception(f"OpenRouter API Error ({resp.status_code}): {error_detail}") from e


# ============================================================
# JSON array extraction
# ============================================================

def extract_json_array(text):
    first = text.find("[")
    last = text.rfind("]")
    if first == -1 or last == -1:
        raise ValueError("LLM output did not contain a JSON array.")
    return json.loads(text[first:last+1])


# ============================================================
# Prompt builders
# ============================================================

def build_pt_prompts(keywords, sample_df, instr_df, title_column):
    # Create a compact sample JSON showing both taxonomy columns and title
    sample_json = sample_df.head(100).to_json(orient="records", force_ascii=False)
    instr_json = instr_df.head(20).to_json(orient="records", force_ascii=False) if instr_df is not None else "[]"
    
    keywords_text = "\n".join([f"- {kw}" for kw in keywords[:200]])

    system_prompt = """
You are a product feed optimizer generating transformation rules for `product_type` (Feedonomics format).

IMPORTANT CONTEXT:
- Primary source: pdxtstyletax column (specific product styles like "Plain Denim Jacket")
- Secondary source: pdxtcat_testing column (subcategories like "Jackets & Coats")
- Validation: You MUST check if pdxtstyletax logically matches the product title
- Fallback: If pdxtstyletax doesn't match title, extract keywords from title instead
- CRITICAL: The final product type in the hierarchy MUST be PLURALIZED
- CRITICAL: The "then" value MUST be wrapped in single quotes
- CRITICAL: export_id is ALWAYS 0, enabled is ALWAYS "1"
- CRITICAL: Condition must check BOTH pdxtstyletax AND pdxtcat_testing columns

VALIDATION LOGIC:
1. For each product, compare pdxtstyletax value with the title
2. If they match logically (e.g., "Plain Denim Jacket" in title contains "jacket" or "denim"):
   - Use pdxtstyletax value for the condition
   - PLURALIZE it for the hierarchy (e.g., "Plain Denim Jacket" → "Plain Denim Jackets")
   - Build hierarchy: "Category > pdxtcat_testing > Pluralized pdxtstyletax"
   - Mark as MATCH in validation_status
3. If they DON'T match (e.g., pdxtstyletax="Dress" but title="Men's Cotton Shirt"):
   - Extract the actual product type from title (e.g., "Casual Shirt")
   - PLURALIZE it for the hierarchy (e.g., "Casual Shirt" → "Casual Shirts")
   - Build hierarchy: "Category > Subcategory > Pluralized Keyword"
   - Mark as MISMATCH in validation_status

RULE STRUCTURE (EXACT FORMAT):
{
  "field_name": "product_type",
  "export_id": 0,
  "if": "[product_type] equal '' AND [pdxtstyletax] contains '<singular_keyword>' AND [pdxtcat_testing] contains '<singular_keyword>'",
  "then": "'Category > Subcategory > <pluralized_keyword>'",
  "enabled": "1",
  "validation_status": "MATCH" or "MISMATCH"
}

EXAMPLES:

Example 1 (Match - use pdxtstyletax):
Title: "Women's Plain Denim Jacket Blue"
pdxtstyletax: "Plain Denim Jacket"
pdxtcat_testing: "Jackets & Coats"
✓ Match! Output:
{
  "field_name": "product_type",
  "export_id": 0,
  "if": "[product_type] equal '' AND [pdxtstyletax] contains 'Plain Denim Jacket' AND [pdxtcat_testing] contains 'Plain Denim Jacket'",
  "then": "'Clothing > Jackets & Coats > Plain Denim Jackets'",
  "enabled": "1",
  "validation_status": "MATCH"
}

Example 2 (Mismatch - extract from title):
Title: "Men's Casual Cotton Shirt White"
pdxtstyletax: "Dress"
pdxtcat_testing: "Shirts & Tops"
✗ Mismatch! Extract "Casual Shirt" from title. Output:
{
  "field_name": "product_type",
  "export_id": 0,
  "if": "[product_type] equal '' AND [title] contains 'Casual Shirt' AND [pdxtcat_testing] contains 'Casual Shirt'",
  "then": "'Clothing > Shirts & Tops > Casual Shirts'",
  "enabled": "1",
  "validation_status": "MISMATCH"
}

PLURALIZATION RULES:
- Add 's' to most words: "Jacket" → "Jackets", "Shirt" → "Shirts"
- Words ending in 's', 'x', 'z', 'ch', 'sh': add 'es': "Dress" → "Dresses"
- Words ending in consonant + 'y': change 'y' to 'ies': "Accessory" → "Accessories"
- Irregular plurals: "Scarf" → "Scarves", "Leaf" → "Leaves"
- Keep compound words intact: "Plain Denim Jacket" → "Plain Denim Jackets"

CRITICAL REQUIREMENTS:
- Validate EVERY pdxtstyletax value against its title
- Use pdxtstyletax when it matches, extract from title when it doesn't
- For mismatches, use [title] in the condition instead of [pdxtstyletax]
- ALWAYS include BOTH [pdxtstyletax]/[title] AND [pdxtcat_testing] in the condition
- ALWAYS pluralize the final product type in the hierarchy
- Keep singular form in the condition, plural in the output
- ALWAYS wrap the "then" value in single quotes: 'Category > Subcategory > Product'
- ALWAYS use export_id: 0 and enabled: "1" for every rule
- ALWAYS include validation_status field: "MATCH" or "MISMATCH"
- Always preserve exact casing for keywords
- Output ONLY a JSON array
- No markdown, no explanation
"""

    user_prompt = f"""
PRODUCT STYLES FROM pdxtstyletax (total {len(keywords)} styles):
{keywords_text}

SAMPLE DATA (showing pdxtstyletax, pdxtcat_testing, and title):
{sample_json[:5000]}

TASK:
1. For EACH unique pdxtstyletax value, find sample products with that value
2. Check if pdxtstyletax logically matches the product titles
3. If MATCH: Create rule using pdxtstyletax (singular in condition, PLURAL in hierarchy)
4. If MISMATCH: Extract correct keyword from title (singular in condition, PLURAL in hierarchy)

REMEMBER: 
- Wrap the "then" value in single quotes like: 'Clothing > Category > Product'
- Use export_id: 0 and enabled: "1" for ALL rules
- Include BOTH columns in condition: [pdxtstyletax] AND [pdxtcat_testing] (or [title] AND [pdxtcat_testing] for mismatches)
- Add validation_status field to track matches vs mismatches

Generate product_type rules for ALL {len(keywords[:200])} styles above.
Return ONLY a pure JSON array.
"""
    return system_prompt, user_prompt


def build_gpc_prompts(pt_rules, taxonomy):
    pt_json = json.dumps(pt_rules, ensure_ascii=False)
    taxonomy_text = "\n".join(taxonomy[:300])

    system_prompt = """
You generate google_product_category rules.

Steps:
1. For EVERY product_type rule provided, create a corresponding GPC rule
2. Extract the main product keyword from each product_type
3. Match to the closest valid Google Product Taxonomy category
4. Output Feedonomics-compatible rules:

- field_name: "google_product_category"
- export_id: 0 (always 0)
- if: "[google_product_category] equal '' AND [product_type] contains 'keyword'"
- then: 'EXACT taxonomy category' (wrapped in single quotes)
- enabled: "1"

CRITICAL REQUIREMENTS:
- Generate a GPC rule for EVERY product_type rule provided
- Match to valid taxonomy categories only
- ALWAYS wrap the "then" value in single quotes: 'Category > Subcategory'
- ALWAYS use export_id: 0 and enabled: "1"
- Output MUST be a JSON array only
- No markdown, no explanation
"""

    user_prompt = f"""
PRODUCT TYPE RULES (total {len(pt_rules)} rules):
{pt_json[:20000]}

GOOGLE PRODUCT TAXONOMY:
{taxonomy_text}

IMPORTANT: Generate google_product_category rules for ALL {len(pt_rules)} product_type rules.
Match each product type to the most appropriate Google taxonomy category.
REMEMBER: Wrap the "then" value in single quotes like: 'Apparel & Accessories > Clothing > Shirts'

Return ONLY a pure JSON array.
"""

    return system_prompt, user_prompt


# ============================================================
# Rule generators
# ============================================================

def generate_pt_rules(raw_df, instr_df, model, api_key):
    # Extract unique keywords from all rows
    with st.spinner("Analyzing all 85,000+ products to extract unique keywords..."):
        keywords, title_column = extract_product_keywords(raw_df, max_keywords=500)
        st.success(f"Found {len(keywords)} unique product keywords across all rows")
    
    # Create representative sample
    sample_df = create_sample_with_keywords(raw_df, keywords, title_column)
    
    # Build prompts with keywords
    sys_prompt, usr_prompt = build_pt_prompts(keywords, sample_df, instr_df, title_column)
    
    # Call LLM
    with st.spinner(f"Generating rules for {len(keywords)} product types..."):
        text = call_openrouter(model, sys_prompt, usr_prompt, api_key)
    
    rules = extract_json_array(text)

    cleaned = []
    match_count = 0
    mismatch_count = 0
    
    for i, r in enumerate(rules):
        cond = str(r.get("if", "")).strip()
        then_val = str(r.get("then", "")).strip()
        validation_status = str(r.get("validation_status", "UNKNOWN")).strip()
        
        if cond and then_val:
            cleaned.append({
                "field_name": "product_type",
                "export_id": i,
                "if": cond,
                "then": then_val,
                "enabled": "1",
            })
            
            # Track validation status
            if validation_status == "MATCH":
                match_count += 1
            elif validation_status == "MISMATCH":
                mismatch_count += 1
    
    # Display discrepancy statistics
    if match_count > 0 or mismatch_count > 0:
        st.info(f"📊 Validation Results: {match_count} matches, {mismatch_count} discrepancies found")
    
    return cleaned


def generate_gpc_rules(pt_rules, taxonomy, model, api_key):
    """
    Generate GPC rules by extracting base nouns from product_type rules.
    Groups multiple nouns that map to the same category using contains_any.
    """
    
    # Extract unique base nouns from product_type rules
    base_nouns = set()
    for rule in pt_rules:
        if_condition = rule.get('if', '')
        # Extract the keyword from the condition
        # Format: "[product_type] equal '' AND [title] contains 'keyword'"
        import re
        match = re.search(r"contains ['\"]([^'\"]+)['\"]", if_condition)
        if match:
            keyword = match.group(1).strip()
            # Remove adjectives - take the last word (the noun)
            words = keyword.split()
            base_noun = words[-1] if words else keyword
            base_nouns.add(base_noun)
    
    base_nouns_list = sorted(list(base_nouns))
    
    st.info(f"Extracted {len(base_nouns_list)} unique base nouns for GPC mapping")
    
    # Now ask LLM to map these base nouns to Google taxonomy
    taxonomy_text = "\n".join(taxonomy[:500])
    nouns_text = "\n".join([f"- {noun}" for noun in base_nouns_list])
    
    system_prompt = """
You are mapping apparel product nouns to Google Product Taxonomy categories.

For EACH noun provided, find the MOST SPECIFIC matching category from the Google Product Taxonomy.

Output format: JSON array with objects containing:
- noun: the product noun
- category: the EXACT Google Product Taxonomy category

CRITICAL:
- Map EVERY noun provided
- Use EXACT taxonomy categories only
- Output ONLY a JSON array
- No markdown, no explanation
"""

    user_prompt = f"""
PRODUCT NOUNS TO MAP (total {len(base_nouns_list)}):
{nouns_text}

GOOGLE PRODUCT TAXONOMY:
{taxonomy_text}

Map EVERY noun above to its most appropriate Google Product Taxonomy category.
Return ONLY a JSON array: [{{"noun": "shirt", "category": "Apparel & Accessories > Clothing > Shirts & Tops"}}, ...]
"""

    # Call LLM for mapping
    with st.spinner(f"Mapping {len(base_nouns_list)} product nouns to Google taxonomy..."):
        text = call_openrouter(model, system_prompt, user_prompt, api_key)
    
    mappings = extract_json_array(text)
    
    # Validate and group nouns by category
    taxonomy_set = set(taxonomy)
    category_to_nouns = {}  # category -> list of nouns
    
    for mapping in mappings:
        noun = str(mapping.get("noun", "")).strip()
        category = str(mapping.get("category", "")).strip()
        
        if not noun or not category:
            continue
        
        # Validate category exists in taxonomy
        if category not in taxonomy_set:
            # Fuzzy match fallback
            matches = [t for t in taxonomy if category.lower() in t.lower() or noun.lower() in t.lower()]
            if matches:
                category = matches[0]
            else:
                # Skip if no match found
                continue
        
        # Group nouns by category
        if category not in category_to_nouns:
            category_to_nouns[category] = []
        category_to_nouns[category].append(noun)
    
    # Create GPC rules with contains_any for multiple nouns
    cleaned = []
    export_id = 0
    
    for category, nouns in sorted(category_to_nouns.items()):
        if len(nouns) == 1:
            # Single noun - use simple contains
            cleaned.append({
                "field_name": "google_product_category",
                "export_id": export_id,
                "if": f"[google_product_category] equal '' AND [product_type] contains '{nouns[0]}'",
                "then": f"'{category}'",
                "enabled": "1",
            })
        else:
            # Multiple nouns - use contains_any
            nouns_str = ", ".join([f"'{noun}'" for noun in sorted(nouns)])
            cleaned.append({
                "field_name": "google_product_category",
                "export_id": export_id,
                "if": f"[google_product_category] equal '' AND [product_type] contains_any ({nouns_str})",
                "then": f"'{category}'",
                "enabled": "1",
            })
        export_id += 1
    
    st.success(f"Grouped {len(base_nouns_list)} nouns into {len(cleaned)} GPC rules")
    
    return cleaned


# ============================================================
# STREAMLIT UI
# ============================================================

st.set_page_config(page_title="Feed Optimizer (OpenRouter)", layout="wide")

st.title("🧠 Feed Optimizer – Rule Generator (OpenRouter Version)")
st.write("Generate Product Type & GPC Rules using OpenRouter models.")
st.info("✨ This version analyzes ALL rows in your dataset to extract unique product types, then generates comprehensive rules.")

with st.sidebar:
    st.header("🔑 API Settings")
    
    # Try to load API key from environment variable
    env_api_key = os.getenv("OPENROUTER_API_KEY", "")
    
    if env_api_key and env_api_key != "your_api_key_here":
        api_key = env_api_key
        st.success("✅ API key loaded from .env file")
    else:
        api_key = st.text_input("OpenRouter API Key", type="password", help="Or add OPENROUTER_API_KEY to .env file")
    
    model = "kwaipilot/kat-coder-pro-v1:free"  # Using Kwaipilot free model
    st.info(f"Using model: {model}")

st.subheader("1. Upload Files")

raw_file = st.file_uploader("Raw Data (CSV/Excel)", type=["csv", "xls", "xlsx", "txt"])
instr_file = st.file_uploader("Client Instructions (CSV/Excel)", type=["csv", "xls", "xlsx"])

# Load GPC taxonomy from local file
TAXONOMY_FILE = "google_taxonomy.txt"
if not os.path.exists(TAXONOMY_FILE):
    st.error(f"❌ Google taxonomy file not found: {TAXONOMY_FILE}")
    st.info("Please ensure google_taxonomy.txt is in the same directory as this script.")
    st.stop()

st.success(f"✅ Using Google taxonomy file: {TAXONOMY_FILE}")

start = st.button("🚀 Generate Rules")


if start:
    if not api_key:
        st.error("Please enter your OpenRouter API key.")
        st.stop()

    if raw_file is None:
        st.error("Please upload Raw Data file.")
        st.stop()

    with st.spinner("Loading files…"):
        raw_df = load_table(raw_file)
        instr_df = load_table(instr_file) if instr_file else None
        
        # Load taxonomy from local file
        with open(TAXONOMY_FILE, 'r', encoding='utf-8') as f:
            taxonomy = [line.strip() for line in f if line.strip()]

    st.success(f"Files loaded. Raw data contains {len(raw_df):,} rows.")

    # PRODUCT TYPE RULES
    st.subheader("2. Generating Product Type Rules…")
    try:
        pt_rules = generate_pt_rules(raw_df, instr_df, model, api_key)
    except Exception as e:
        st.error("Error generating product type rules.")
        st.exception(e)
        st.stop()

    df_pt = pd.DataFrame(pt_rules)
    st.success(f"Generated {len(pt_rules)} product type rules!")
    st.write(df_pt)

    pt_csv = df_pt.to_csv(index=False).encode("utf-8")
    st.download_button("Download Product Type Rules CSV", pt_csv, "product_type_rules.csv", "text/csv")

    # GPC RULES
    st.subheader("3. Generating Google Product Category Rules…")
    with st.spinner("OpenRouter model generating GPC rules…"):
        try:
            gpc_rules = generate_gpc_rules(pt_rules, taxonomy, model, api_key)
        except Exception as e:
            st.error("Error generating GPC rules.")
            st.exception(e)
            st.stop()

    df_gpc = pd.DataFrame(gpc_rules)
    st.success(f"Generated {len(gpc_rules)} GPC rules!")
    st.write(df_gpc)

    gpc_csv = df_gpc.to_csv(index=False).encode("utf-8")
    st.download_button("Download GPC Rules CSV", gpc_csv, "gpc_rules.csv", "text/csv")

    st.success("All rules generated successfully!")
