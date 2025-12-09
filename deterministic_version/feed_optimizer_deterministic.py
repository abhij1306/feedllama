import streamlit as st
import pandas as pd
import csv
from pathlib import Path
from typing import Dict, List, Set, Tuple

# ============================================================
# Configuration
# ============================================================

UNIVERSAL_KEYWORD_FILE = "keyword_classes.csv"

# ============================================================
# File Loading
# ============================================================

def load_table(uploaded_file):
    """Load CSV or Excel file into DataFrame."""
    if uploaded_file is None:
        return None
    name = uploaded_file.name.lower()
    if name.endswith(".csv") or name.endswith(".txt"):
        return pd.read_csv(uploaded_file)
    elif name.endswith(".xls") or name.endswith(".xlsx"):
        return pd.read_excel(uploaded_file)
    else:
        raise ValueError(f"Unsupported file type: {name}")


def load_keyword_classes(file_path: str) -> Dict[str, List[str]]:
    """
    Load universal keyword classes from CSV.
    
    Returns:
        Dictionary mapping category to list of keywords
    """
    keyword_classes = {}
    
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            category = row['category_class']
            keywords_str = row['keywords']
            keywords = [kw.strip() for kw in keywords_str.split(',')]
            keyword_classes[category] = keywords
    
    return keyword_classes


# ============================================================
# Keyword Matching (Optimized)
# ============================================================

def extract_unique_values_from_columns(df: pd.DataFrame, 
                                       selected_columns: List[str]) -> Set[str]:
    """
    Extract base nouns (last word) from selected columns.
    Only extracts nouns, not adjectives or full phrases.
    
    Args:
        df: DataFrame to extract from
        selected_columns: Columns to extract values from
    
    Returns:
        Set of base nouns found in the data
    """
    unique_nouns = set()
    
    for col in selected_columns:
        if col in df.columns:
            # Get all unique non-null values
            values = df[col].dropna().astype(str).unique()
            for val in values:
                # Clean and split the value
                cleaned_val = val.replace('>', ' ').replace('&', ' ').replace(',', ' ')
                words = cleaned_val.split()
                
                # Extract the last word (the noun)
                if words:
                    base_noun = words[-1].strip()
                    # Skip very short words and common words
                    skip_words = {'and', 'or', 'the', 'a', 'an', 'for', 'with', 'by', 'in', 'on'}
                    if base_noun and len(base_noun) > 2 and base_noun.lower() not in skip_words:
                        unique_nouns.add(base_noun)
    
    return unique_nouns


# ============================================================
# Pluralization
# ============================================================

def pluralize(word: str) -> str:
    """
    Apply English pluralization rules.
    
    Args:
        word: Singular word
    
    Returns:
        Pluralized word
    """
    # Irregular plurals
    irregulars = {
        'child': 'children',
        'person': 'people',
        'man': 'men',
        'woman': 'women',
        'tooth': 'teeth',
        'foot': 'feet',
        'mouse': 'mice',
        'goose': 'geese',
        'scarf': 'scarves',
        'leaf': 'leaves',
        'knife': 'knives',
        'life': 'lives',
        'half': 'halves',
    }
    
    word_lower = word.lower()
    
    # Check irregular
    if word_lower in irregulars:
        return irregulars[word_lower].capitalize() if word[0].isupper() else irregulars[word_lower]
    
    # Already plural
    if word_lower.endswith('s') and len(word) > 3:
        return word
    
    # Rules
    if word_lower.endswith(('s', 'x', 'z', 'ch', 'sh')):
        return word + 'es'
    elif word_lower.endswith('y') and len(word) > 1 and word[-2] not in 'aeiou':
        return word[:-1] + 'ies'
    elif word_lower.endswith('f'):
        return word[:-1] + 'ves'
    elif word_lower.endswith('fe'):
        return word[:-2] + 'ves'
    else:
        return word + 's'


def pluralize_last_tier(hierarchy: str) -> str:
    """
    Pluralize only the last tier of a hierarchy.
    
    Args:
        hierarchy: Full hierarchy string like "Clothing > Hoodies > Basic Hoodie"
    
    Returns:
        Hierarchy with pluralized last tier
    """
    parts = [p.strip() for p in hierarchy.split('>')]
    
    if not parts:
        return hierarchy
    
    # Pluralize last part
    last_part = parts[-1]
    words = last_part.split()
    
    if words:
        # Pluralize the last word
        words[-1] = pluralize(words[-1])
        parts[-1] = ' '.join(words)
    
    return ' > '.join(parts)


# ============================================================
# Rule Generation
# ============================================================

def generate_product_type_rules(matches: Dict[str, List[int]],
                                df: pd.DataFrame,
                                client_instructions: pd.DataFrame,
                                selected_columns: List[str]) -> List[Dict]:
    """
    Generate product_type rules from keyword matches.
    
    Args:
        matches: Dictionary of keyword to matching row indices
        df: Raw data DataFrame
        client_instructions: DataFrame with product_type hierarchies
        selected_columns: Columns used for matching
    
    Returns:
        List of product_type rule dictionaries
    """
    rules = []
    
    # Assume client instructions has a column with hierarchies
    # Try to find it
    hierarchy_col = None
    for col in client_instructions.columns:
        if 'hierarchy' in col.lower() or 'product_type' in col.lower():
            hierarchy_col = col
            break
    
    if not hierarchy_col and len(client_instructions.columns) > 0:
        hierarchy_col = client_instructions.columns[0]
    
    for keyword, row_indices in matches.items():
        # Find hierarchy from client instructions
        # Match keyword in client instructions
        hierarchy = None
        for _, instr_row in client_instructions.iterrows():
            if hierarchy_col and pd.notna(instr_row[hierarchy_col]):
                hier_str = str(instr_row[hierarchy_col])
                if keyword.lower() in hier_str.lower():
                    hierarchy = hier_str
                    break
        
        if not hierarchy:
            # Default hierarchy if not found
            hierarchy = f"Products > {keyword}"
        
        # Pluralize last tier
        pluralized_hierarchy = pluralize_last_tier(hierarchy)
        
        # Build condition
        conditions = [f"[{col}] contains '{keyword}'" for col in selected_columns]
        condition = "[product_type] equal '' AND " + " AND ".join(conditions)
        
        # Create rule
        rules.append({
            "field_name": "product_type",
            "export_id": 0,
            "if": condition,
            "then": f"'{pluralized_hierarchy}'",
            "enabled": "1"
        })
    
    return rules




def generate_gpc_rules(product_type_rules: List[Dict]) -> List[Dict]:
    """
    Generate GPC rules from product_type rules.
    Matches base nouns to GPC taxonomy (similar to LLM version logic).
    
    Args:
        product_type_rules: List of product_type rules
    
    Returns:
        List of GPC rule dictionaries
    """
    import re
    import os
    
    # Load actual GPC taxonomy
    gpc_taxonomy_file = "../google_taxonomy.txt"
    if not os.path.exists(gpc_taxonomy_file):
        gpc_taxonomy_file = "google_taxonomy.txt"
    
    gpc_taxonomy = []
    if os.path.exists(gpc_taxonomy_file):
        with open(gpc_taxonomy_file, 'r', encoding='utf-8') as f:
            gpc_taxonomy = [line.strip() for line in f if line.strip()]
    
    # Extract base nouns from product_type rules (last word only)
    base_nouns = set()
    for rule in product_type_rules:
        if_cond = rule['if']
        # Find 'contains' clauses
        matches = re.findall(r"contains '([^']+)'", if_cond)
        
        for keyword in matches:
            # Extract base noun (last word)
            words = keyword.split()
            if words:
                base_noun = words[-1]  # Last word is the noun
                base_nouns.add(base_noun)
    
    base_nouns_list = sorted(list(base_nouns))
    
    # Map base nouns to GPC taxonomy (direct matching like LLM version)
    noun_to_gpc = {}
    taxonomy_set = set(gpc_taxonomy)
    
    for noun in base_nouns_list:
        noun_lower = noun.lower()
        best_match = None
        
        # Try to find exact match in GPC taxonomy
        for gpc_path in gpc_taxonomy:
            # Split GPC path into words
            gpc_words = gpc_path.replace('>', ' ').replace('&', ' ').replace(',', ' ').split()
            gpc_words_lower = [w.lower() for w in gpc_words]
            
            # Exact word match
            if noun_lower in gpc_words_lower:
                # Prefer longer, more specific paths
                if not best_match or len(gpc_path) > len(best_match):
                    best_match = gpc_path
        
        # Fuzzy fallback (like LLM version does)
        if not best_match:
            matches = [t for t in gpc_taxonomy if noun_lower in t.lower()]
            if matches:
                # Pick the most specific (longest) match
                best_match = max(matches, key=len)
        
        if best_match:
            noun_to_gpc[noun] = best_match
    
    # Debug info
    import streamlit as st
    matched_nouns = len(noun_to_gpc)
    st.info(f"🔍 GPC Matching: {matched_nouns}/{len(base_nouns_list)} keywords matched to GPC taxonomy")
    
    # Group nouns by GPC category (like LLM version)
    category_to_nouns = {}
    for noun, gpc_path in noun_to_gpc.items():
        if gpc_path not in category_to_nouns:
            category_to_nouns[gpc_path] = []
        category_to_nouns[gpc_path].append(noun)
    
    # Generate GPC rules
    gpc_rules = []
    export_id = 0
    
    for gpc_path, nouns in sorted(category_to_nouns.items()):
        if len(nouns) == 1:
            # Single noun
            condition = f"[google_product_category] equal '' AND [product_type] contains '{nouns[0]}'"
        else:
            # Multiple nouns - use contains_any
            nouns_str = ", ".join([f"'{noun}'" for noun in sorted(nouns)])
            condition = f"[google_product_category] equal '' AND [product_type] contains_any ({nouns_str})"
        
        gpc_rules.append({
            "field_name": "google_product_category",
            "export_id": export_id,
            "if": condition,
            "then": f"'{gpc_path}'",
            "enabled": "1"
        })
        export_id += 1
    
    st.success(f"Grouped {len(base_nouns_list)} nouns into {len(gpc_rules)} GPC rules")
    
    return gpc_rules





# ============================================================
# Streamlit UI
# ============================================================

st.set_page_config(page_title="Feed Optimizer (Deterministic)", layout="wide")

st.title("🎯 Feed Optimizer – Deterministic Rule Generator")
st.write("Generate Product Type & GPC Rules using keyword matching")

# Check for universal keyword file
if not Path(UNIVERSAL_KEYWORD_FILE).exists():
    st.error(f"❌ Universal keyword file not found: {UNIVERSAL_KEYWORD_FILE}")
    st.info("Please run: `python create_keyword_classes.py --input google_taxonomy.txt`")
    st.stop()

# Load keyword classes
with st.spinner("Loading universal keyword classes..."):
    keyword_classes = load_keyword_classes(UNIVERSAL_KEYWORD_FILE)
    total_keywords = sum(len(kws) for kws in keyword_classes.values())

st.success(f"✅ Loaded {len(keyword_classes)} category classes with {total_keywords} keywords")

with st.expander("📦 View Keyword Classes"):
    for category in sorted(keyword_classes.keys())[:10]:
        st.write(f"**{category}**: {', '.join(keyword_classes[category][:10])}...")

st.subheader("1. Upload Files")

client_file = st.file_uploader("Client Instructions (CSV/Excel with product_type hierarchies)", 
                               type=["csv", "xls", "xlsx"])
raw_file = st.file_uploader("Raw Data (CSV/Excel)", type=["csv", "xls", "xlsx", "txt"])

if client_file and raw_file:
    client_df = load_table(client_file)
    raw_df = load_table(raw_file)
    
    st.success(f"✅ Loaded {len(raw_df):,} rows from raw data")
    st.success(f"✅ Loaded {len(client_df):,} hierarchies from client instructions")
    
    st.subheader("2. Select Columns to Search")
    
    available_columns = raw_df.columns.tolist()
    selected_columns = st.multiselect(
        "Select columns to search for keywords",
        options=available_columns,
        help="Select one or more columns where keywords should be searched"
    )
    
    if selected_columns:
        st.info(f"Will search in: {', '.join(selected_columns)}")
        
        if st.button("🚀 Generate Rules"):
            # Step 1: Extract unique keywords from user's data columns
            with st.spinner(f"Extracting keywords from {len(selected_columns)} columns..."):
                data_keywords = extract_unique_values_from_columns(raw_df, selected_columns)
            
            st.success(f"✅ Found {len(data_keywords)} unique keywords in your data")
            
            # Step 2: Match keywords to GPC categories using universal keyword list
            with st.spinner("Matching keywords to GPC categories..."):
                # Build reverse mapping: keyword -> GPC category
                keyword_to_category = {}
                for category, keywords in keyword_classes.items():
                    for kw in keywords:
                        keyword_to_category[kw] = category
                
                # Match data keywords to categories
                keyword_category_map = {}
                for data_kw in data_keywords:
                    # Case-insensitive match
                    for gpc_kw, category in keyword_to_category.items():
                        if data_kw.lower() == gpc_kw.lower() or data_kw.lower() in gpc_kw.lower():
                            keyword_category_map[data_kw] = {
                                'category': category,
                                'gpc_keyword': gpc_kw
                            }
                            break
            
            st.success(f"✅ Matched {len(keyword_category_map)} keywords to GPC categories")
            
            # Step 3: For each keyword, lookup hierarchy in client instructions
            with st.spinner("Looking up hierarchies in client instructions..."):
                # Find hierarchy column in client instructions
                hierarchy_col = None
                for col in client_df.columns:
                    if 'hierarchy' in col.lower() or 'product_type' in col.lower():
                        hierarchy_col = col
                        break
                
                if not hierarchy_col and len(client_df.columns) > 0:
                    hierarchy_col = client_df.columns[0]
                
                # For each keyword, find hierarchy
                keyword_hierarchies = {}
                for keyword, info in keyword_category_map.items():
                    hierarchy = None
                    
                    # Try to find in client instructions
                    for _, instr_row in client_df.iterrows():
                        if hierarchy_col and pd.notna(instr_row[hierarchy_col]):
                            hier_str = str(instr_row[hierarchy_col])
                            # Match keyword or category
                            if (keyword.lower() in hier_str.lower() or 
                                info['category'].lower() in hier_str.lower()):
                                hierarchy = hier_str
                                break
                    
                    # Fallback: create hierarchy from GPC category + keyword
                    if not hierarchy:
                        # Use GPC category as base and add keyword
                        # Example: "Apparel & Accessories" + "Jacket" → "Apparel & Accessories > Jackets"
                        gpc_category = info['category']
                        # Create simple 2-tier hierarchy
                        hierarchy = f"{gpc_category} > {keyword}"
                    
                    keyword_hierarchies[keyword] = hierarchy
            
            st.success(f"✅ Found hierarchies for {len(keyword_hierarchies)} keywords")
            st.info(f"📊 Using client instructions: {sum(1 for h in keyword_hierarchies.values() if '>' in h and h not in keyword_classes.keys())} | Using GPC fallback: {sum(1 for h in keyword_hierarchies.values() if h in keyword_classes.keys())}")
            
            # Step 4: Generate product_type rules
            with st.spinner("Generating product_type rules..."):
                pt_rules = []
                for keyword, hierarchy in keyword_hierarchies.items():
                    # Pluralize last tier
                    pluralized_hierarchy = pluralize_last_tier(hierarchy)
                    
                    # Build condition
                    conditions = [f"[{col}] contains '{keyword}'" for col in selected_columns]
                    condition = "[product_type] equal '' AND " + " AND ".join(conditions)
                    
                    # Create rule
                    pt_rules.append({
                        "field_name": "product_type",
                        "export_id": 0,
                        "if": condition,
                        "then": f"'{pluralized_hierarchy}'",
                        "enabled": "1"
                    })
            
            st.success(f"✅ Generated {len(pt_rules)} product_type rules")
            
            # Generate GPC rules
            with st.spinner("Generating GPC rules..."):
                gpc_rules = generate_gpc_rules(pt_rules)
            
            st.success(f"✅ Generated {len(gpc_rules)} GPC rules")
            
            # Display and download
            st.subheader("3. Product Type Rules")
            df_pt = pd.DataFrame(pt_rules)
            st.dataframe(df_pt, use_container_width=True)
            
            pt_csv = df_pt.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download Product Type Rules", pt_csv, "product_type_rules.csv", "text/csv")
            
            st.subheader("4. GPC Rules")
            df_gpc = pd.DataFrame(gpc_rules)
            st.dataframe(df_gpc, use_container_width=True)
            
            gpc_csv = df_gpc.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download GPC Rules", gpc_csv, "gpc_rules.csv", "text/csv")
