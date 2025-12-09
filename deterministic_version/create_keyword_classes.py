#!/usr/bin/env python
"""
GPC Keyword Class Generator - Create Keyword Categories

This script parses the Google Product Category (GPC) taxonomy and creates
keyword classes/categories for easy reference and matching.

Usage:
    python create_keyword_classes.py --input google_taxonomy.txt --output keyword_classes.csv

Output Format:
    category_class,keywords,gpc_path
    Apparel,"Shirts,Tops,Dresses,Pants,Jackets",Apparel & Accessories > Clothing
    Electronics,"Laptops,Phones,Tablets,Computers",Electronics
"""

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Set
import re


def parse_gpc_taxonomy(gpc_file: Path) -> Dict[str, Set[str]]:
    """
    Parse GPC taxonomy and group keywords by top-level category.
    
    Args:
        gpc_file: Path to Google taxonomy TXT file
    
    Returns:
        Dictionary mapping category class to set of keywords
    """
    category_keywords = {}
    
    with open(gpc_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            
            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue
            
            # Split by '>' to get hierarchy
            parts = [p.strip() for p in line.split('>')]
            
            if not parts:
                continue
            
            # Top-level category is the class
            top_category = parts[0]
            
            # Initialize category if not exists
            if top_category not in category_keywords:
                category_keywords[top_category] = {
                    'keywords': set(),
                    'full_paths': set()
                }
            
            # Extract keywords from all levels
            for part in parts:
                # Split by '&', ',', '/' to get individual items
                items = re.split(r'[&,/]', part)
                
                for item in items:
                    words = item.strip().split()
                    
                    # Skip common words
                    skip_words = {'and', 'or', 'the', 'a', 'an', 'for', 'with', 'by', 'accessories'}
                    
                    for word in words:
                        cleaned = word.strip()
                        if (cleaned and 
                            cleaned.lower() not in skip_words and 
                            len(cleaned) > 2 and
                            not cleaned.isdigit()):
                            category_keywords[top_category]['keywords'].add(cleaned)
            
            # Store full path
            category_keywords[top_category]['full_paths'].add(line)
    
    return category_keywords


def save_keyword_classes(category_keywords: Dict, output_file: Path):
    """
    Save keyword classes to CSV file.
    
    Args:
        category_keywords: Dictionary of category to keywords mapping
        output_file: Path to output CSV file
    """
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['category_class', 'keyword_count', 'keywords', 'sample_gpc_paths'])
        
        # Sort by category name
        for category in sorted(category_keywords.keys()):
            data = category_keywords[category]
            keywords = sorted(data['keywords'])
            sample_paths = list(data['full_paths'])[:3]  # First 3 paths as samples
            
            writer.writerow([
                category,
                len(keywords),
                ','.join(keywords),
                ' | '.join(sample_paths)
            ])


def print_summary(category_keywords: Dict):
    """Print summary of keyword classes."""
    print("\n" + "="*80)
    print("KEYWORD CLASSES SUMMARY")
    print("="*80)
    
    for category in sorted(category_keywords.keys()):
        data = category_keywords[category]
        keywords = sorted(data['keywords'])
        
        print(f"\n📦 {category}")
        print(f"   Keywords ({len(keywords)}): {', '.join(keywords[:10])}")
        if len(keywords) > 10:
            print(f"   ... and {len(keywords) - 10} more")


def main():
    parser = argparse.ArgumentParser(
        description='Create keyword classes from GPC taxonomy'
    )
    parser.add_argument(
        '--input',
        type=Path,
        required=True,
        help='Path to Google Product Taxonomy TXT file'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default='keyword_classes.csv',
        help='Path to output CSV file (default: keyword_classes.csv)'
    )
    
    args = parser.parse_args()
    
    if not args.input.exists():
        print(f"❌ Error: Input file not found: {args.input}")
        return 1
    
    print(f"📖 Reading GPC taxonomy from: {args.input}")
    category_keywords = parse_gpc_taxonomy(args.input)
    
    print(f"\n✅ Extracted {len(category_keywords)} category classes")
    
    print(f"\n💾 Saving keyword classes to: {args.output}")
    save_keyword_classes(category_keywords, args.output)
    
    print_summary(category_keywords)
    
    print(f"\n✅ Keyword classes file created successfully!")
    print(f"📄 Output: {args.output}")
    
    return 0


if __name__ == '__main__':
    exit(main())
