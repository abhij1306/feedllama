#!/usr/bin/env python
"""
Product Type Repository Generator

This script creates a universal product_type repository file that can be reused
across multiple projects, similar to the Google Product Taxonomy.

Usage:
    python create_product_type_repository.py --output product_type_repository.txt

Output Format (one hierarchy per line):
    Clothing > Tops > T-Shirts
    Clothing > Bottoms > Jeans
    Electronics > Computers > Laptops
"""

import argparse
from pathlib import Path


def create_default_repository():
    """
    Create a default product_type repository with common categories.
    Users can edit this file to add their own custom hierarchies.
    """
    
    categories = [
        # Apparel & Clothing
        "Clothing > Tops > T-Shirts",
        "Clothing > Tops > Shirts",
        "Clothing > Tops > Blouses",
        "Clothing > Tops > Tank Tops",
        "Clothing > Tops > Sweaters",
        "Clothing > Tops > Hoodies",
        "Clothing > Tops > Sweatshirts",
        "Clothing > Bottoms > Jeans",
        "Clothing > Bottoms > Pants",
        "Clothing > Bottoms > Shorts",
        "Clothing > Bottoms > Skirts",
        "Clothing > Bottoms > Leggings",
        "Clothing > Dresses > Casual Dresses",
        "Clothing > Dresses > Formal Dresses",
        "Clothing > Dresses > Maxi Dresses",
        "Clothing > Outerwear > Jackets",
        "Clothing > Outerwear > Coats",
        "Clothing > Outerwear > Blazers",
        "Clothing > Outerwear > Vests",
        "Clothing > Activewear > Sports Bras",
        "Clothing > Activewear > Yoga Pants",
        "Clothing > Activewear > Running Shorts",
        "Clothing > Activewear > Athletic Tops",
        
        # Footwear
        "Footwear > Shoes > Sneakers",
        "Footwear > Shoes > Boots",
        "Footwear > Shoes > Sandals",
        "Footwear > Shoes > Heels",
        "Footwear > Shoes > Flats",
        "Footwear > Shoes > Loafers",
        "Footwear > Athletic Shoes > Running Shoes",
        "Footwear > Athletic Shoes > Training Shoes",
        "Footwear > Athletic Shoes > Basketball Shoes",
        
        # Accessories
        "Accessories > Bags > Handbags",
        "Accessories > Bags > Backpacks",
        "Accessories > Bags > Tote Bags",
        "Accessories > Bags > Crossbody Bags",
        "Accessories > Jewelry > Necklaces",
        "Accessories > Jewelry > Earrings",
        "Accessories > Jewelry > Bracelets",
        "Accessories > Jewelry > Rings",
        "Accessories > Hats > Baseball Caps",
        "Accessories > Hats > Beanies",
        "Accessories > Hats > Sun Hats",
        "Accessories > Belts > Leather Belts",
        "Accessories > Belts > Fabric Belts",
        "Accessories > Scarves > Silk Scarves",
        "Accessories > Scarves > Winter Scarves",
        
        # Electronics
        "Electronics > Computers > Laptops",
        "Electronics > Computers > Desktops",
        "Electronics > Computers > Tablets",
        "Electronics > Mobile Devices > Smartphones",
        "Electronics > Mobile Devices > Smart Watches",
        "Electronics > Audio > Headphones",
        "Electronics > Audio > Earbuds",
        "Electronics > Audio > Speakers",
        "Electronics > Cameras > DSLR Cameras",
        "Electronics > Cameras > Mirrorless Cameras",
        "Electronics > Cameras > Action Cameras",
        
        # Home & Garden
        "Home & Garden > Furniture > Sofas",
        "Home & Garden > Furniture > Chairs",
        "Home & Garden > Furniture > Tables",
        "Home & Garden > Furniture > Beds",
        "Home & Garden > Decor > Wall Art",
        "Home & Garden > Decor > Vases",
        "Home & Garden > Decor > Candles",
        "Home & Garden > Kitchen > Cookware",
        "Home & Garden > Kitchen > Dinnerware",
        "Home & Garden > Kitchen > Utensils",
        "Home & Garden > Bedding > Sheets",
        "Home & Garden > Bedding > Comforters",
        "Home & Garden > Bedding > Pillows",
        
        # Beauty & Personal Care
        "Beauty > Makeup > Foundation",
        "Beauty > Makeup > Lipstick",
        "Beauty > Makeup > Mascara",
        "Beauty > Makeup > Eyeshadow",
        "Beauty > Skincare > Moisturizers",
        "Beauty > Skincare > Cleansers",
        "Beauty > Skincare > Serums",
        "Beauty > Haircare > Shampoo",
        "Beauty > Haircare > Conditioner",
        "Beauty > Haircare > Hair Styling Products",
        "Beauty > Fragrances > Perfumes",
        "Beauty > Fragrances > Colognes",
        
        # Sports & Outdoors
        "Sports > Equipment > Yoga Mats",
        "Sports > Equipment > Dumbbells",
        "Sports > Equipment > Resistance Bands",
        "Sports > Outdoor Gear > Tents",
        "Sports > Outdoor Gear > Sleeping Bags",
        "Sports > Outdoor Gear > Backpacks",
        "Sports > Team Sports > Basketballs",
        "Sports > Team Sports > Soccer Balls",
        "Sports > Team Sports > Footballs",
    ]
    
    return categories


def save_repository(categories, output_file):
    """Save product_type repository to file."""
    with open(output_file, 'w', encoding='utf-8') as f:
        for category in sorted(categories):
            f.write(category + '\n')


def main():
    parser = argparse.ArgumentParser(
        description='Create universal product_type repository'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default='product_type_repository.txt',
        help='Path to output file (default: product_type_repository.txt)'
    )
    
    args = parser.parse_args()
    
    print("📦 Creating product_type repository...")
    categories = create_default_repository()
    
    print(f"💾 Saving {len(categories)} product type hierarchies to: {args.output}")
    save_repository(categories, args.output)
    
    print("✅ Product type repository created successfully!")
    print(f"\n📄 File: {args.output}")
    print(f"📊 Total hierarchies: {len(categories)}")
    print("\n💡 You can edit this file to add your own custom product type hierarchies.")
    print("   Format: Category > Subcategory > Product Type")
    
    return 0


if __name__ == '__main__':
    exit(main())
