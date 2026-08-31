#!/usr/bin/env python3
"""
Test script for offer name normalization
Demonstrates how the enhanced matching handles extracted offer names with branch prefixes and prices.
"""
import re


def normalize_offer_hint(hint: str) -> str:
    """Normalize extracted offer names for better matching."""
    # First, remove newlines and normalize whitespace
    text = re.sub(r'[\r\n]+', ' ', hint)
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove trailing prices: numbers with SAR/ريال/SR or standalone numbers at end
    text = re.sub(r'\s+\d+[\s\.]*(ريال|SAR|SR|جنيه|دينار|درهم)\s*$', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s+\d{2,}[\s\.]*$', '', text)  # "... 750" → "..."
    
    # Remove branch name patterns at the beginning
    # Pattern: "فرع السنابل" or "فرع X" at start
    text = re.sub(r'^فرع\s+\S+\s+', '', text)  # "فرع السنابل عرض..." → "عرض..."
    
    # Alternative patterns for branches
    text = re.sub(r'^(في|فى)\s+فرع\s+\S+\s+', '', text)  # "في فرع X عرض..." → "عرض..."
    
    return text.strip().lower()


def norm_ar_fast(text: str) -> str:
    """Quick Arabic normalization for matching."""
    t = text.lower()
    # Remove tashkeel
    t = re.sub(r'[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06DC\u06DF-\u06E4\u06E7\u06E8\u06EA-\u06ED]', '', t)
    # Normalize hamza
    t = t.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا").replace("ٱ", "ا")
    t = t.replace("ؤ", "و").replace("ئ", "ي")
    t = t.replace("ة", "ه").replace("ى", "ي")
    return t


def test_normalization():
    """Test the normalization with the provided example."""
    
    # Test case: extracted offer name with branch prefix and price suffix
    extracted = "فرع السنابل عرض التركيبه الزيركون\r\n 750"
    
    # Database offer names (examples of what might be in the database)
    db_offers = [
        "عرض التركيبه الزيركون",
        "تركيبة للسن الواحد (زيركون) أو الايماكس",
        "تركيبة / تلبيسة بجهاز سيريك ( زيركون أو ايماكس) 1 visit",
        "عرض الزيركون الكامل",
        "باقة تركيبات الأسنان زيركون",
    ]
    
    print("=" * 80)
    print("OFFER NAME NORMALIZATION TEST")
    print("=" * 80)
    print(f"\nExtracted offer name:")
    print(f"  Raw: {extracted!r}")
    
    # Step 1: Normalize
    cleaned = normalize_offer_hint(extracted)
    print(f"  Cleaned: {cleaned!r}")
    
    # Step 2: Arabic normalization
    normalized = norm_ar_fast(cleaned)
    print(f"  Normalized: {normalized!r}")
    
    print(f"\n{'Database Offer':<60} {'Match Score':<12} {'Method'}")
    print("-" * 80)
    
    # Test matching with rapidfuzz (simulate what the code does)
    try:
        from rapidfuzz import fuzz
        
        for offer in db_offers:
            offer_norm = norm_ar_fast(offer.lower())
            
            # Try different matching strategies
            score1 = fuzz.partial_ratio(cleaned, offer.lower())
            score2 = fuzz.partial_ratio(normalized, offer_norm)
            score3 = fuzz.token_set_ratio(normalized, offer_norm)
            
            best_score = max(score1, score2, score3)
            method = "partial" if best_score == score1 else "partial_norm" if best_score == score2 else "token_set"
            
            match_indicator = "✓" if best_score >= 75 else "✗"
            print(f"{match_indicator} {offer:<57} {best_score:>5.1f}        {method}")
    
    except ImportError:
        print("\nrapi dfuzz not installed. Install with: pip install rapidfuzz")
        print("\nShowing normalized forms only:")
        for offer in db_offers:
            offer_norm = norm_ar_fast(offer.lower())
            print(f"  {offer:<60}")
            print(f"    → {offer_norm}")
    
    print("\n" + "=" * 80)
    print("EXPLANATION:")
    print("=" * 80)
    print("""
The normalization process:
1. Removes branch prefix "فرع السنابل" 
2. Removes price suffix "750"
3. Removes newlines (\\r\\n)
4. Normalizes Arabic characters (hamza variants, ta marbuta, etc.)
5. Converts to lowercase for case-insensitive matching

This allows the extracted offer name to match any of the related database offers
that contain "زيركون" (zircon) even if the exact phrasing differs.

The matching uses multiple strategies:
- partial_ratio: Finds best substring match
- partial_ratio_norm: Same but with Arabic normalization
- token_set_ratio: Word order independent matching

Threshold: 75+ score is considered a match (lowered from 85 for normalized matching)
""")


if __name__ == "__main__":
    test_normalization()
