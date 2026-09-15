# Quick Reference: Offer Name Normalization

## Problem
```
Extracted: "فرع السنابل عرض التركيبه الزيركون\r\n 750"
Database: "عرض التركيبه الزيركون"
         "تركيبة للسن الواحد (زيركون) أو الايماكس"
         "تركيبة / تلبيسة بجهاز سيريك ( زيركون أو ايماكس) 1 visit"
```
**Issue**: Extracted name has branch prefix + price suffix → fails to match

## Solution
Enhanced normalization in `app/offers_node/crm_offers.py`:

### Normalization Steps
1. Remove `\r\n` → normalize whitespace
2. Remove trailing prices (`750`, `500 ريال`)
3. Remove branch prefixes (`فرع السنابل`)
4. Normalize Arabic chars (أ→ا, ة→ه, ى→ي)
5. Lowercase for case-insensitive match

### Matching Strategies
```python
# Uses 5 different matching methods, takes best score:
1. partial_ratio(cleaned_hint, offer_ar)
2. partial_ratio(cleaned_hint, offer_en)
3. partial_ratio(normalized_ar, offer_ar_normalized)
4. token_set_ratio(normalized_ar, offer_ar_normalized)  # word order independent
5. token_set_ratio(cleaned_hint, offer_en)
```

### Threshold
- **Before**: 85 (strict)
- **After**: 75 (lenient for normalized matching)

## Example Flow
```
Input:  "فرع السنابل عرض التركيبه الزيركون\r\n 750"
    ↓
Step 1: Remove \r\n, normalize whitespace
    → "فرع السنابل عرض التركيبه الزيركون 750"
    ↓
Step 2: Remove price "750"
    → "فرع السنابل عرض التركيبه الزيركون"
    ↓
Step 3: Remove branch "فرع السنابل"
    → "عرض التركيبه الزيركون"
    ↓
Step 4: Normalize Arabic (already normalized in this case)
    → "عرض التركيبه الزيركون"
    ↓
Result: MATCH! (Score: 100)
```

## Code Location
**File**: `app/offers_node/crm_offers.py`
**Function**: `get_offers_for_specialty()` 
**Section**: "Direct offer-name lookup (fast path)" (around line 456)

## Testing
```bash
cd /home/ai/Workspace/Rafik/QA_System-main
python3 test_offer_normalization.py
```

## Logging Output
```
[offers] direct name match: hint='فرع السنابل عرض التركيبه الزيركون\r\n 750' 
         → cleaned='عرض التركيبه الزيركون' 
         → 'عرض التركيبه الزيركون' (score=100)

[offers] Found 3 related offers:
[offers]   #1: score=100 | 'عرض التركيبه الزيركون' | Status=Active | Price=750
[offers]   #2: score=78  | 'تركيبة للسن الواحد (زيركون) أو الايماكس' | Status=Active | Price=800
[offers]   #3: score=76  | 'تركيبة / تلبيسة بجهاز سيريك' | Status=Active | Price=900
```

## Key Benefits
✅ Handles noisy extracted names  
✅ Finds all related offers (e.g., all "زيركون" variants)  
✅ Word-order independent matching  
✅ Arabic character normalization  
✅ Detailed logging for debugging  
✅ Backward compatible  

## Next Steps
1. Test with your actual database offers
2. Monitor logs for matching scores
3. Adjust threshold (75) if needed based on real data
4. Consider adding semantic matching as fallback
