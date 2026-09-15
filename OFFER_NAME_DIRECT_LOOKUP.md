# Direct Offer Name Lookup Enhancement

## Problem
Previously, the system required a specialty to be extracted from the transcript before fetching offers:

```
2026-08-24 13:38:00,746 | INFO | app.agent.nodes | fetch_crm_offers_for_call | 
call_id=3271CF9B-3C9B-F111-9B33-000D3AA9D409 — no specialty extracted from transcript, 
skipping CRM fetch
```

**Issue**: When the agent mentions a specific offer name (e.g., "فرع السنابل عرض التركيبه الزيركون 750") but doesn't mention the specialty, the system would skip the CRM fetch entirely.

## Solution

### 1. Enhanced Node Logic (`fetch_crm_offers_for_call`)

**File**: `app/agent/nodes.py` (lines 461-480)

The node now checks for **both** specialty AND offer name:

```python
# Check if offer name is directly mentioned in the transcript
offer_name_hint: str = (details.get("offer_name") or "").strip()

# If no specialty but offer name is mentioned, we can still proceed
# The enhanced fast-path matching will handle the direct offer name lookup
if not specialty_en and not offer_name_hint:
    logger.info(
        "fetch_crm_offers_for_call | call_id=%s — no specialty or offer name "
        "extracted from transcript, skipping CRM fetch",
        call_id,
    )
    return {
        "crm_offers_context": "",
        "node_trace": _trace(state, "fetch_crm_offers_for_call"),
    }

if not specialty_en and offer_name_hint:
    logger.info(
        "fetch_crm_offers_for_call | call_id=%s — no specialty extracted but "
        "offer name mentioned: %r, proceeding with direct name lookup",
        call_id, offer_name_hint,
    )
```

### 2. Generic Specialty Fallback

When an offer name is mentioned but specialty is missing, use a generic specialty value:

```python
# When offer name is mentioned but specialty is missing, use a generic
# specialty value that won't filter results. The direct name lookup
# (fast-path) will find the offer regardless of specialty.
effective_specialty = specialty_en or "General"

offers = get_offers_for_specialty(
    specialty_en=effective_specialty,
    service_hint=offer_name_hint,  # direct offer name if mentioned
    specialty_only=not bool(offer_name_hint),
    patient_gender=patient_gender,
)
```

### 3. How It Works with Fast-Path Matching

The enhanced fast-path matching in `crm_offers.py` (from previous improvement) handles the direct name lookup:

1. **Offer name mentioned**: `"فرع السنابل عرض التركيبه الزيركون 750"`
2. **No specialty extracted**: `specialty_en = ""`
3. **New behavior**: Uses `effective_specialty = "General"` to proceed
4. **Fast-path kicks in**: 
   - Normalizes the offer name: removes branch prefix, price suffix
   - Result: `"عرض التركيبه الزيركون"`
   - Searches all offers across specialties with this cleaned name
   - Finds matches regardless of specialty
5. **Returns matching offers**: All "زيركون" related offers found

## Flow Comparison

### Before (Fails)
```
Transcript: "Agent mentioned عرض التركيبه الزيركون from فرع السنابل for 750 SAR"
    ↓
Extract: offer_name = "فرع السنابل عرض التركيبه الزيركون 750"
         specialty_en = ""  (not mentioned)
    ↓
Check: specialty_en is empty?
    ↓
❌ SKIP: "no specialty extracted from transcript, skipping CRM fetch"
    ↓
Result: No offers returned, evaluation fails
```

### After (Works)
```
Transcript: "Agent mentioned عرض التركيبه الزيركون from فرع السنابل for 750 SAR"
    ↓
Extract: offer_name = "فرع السنابل عرض التركيبه الزيركون 750"
         specialty_en = ""  (not mentioned)
    ↓
Check: specialty_en is empty BUT offer_name is present?
    ↓
✅ PROCEED: "no specialty but offer name mentioned, proceeding with direct lookup"
    ↓
Normalize: "فرع السنابل عرض التركيبه الزيركون 750"
           → "عرض التركيبه الزيركون"
    ↓
Search: Fast-path direct name matching across ALL specialties
    ↓
Find: • عرض التركيبه الزيركون (100% match)
      • تركيبة للسن الواحد (زيركون) أو الايماكس (78% match)
      • تركيبة / تلبيسة بجهاز سيريك ( زيركون أو ايماكس) (76% match)
    ↓
Result: ✅ Offers returned, evaluation proceeds successfully
```

## Benefits

✅ **No specialty required** when offer name is mentioned  
✅ **Fast-path matching** finds offers across all specialties  
✅ **Backward compatible** - still uses specialty when available  
✅ **Better logging** - clear messages for each case  
✅ **Handles noisy names** - branch prefixes and prices removed  

## Logging Examples

### Case 1: Offer Name Only (No Specialty)
```
[INFO] fetch_crm_offers_for_call | call_id=XXX — no specialty extracted but 
       offer name mentioned: 'فرع السنابل عرض التركيبه الزيركون 750', 
       proceeding with direct name lookup

[INFO] direct name match: hint='فرع السنابل عرض التركيبه الزيركون 750' 
       → cleaned='عرض التركيبه الزيركون' 
       → 'عرض التركيبه الزيركون' (score=100)

[INFO] fetch_crm_offers_for_call | call_id=XXX — serialised 3 offer(s)
```

### Case 2: Both Specialty and Offer Name
```
[INFO] fetch_crm_offers_for_call | call_id=XXX specialty='Dental Services' gender=unknown

[INFO] direct name match: hint='عرض التركيبه الزيركون' 
       → cleaned='عرض التركيبه الزيركون' 
       → 'عرض التركيبه الزيركون' (score=100)

[INFO] fetch_crm_offers_for_call | call_id=XXX — serialised 1 offer(s)
```

### Case 3: Neither Specialty Nor Offer Name
```
[INFO] fetch_crm_offers_for_call | call_id=XXX — no specialty or offer name 
       extracted from transcript, skipping CRM fetch
```

## Files Modified

1. **`app/agent/nodes.py`** (lines 461-545)
   - Added offer name check alongside specialty check
   - Uses `effective_specialty = specialty_en or "General"`
   - Enhanced logging for each case

2. **`app/offers_node/crm_offers.py`** (enhanced in previous improvement)
   - Fast-path direct name matching with normalization
   - Multi-strategy fuzzy matching
   - Word-order independent matching

## Testing

Test with transcript containing offer name but no specialty:

```python
state = {
    "appointment_details": {
        "offer_name": "فرع السنابل عرض التركيبه الزيركون 750",
        "specialty_name": "",  # Empty - not mentioned
    }
}

# Should now proceed with direct name lookup and find matching offers
```

## Configuration

No configuration changes required. The enhancement is automatic and backward compatible.
