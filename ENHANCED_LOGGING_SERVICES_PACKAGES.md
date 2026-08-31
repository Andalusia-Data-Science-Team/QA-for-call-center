# Enhanced Logging for Services and Packages

## Overview

Added detailed structured logging for retrieved services and packages, matching the format already used for offers. This provides better visibility and debugging capabilities for all CRM data layers.

## Problem

Previously, services and packages had minimal logging:
```
[INFO] fetch_crm_services_for_call | call_id=XXX — serialised 3 service(s) (456 chars)
```

No details about:
- Which services were retrieved
- Service names (Arabic/English)
- Prices
- Specialty mappings
- Whether service hint matched

## Solution

### Enhanced Service Logging

**File:** `app/agent/nodes.py` (lines 620-652)

```python
# Log detailed service information (same format as offers)
for i, svc in enumerate(services, 1):
    is_primary = not svc.get("_is_alternative", False)
    flag = "[PRIMARY]" if is_primary else "[ALTERNATIVE]"
    logger.info(
        "fetch_crm_services_for_call | call_id=%s — retrieved service #%d %s:\n"
        "  cr301_title       : %s\n"
        "  cr301_service     : %s\n"
        "  cr301_servicear   : %s\n"
        "  cr301_specialtyname: %s\n"
        "  cr301_price       : %s\n"
        "  cr18c_buname      : %s\n"
        "  _service_matched  : %s",
        call_id, i, flag,
        svc.get("cr301_title"),
        svc.get("cr301_service"),
        svc.get("cr301_servicear"),
        svc.get("cr301_specialtyname"),
        svc.get("cr301_price"),
        svc.get("cr18c_buname"),
        svc.get("_service_matched", False),
    )
```

### Enhanced Package Logging

**File:** `app/agent/nodes.py` (lines 702-734)

```python
# Log detailed package information (same format as offers)
for i, pkg in enumerate(packages, 1):
    is_primary = not pkg.get("_is_alternative", False)
    flag = "[PRIMARY]" if is_primary else "[ALTERNATIVE]"
    logger.info(
        "fetch_crm_packages_for_call | call_id=%s — retrieved package #%d %s:\n"
        "  cr301_title       : %s\n"
        "  cr301_service     : %s\n"
        "  cr301_servicear   : %s\n"
        "  cr301_specialtyname: %s\n"
        "  cr301_price       : %s\n"
        "  cr18c_buname      : %s\n"
        "  _service_matched  : %s",
        call_id, i, flag,
        pkg.get("cr301_title"),
        pkg.get("cr301_service"),
        pkg.get("cr301_servicear"),
        pkg.get("cr301_specialtyname"),
        pkg.get("cr301_price"),
        pkg.get("cr18c_buname"),
        pkg.get("_service_matched", False),
    )
```

## Logging Examples

### Example 1: Service Retrieved

```
[INFO] fetch_crm_services_for_call | call_id=3271CF9B-3C9B-F111-9B33-000D3AA9D409 specialty='Dental Services' bu=LIVE hint='خدمة التركيبه الزيركون'

[INFO] [services] direct name match: hint='خدمة التركيبه الزيركون' → cleaned='خدمة التركيبه الزيركون' → 'خدمة التركيبه الزيركون' (score=100)

[INFO] [services] Found 3 related services:
[INFO] [services]   #1: score=100 | 'خدمة التركيبه الزيركون' | Price=750.0
[INFO] [services]   #2: score=78  | 'تركيبة زيركون للسن الواحد' | Price=800.0
[INFO] [services]   #3: score=76  | 'تركيبة بجهاز سيريك' | Price=900.0

[INFO] [services] direct name match RETURNED:
  cr301_title      : خدمة التركيبه الزيركون
  cr301_service    : Zircon Crown Service
  cr301_servicear  : خدمة التركيبه الزيركون
  cr301_specialtyname: Dental Services
  cr301_price      : 750.0
  cr18c_buname     : LIVE
  match_score      : 100

[INFO] fetch_crm_services_for_call | call_id=3271CF9B-3C9B-F111-9B33-000D3AA9D409 — retrieved service #1 [PRIMARY]:
  cr301_title       : خدمة التركيبه الزيركون
  cr301_service     : Zircon Crown Service
  cr301_servicear   : خدمة التركيبه الزيركون
  cr301_specialtyname: Dental Services
  cr301_price       : 750.0
  cr18c_buname      : LIVE
  _service_matched  : True

[INFO] fetch_crm_services_for_call | call_id=3271CF9B-3C9B-F111-9B33-000D3AA9D409 — serialised 1 service(s) (234 chars)
```

### Example 2: Package Retrieved

```
[INFO] fetch_crm_packages_for_call | call_id=3271CF9B-3C9B-F111-9B33-000D3AA9D409 specialty='Dental Services' bu=LIVE hint='باقة الأسنان الكاملة'

[INFO] [packages] direct name match: hint='باقة الأسنان الكاملة' → cleaned='باقة الأسنان الكاملة' → 'باقة الأسنان الكاملة' (score=100)

[INFO] [packages] Found 2 related packages:
[INFO] [packages]   #1: score=100 | 'باقة الأسنان الكاملة' | Price=1500.0
[INFO] [packages]   #2: score=82  | 'باقة الأسنان الشاملة' | Price=1800.0

[INFO] [packages] direct name match RETURNED:
  cr301_title      : باقة الأسنان الكاملة
  cr301_service    : Complete Dental Package
  cr301_servicear  : باقة الأسنان الكاملة
  cr301_specialtyname: Dental Services
  cr301_price      : 1500.0
  cr18c_buname     : LIVE
  match_score      : 100

[INFO] fetch_crm_packages_for_call | call_id=3271CF9B-3C9B-F111-9B33-000D3AA9D409 — retrieved package #1 [PRIMARY]:
  cr301_title       : باقة الأسنان الكاملة
  cr301_service     : Complete Dental Package
  cr301_servicear   : باقة الأسنان الكاملة
  cr301_specialtyname: Dental Services
  cr301_price       : 1500.0
  cr18c_buname      : LIVE
  _service_matched  : True

[INFO] fetch_crm_packages_for_call | call_id=3271CF9B-3C9B-F111-9B33-000D3AA9D409 — serialised 1 package(s) (267 chars)
```

### Example 3: Multiple Services with Primary and Alternative

```
[INFO] fetch_crm_services_for_call | call_id=XXX — retrieved service #1 [PRIMARY]:
  cr301_title       : خدمة التركيبه الزيركون
  cr301_service     : Zircon Crown Service
  cr301_servicear   : خدمة التركيبه الزيركون
  cr301_specialtyname: Dental Services
  cr301_price       : 750.0
  cr18c_buname      : LIVE
  _service_matched  : True

[INFO] fetch_crm_services_for_call | call_id=XXX — retrieved service #2 [ALTERNATIVE]:
  cr301_title       : تركيبة زيركون للسن الواحد
  cr301_service     : Single Tooth Zircon Crown
  cr301_servicear   : تركيبة زيركون للسن الواحد
  cr301_specialtyname: Dental Services
  cr301_price       : 800.0
  cr18c_buname      : LIVE
  _service_matched  : True

[INFO] fetch_crm_services_for_call | call_id=XXX — retrieved service #3 [ALTERNATIVE]:
  cr301_title       : تركيبة بجهاز سيريك
  cr301_service     : CEREC Crown
  cr301_servicear   : تركيبة بجهاز سيريك
  cr301_specialtyname: Dental Services
  cr301_price       : 900.0
  cr18c_buname      : LIVE
  _service_matched  : True

[INFO] fetch_crm_services_for_call | call_id=XXX — serialised 3 service(s) (678 chars)
```

## Logged Information

For each retrieved service/package:

| Field | Description |
|-------|-------------|
| `call_id` | Unique identifier for the call |
| `#N` | Service/package number (1-indexed) |
| `[PRIMARY]` / `[ALTERNATIVE]` | Flag indicating if this is the primary match |
| `cr301_title` | Service/package title |
| `cr301_service` | English name |
| `cr301_servicear` | Arabic name |
| `cr301_specialtyname` | Specialty category |
| `cr301_price` | Price in SAR |
| `cr18c_buname` | Business unit (LIVE, SNB, etc.) |
| `_service_matched` | Whether direct name matching was used |

## Consistency Across All CRM Layers

| Feature | Offers | Services | Packages |
|---------|--------|----------|----------|
| Detailed logging | ✅ | ✅ | ✅ |
| Primary/Alternative flags | ✅ | ✅ | ✅ |
| All field details | ✅ | ✅ | ✅ |
| Match score info | ✅ | ✅ | ✅ |
| Related items shown | ✅ | ✅ | ✅ |

## Benefits

✅ **Complete visibility** - See exactly what was retrieved  
✅ **Debugging made easy** - All details in one place  
✅ **Consistent format** - Same structure across offers/services/packages  
✅ **Match transparency** - Know if direct name matching was used  
✅ **Price verification** - Quickly verify prices match expectations  
✅ **Specialty validation** - Confirm correct specialty mapping  

## Use Cases

### Debugging
When investigating why a specific service/package was returned:
1. Check the logs for the call_id
2. Find the `retrieved service/package` log entry
3. Verify all fields match expectations
4. Check `_service_matched` to see if direct name matching was used

### Quality Assurance
When validating system behavior:
1. Search logs for specific service/package names
2. Verify prices are correct
3. Confirm specialty mappings
4. Check primary/alternative flagging

### Performance Monitoring
Track which services/packages are being retrieved most often:
1. Grep for "retrieved service" or "retrieved package"
2. Count occurrences by name
3. Monitor price changes over time

## Files Modified

**`app/agent/nodes.py`** (2 sections)
- Lines 620-652: Enhanced service logging
- Lines 702-734: Enhanced package logging

## No Breaking Changes

✅ Only logging changes - no API changes  
✅ No configuration changes required  
✅ No database changes  
✅ No dependency changes  
✅ Backward compatible  

## Related Documentation

- `OFFER_MATCHING_IMPROVEMENTS.md` - Offer normalization
- `SERVICES_PACKAGES_DIRECT_LOOKUP.md` - Services & packages normalization
- `COMPLETE_ENHANCEMENT_SUMMARY.md` - Complete overview

## Example Log Grep Commands

```bash
# Find all retrieved services for a specific call
grep "retrieved service" logs/qa_system.log | grep "call_id=3271CF9B"

# Find all services with a specific name
grep "retrieved service" logs/qa_system.log | grep "زيركون"

# Find all packages above a certain price
grep "retrieved package" logs/qa_system.log | grep "cr301_price.*: [2-9][0-9][0-9][0-9]"

# Find all primary matches
grep "retrieved.*PRIMARY" logs/qa_system.log

# Find all direct name matches
grep "_service_matched.*: True" logs/qa_system.log
```

## Summary

Services and packages now have the same detailed, structured logging as offers, providing complete visibility into what was retrieved and making debugging and quality assurance significantly easier.
