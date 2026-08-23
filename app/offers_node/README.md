# Offers Node — Standalone Package
# ==================================
# A self-contained extract of the offers logic from the Andalusia Contact
# Center chatbot, ready to share with team members.

## Files

| File | Purpose |
|---|---|
| `intent_detector.py` | Determine **what the patient wants**: proactive check or offer inquiry |
| `offer_search.py` | **Find the right offer**: resolve specialty → extract hint → search CRM |
| `crm_offers.py` | **CRM data layer** (copy from `services/crm_offers.py` in the main repo) |
| `README.md` | This file |

---

## How It Works

### Step 1 — Detect intent

```python
from intent_detector import detect_offer_intent

intent = detect_offer_intent(user_text, state)
# Returns: "proactive" | "inquiry" | "none"
```

| Intent | When | How triggered |
|---|---|---|
| `"proactive"` | Slot already confirmed (`slot_confirmed=True`) | Automatic — no keyword needed |
| `"inquiry"` | Patient message contains offer keywords | Keywords: عرض, خصم, باقة, offer, discount, promo... |
| `"none"` | Neither condition met | Skip offers entirely |

---

### Step 2A — Proactive check (slot confirmed)

The patient just picked a doctor and time slot.  
Check CRM for a matching active offer and show it.

```python
from offer_search import search_offers, format_offer_response

specialty_en = state.get("speciality")          # e.g. "Dermatology and Cosmatology"
service_hint = state.get("service_hint") or ""  # e.g. "بلازما" (if extracted earlier)

offers = search_offers(
    specialty_en=specialty_en,
    service_hint=service_hint,
    patient_gender=state.get("patient_gender"),
    patient_age=state.get("patient_age"),
)

if offers and offers[0].get("Offer_Status") == "Active":
    msg = format_offer_response(offers[0], lang="ar")
    # → Show offer card and ask patient to confirm
else:
    # → No active offer, pass through to booking
    pass
```

---

### Step 2B — Patient inquiry (explicit keyword)

The patient asked: "هل في عروض على الأسنان؟"

```python
from offer_search import (
    detect_ambiguous_service,
    resolve_specialty,
    extract_service_hint,
    search_offers,
    format_offer_response,
)

user_text = "هل في عروض على الليزر؟"

# Check for ambiguous service first (ليزر = skin OR eye?)
ambiguous_options = detect_ambiguous_service(user_text)
if ambiguous_options:
    # Ask patient to choose:
    # "1. ليزر الجلد (جلدية)  2. ليزر العيون / ليزك (عيون)"
    pass
else:
    specialty_en = resolve_specialty(user_text)  # → "Ophthalmology" or "Dermatology"
    hint = extract_service_hint(user_text)        # → "الليزر"

    if specialty_en:
        offers = search_offers(specialty_en, hint)
        if offers:
            print(format_offer_response(offers[0], lang="ar"))
        else:
            print("ما في عروض متاحة حالياً لهذه الخدمة.")
    else:
        # Specialty unknown — ask patient which specialty they mean
        print("تخصص إيش تبغى تعرف عروضه؟")
```

---

## Disambiguation Flow

When `detect_ambiguous_service()` returns options, show a numbered list and
wait for the patient's choice:

```
Patient: "في عروض على ليزر؟"
Bot:     "هذه الخدمة متوفرة في أكثر من تخصص. أي تخصص يناسبك؟
          1. ليزر الجلد (جلدية)
          2. ليزر العيون / ليزك (عيون)"
Patient: "1"
→ specialty_en = "Dermatology"
→ search_offers("Dermatology", "ليزر")
```

---

## Setup

### Requirements

```
pip install requests pyodbc
```

### CRM dependency

Copy `services/crm_offers.py` from the main repo into this directory.  
It connects to the Dynamics 365 CRM (read-only) and caches results for 30 min.

Configure these environment variables (same as `.env` in the main repo):

```bash
# SQL Server for hospital DB
DB_SERVER=...
DB_DATABASE=...
DB_USERNAME=...
DB_PASSWORD=...

# CRM auth (Azure AD / MSAL)
CRM_SERVER=org2f45e702.crm4.dynamics.com,5558
CRM_CLIENT_ID=51f81489-12ee-4a9e-aaae-a2591f45987d
CRM_USERNAME=...
CRM_PASSWORD=...
```

### Quick test

```python
from offer_search import resolve_specialty, extract_service_hint, search_offers

# Should resolve to "Dental Services"
print(resolve_specialty("في عروض على الاسنان؟"))   # → Dental Services

# Should extract "الليزر"
print(extract_service_hint("هل في عروض بخصوص الليزر؟"))  # → الليزر

# Live CRM search (requires CRM connection)
offers = search_offers("Dermatology and Cosmatology", "بلازما")
for o in offers:
    print(o.get("Offer_Name_AR"), o.get("Offer_Status"))
```

---

## Offer Status Handling

| CRM Status | What to do |
|---|---|
| `Active` | Show offer card, ask patient to book with it |
| `Inactive` | Only surface if patient explicitly asked — "هذا العرض انتهى، نكمل بدونه؟" |
| `Draft` | Never show to patient |
