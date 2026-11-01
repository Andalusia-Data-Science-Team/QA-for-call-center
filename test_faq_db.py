"""Test script to verify FAQ database connectivity and query."""

from app.FAQs.faq_validation import lookup_faq_record, detect_faq_escalation

# Test database connection
def test_faq_lookup():
    """Test FAQ record lookup from SQL database."""
    result = lookup_faq_record(
        patient_phone="555555555",
        call_date="2024-01-01",
        agent_name="Test Agent",
        agent_email="test@example.com",
        db_key="SERVICE_HUB"
    )
    
    print("FAQ Lookup Result:")
    print(f"Status: {result.get('status')}")
    print(f"Message: {result.get('message')}")
    if result.get('record'):
        print(f"Record ID: {result['record'].get('ID')}")
        print(f"Inquiry: {result['record'].get('Inquiry')}")
        print(f"CST Inquiry: {result['record'].get('CST Inquiry / Request Details')}")
    print()

def test_escalation_detection():
    """Test escalation detection in transcript."""
    # Test with agent saying escalation
    transcript1 = """Agent: تم تقديم طلب للقسم المختص
Patient: شكراً"""
    
    print("Test 1 - Agent mentions escalation:")
    print(f"Escalation detected: {detect_faq_escalation(transcript1)}")
    print()
    
    # Test with patient saying it (should not detect)
    transcript2 = """Patient: هل تم تقديم الطلب؟
Agent: نعم سيتم المراجعة"""
    
    print("Test 2 - Patient asks about escalation:")
    print(f"Escalation detected: {detect_faq_escalation(transcript2)}")
    print()
    
    # Test with no labels (should check full text)
    transcript3 = "تم رفع الطلب للجهة المختصة"
    
    print("Test 3 - No labels:")
    print(f"Escalation detected: {detect_faq_escalation(transcript3)}")
    print()

if __name__ == "__main__":
    print("=" * 60)
    print("FAQ Database Migration Test")
    print("=" * 60)
    print()
    
    print("Testing escalation detection...")
    print("-" * 60)
    test_escalation_detection()
    
    print("\nTesting FAQ lookup...")
    print("-" * 60)
    test_faq_lookup()
    
    print("=" * 60)
    print("Test complete!")
