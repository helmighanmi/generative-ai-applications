from pdf_classifier_implementation import *

"""
# Quick start with original approach
hybrid = HybridClassifier()
result = hybrid.classify(page)  # PageType.DOCUMENT or DIAGRAM

# Improved approach with calibration
improved = ImprovedPDFClassifier()
improved.calibrate([page1, page2, page3, page4, page5])  # 5 sample pages
result = improved.classify(page)  # Includes uncertainty + anomaly detection
"""


# Example usage
doc = fitz.open("2408.09869v5.pdf")

# Original approach
hybrid = HybridClassifier()

# Improved approach - BINARY ONLY
improved = ImprovedPDFClassifier(calibration_pages=5)
improved.calibrate([doc[i] for i in range(min(5, len(doc)))])

for i in range(len(doc)):
    page = doc[i]
    
    orig = hybrid.classify(page)
    imp = improved.classify(page)
    
    # Both return only DOCUMENT or DIAGRAM (no AMBIGUOUS)
    print(f"Page {i+1}: Original={orig.page_type.name}, Improved={imp.page_type.name}, "
            f"Confidence={imp.confidence:.2f}")

# Get confidence statistics
stats = improved.get_confidence_stats()
print(f"\nStats: {stats}")

doc.close()