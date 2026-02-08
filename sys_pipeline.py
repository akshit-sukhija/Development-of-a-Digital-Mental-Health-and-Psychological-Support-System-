"""
HealthNav AI – High-Level Processing Pipeline
Non-diagnostic guidance flow
"""

def process_user_input(text, sentiment_fn, support_mapper):
    sentiment, confidence = sentiment_fn(text)
    support_level, message = support_mapper(text, sentiment)

    return {
        "input": text,
        "sentiment": sentiment,
        "confidence": confidence,
        "support_level": support_level,
        "message": message
    }
