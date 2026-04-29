"""
cogops/prompts/messages.py

User-facing fallback strings. Centralized here so the reasoning loop and
orchestrator can emit consistent messages without taking a config dependency.
"""

ERROR_FALLBACK_BN = (
    "একটি প্রযুক্তিগত ত্রুটির কারণে আমি এই মুহূর্তে সাহায্য করতে পারছি না। "
    "অনুগ্রহ করে কিছুক্ষণ পর আবার চেষ্টা করুন।"
)

SERVER_LOAD_FALLBACK_BN = (
    "এই মুহূর্তে সার্ভারে অতিরিক্ত চাপ থাকার কারণে আমি আপনার প্রশ্নের সম্পূর্ণ "
    "উত্তর প্রস্তুত করতে পারিনি। অনুগ্রহ করে কিছুক্ষণ পর আবার চেষ্টা করুন।"
)
