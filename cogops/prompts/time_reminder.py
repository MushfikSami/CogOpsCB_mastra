"""
cogops/prompts/time_reminder.py

Builds the per-turn "time/locale reminder" system message injected just before
the user message in the composer's input.

The server clock is UTC (or whatever the host happens to be); Bangladesh is
UTC+6 (Asia/Dhaka, no DST). Always compute Bangladesh-local time from a
fixed offset — never trust the server's local time.

Usage:
    msg = build_time_reminder()
    messages = [
        {"role": "system", "content": composer_system_prompt},
        *history,
        {"role": "system", "content": msg},
        {"role": "user", "content": user_payload},
    ]
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone


# Asia/Dhaka is a fixed UTC+6 offset (Bangladesh does not observe DST as of
# 2024+). Using a manual offset is safe and avoids depending on the host
# system's tzdata being current.
BST = timezone(timedelta(hours=6), name="Asia/Dhaka")


_BENGALI_DIGITS = str.maketrans("0123456789", "০১২৩৪৫৬৭৮৯")

_BN_WEEKDAYS = {
    "Monday":    "সোমবার",
    "Tuesday":   "মঙ্গলবার",
    "Wednesday": "বুধবার",
    "Thursday":  "বৃহস্পতিবার",
    "Friday":    "শুক্রবার",
    "Saturday":  "শনিবার",
    "Sunday":    "রবিবার",
}

_BN_MONTHS = {
    "January":   "জানুয়ারি",
    "February":  "ফেব্রুয়ারি",
    "March":     "মার্চ",
    "April":     "এপ্রিল",
    "May":       "মে",
    "June":      "জুন",
    "July":      "জুলাই",
    "August":    "আগস্ট",
    "September": "সেপ্টেম্বর",
    "October":   "অক্টোবর",
    "November":  "নভেম্বর",
    "December":  "ডিসেম্বর",
}


def _to_bengali_digits(s: str) -> str:
    return s.translate(_BENGALI_DIGITS)


def build_time_reminder(now: datetime | None = None) -> str:
    """Render the time-reminder system message.

    Always emits Bangladesh-local time (Asia/Dhaka, UTC+6) regardless of the
    host's timezone. Includes both English and Bengali date / weekday so the
    composer has whichever form it needs for the user.

    Args:
        now: optional datetime; defaults to current Bangladesh time. If
            given without tzinfo, it is *interpreted as Bangladesh time*.

    Returns:
        A single string suitable as a `{"role": "system", "content": ...}` body.
    """
    if now is None:
        now = datetime.now(BST)
    elif now.tzinfo is None:
        now = now.replace(tzinfo=BST)
    else:
        now = now.astimezone(BST)

    day = now.strftime("%d").lstrip("0") or "0"
    month_en = now.strftime("%B")
    month_bn = _BN_MONTHS.get(month_en, month_en)
    year = now.strftime("%Y")
    weekday_en = now.strftime("%A")
    weekday_bn = _BN_WEEKDAYS.get(weekday_en, weekday_en)
    hhmm = now.strftime("%H:%M")

    bn_day = _to_bengali_digits(day)
    bn_year = _to_bengali_digits(year)
    bn_hhmm = _to_bengali_digits(hhmm)

    return (
        "[Time reminder — Bangladesh Standard Time (Asia/Dhaka, UTC+6, no DST). "
        "The server clock is UTC; THIS is the canonical 'now' for Bangladesh. "
        "Use it only if the user asks about deadlines, today, weekday, or "
        "office hours.]\n"
        f"- Date:     {day} {month_en} {year}   ({bn_day} {month_bn} {bn_year})\n"
        f"- Weekday:  {weekday_en}              ({weekday_bn})\n"
        f"- Time:     {hhmm} BST                ({bn_hhmm} বাংলাদেশ সময়)\n"
        "- Government office hours (usually): Sunday–Thursday 09:00–17:00; "
        "closed Friday & Saturday (weekly holiday)."
    )
