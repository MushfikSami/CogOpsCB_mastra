/**
 * time-reminder.ts — port of cogops/prompts/time_reminder.py
 *
 * Always emits Bangladesh-local time (Asia/Dhaka, UTC+6, no DST) regardless of
 * the host timezone. The server clock is treated as UTC.
 */

const BN_DIGITS = ["০", "১", "২", "৩", "৪", "৫", "৬", "৭", "৮", "৯"];

const BN_WEEKDAYS: Record<string, string> = {
  Monday: "সোমবার",
  Tuesday: "মঙ্গলবার",
  Wednesday: "বুধবার",
  Thursday: "বৃহস্পতিবার",
  Friday: "শুক্রবার",
  Saturday: "শনিবার",
  Sunday: "রবিবার",
};

const BN_MONTHS: Record<string, string> = {
  January: "জানুয়ারি",
  February: "ফেব্রুয়ারি",
  March: "মার্চ",
  April: "এপ্রিল",
  May: "মে",
  June: "জুন",
  July: "জুলাই",
  August: "আগস্ট",
  September: "সেপ্টেম্বর",
  October: "অক্টোবর",
  November: "নভেম্বর",
  December: "ডিসেম্বর",
};

function toBengaliDigits(s: string): string {
  return s.replace(/[0-9]/g, (d) => BN_DIGITS[Number(d)]);
}

export function buildTimeReminder(now: Date = new Date()): string {
  // Convert to Asia/Dhaka (UTC+6) from UTC epoch.
  const dhaka = new Date(now.getTime() + 6 * 60 * 60 * 1000);
  const day = String(dhaka.getUTCDate());
  const monthEn = dhaka.toLocaleString("en-US", { month: "long", timeZone: "UTC" });
  const monthBn = BN_MONTHS[monthEn] ?? monthEn;
  const year = String(dhaka.getUTCFullYear());
  const weekdayEn = dhaka.toLocaleString("en-US", { weekday: "long", timeZone: "UTC" });
  const weekdayBn = BN_WEEKDAYS[weekdayEn] ?? weekdayEn;
  const hh = String(dhaka.getUTCHours()).padStart(2, "0");
  const mm = String(dhaka.getUTCMinutes()).padStart(2, "0");
  const hhmm = `${hh}:${mm}`;

  const bnDay = toBengaliDigits(day);
  const bnYear = toBengaliDigits(year);
  const bnHhmm = toBengaliDigits(hhmm);

  return (
    "[Time reminder — Bangladesh Standard Time (Asia/Dhaka, UTC+6, no DST). " +
    "The server clock is UTC; THIS is the canonical 'now' for Bangladesh. " +
    "Use it only if the user asks about deadlines, today, weekday, or office hours.]\n" +
    `- Date:     ${day} ${monthEn} ${year}   (${bnDay} ${monthBn} ${bnYear})\n` +
    `- Weekday:  ${weekdayEn}              (${weekdayBn})\n` +
    `- Time:     ${hhmm} BST                (${bnHhmm} বাংলাদেশ সময়)\n` +
    "- Government office hours (usually): Sunday–Thursday 09:00–17:00; " +
    "closed Friday & Saturday (weekly holiday)."
  );
}
