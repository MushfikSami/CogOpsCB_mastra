/**
 * responses.ts — static bilingual response templates.
 *
 * Copied verbatim from configs/config.yml `responses:` so refusal/guard-rail
 * text is byte-identical to the Python service.
 */

export const RESPONSES = {
  refusal_text_bn: "দুঃখিত, এই প্রশ্নের জন্য নির্ভরযোগ্য সরকারি তথ্য পাওয়া যায়নি।",
  political_refusal_text_bn:
    "আমি একটি নিরপেক্ষ সরকারি সেবা সহকারী। রাজনৈতিক তুলনা বা মতামত দেওয়া আমার পক্ষে সম্ভব নয়।",
  personal_law_refusal_text_bn:
    "এই প্রশ্নটি ব্যক্তিগত আইন বা ধর্মীয় বিধানের পরামর্শের বিষয়। সঠিক উত্তরের জন্য সংশ্লিষ্ট বিশেষজ্ঞের সাথে পরামর্শ করুন।",
  self_harm_refusal_text_bn:
    "আমি আপনাকে এই মুহূর্তে সাহায্য করতে পারব না। অনুগ্রহ করে ৯৯৯ বা নিকটতম হাসপাতালে যোগাযোগ করুন।",
  illegal_refusal_text_bn: "এই ধরনের কার্যকলাপ সম্পর্কে তথ্য দেওয়া আমার পক্ষে সম্ভব নয়।",
  input_invalid_refusal_bn:
    "দুঃখিত, প্রশ্নটি বোঝা গেল না বা সীমার বাইরে। অনুগ্রহ করে স্পষ্ট, সংক্ষিপ্ত প্রশ্ন করুন।",
  chitchat_greeting_bn:
    "স্বাগতম! আমি বাংলাদেশ সরকারের সেবা সম্পর্কিত প্রশ্নে সাহায্য করতে পারি — যেমন এনআইডি, পাসপোর্ট, ট্যাক্স, সনদ, ইত্যাদি। আপনি কোন বিষয়ে জানতে চান?",
  system_probe_response_bn:
    "আমি বাংলাদেশ সরকারের ডিজিটাল সহকারী 'আশা'। আমার কাজ নাগরিকদের সরকারি সেবা সংক্রান্ত তথ্য সহজভাবে বাংলায় পৌঁছে দেওয়া।",
};

/** Map a guard-rail category → its refusal template (mirrors _guard_rail_response). */
export function guardRailResponse(category: string | null | undefined): string {
  switch (category) {
    case "self_harm":
      return RESPONSES.self_harm_refusal_text_bn;
    case "illegal":
      return RESPONSES.illegal_refusal_text_bn;
    case "political_comparison":
      return RESPONSES.political_refusal_text_bn;
    case "personal_attack":
      return RESPONSES.personal_law_refusal_text_bn;
    case "system_probe":
      return RESPONSES.system_probe_response_bn;
    default:
      return RESPONSES.refusal_text_bn;
  }
}
