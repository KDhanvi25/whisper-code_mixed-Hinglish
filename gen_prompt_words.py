# Paste this into a file named generate_prompt_words.py

# --- Step 1: Paste your top Hindi words list here ---
top_words_raw = """
है के पर में करें को हैं और लिए से click का अब इस हम की एक यह क्लिक सकते आप कि tutorial चुनें text करने कर करते भी प्रदर्शित slide spoken करके फिर box file उपयोग document हो मैं कैसे इसे होता साथ टैब बॉक्स यहाँ जो project button पहले menu रूप type option नहीं सेव करना देख द्वारा करता गया दें अधिक या ध्यान जानकारी नीचे ही थंडरबर्ड options print हूँ पैनल अपने हमने फाइल ओर name window तो dialog font प्रस्तुति आपके बारे देखते slides किया उपलब्ध view बटन पास शब्द ok डायलॉग चलिए जैसे यदि मेल
"""

# --- Step 2: UI / non-content words to remove ---
junk_words = {
    "click", "क्लिक", "tutorial", "slide", "spoken", "box", "file", "document",
    "project", "button", "menu", "option", "options", "print", "dialog", "font",
    "slides", "window", "view", "type", "save", "बटन", "डायलॉग", "फाइल",
    "text", "choose", "select", "panel", "प्रस्तुति", "ok", "थंडरबर्ड"
}

# --- Step 3: Domain-relevant Hinglish connector words ---
hinglish_words = [
    "school", "exam", "homework", "papa", "mummy", "cricket", "whatsapp", "delhi",
    "office", "phone", "movie", "doctor", "hospital", "train", "ticket", "class",
    "sir", "teacher", "india", "english", "hindi", "kal", "ghar", "please", "okay",
    "thanks", "shopping", "market", "mobile", "email", "website"
]

# --- Step 4: Clean and process ---
top_words = top_words_raw.strip().split()
cleaned_words = [w for w in top_words if w.lower() not in junk_words]

# --- Step 5: Add Hinglish words ---
final_prompt_words = list(dict.fromkeys(cleaned_words + hinglish_words))  # preserve order, remove duplicates

# --- Step 6: Save to file ---
with open("prompt_words.txt", "w", encoding="utf-8") as f:
    for word in final_prompt_words:
        f.write(word + " ")

print(f"✅ Saved {len(final_prompt_words)} prompt words to prompt_words.txt")
