import pandas as pd
import re
import nltk
from nltk.corpus import words
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate

# Download English words if not already present
nltk.download('words', quiet=True)
ENGLISH_DICT = set(w.lower() for w in words.words())

# For a production system, we would load a massive Hindi corpus here.
# For this assignment, we use a basic mock set to demonstrate the approach.
HINDI_DICT = {'है', 'और', 'मैं', 'तुम', 'हम', 'क्या', 'नहीं', 'बहुत', 'अच्छा', 'मुझे', 'मेरा', 'कि', 'तो', 'ये', 'वो'}

def is_invalid_devanagari(word):
    """
    Returns True if the word breaks fundamental Devanagari writing rules.
    """
    word = str(word)
    
    # Rule 1: Cannot start with a matra (dependent vowel)
    if re.match(r'^[\u093E-\u094C\u094E-\u094F]', word):
        return True, "Starts with a matra"
        
    # Rule 2: Cannot have three identical consonants in a row (e.g., ककक)
    if re.search(r'(.)\1\1', word):
        return True, "Three identical characters consecutively"
        
    # Rule 3: Contains non-Devanagari characters (like English letters mixed with Hindi)
    # Allows Devanagari letters, matras, and zero-width joiners
    if not re.match(r'^[\u0900-\u097F\u200C\u200D]+$', word):
        return True, "Contains mixed/invalid scripts"
        
    return False, ""

def check_hinglish(word):
    """
    Transliterates Hindi to Latin and checks against English dictionary.
    """
    latin_word = transliterate(word, sanscript.DEVANAGARI, sanscript.ITRANS).lower()
    
    # Strip trailing 'a' which is common in ITRANS 
    if latin_word.endswith('a') and len(latin_word) > 2:
        latin_word_short = latin_word[:-1]
    else:
        latin_word_short = latin_word

    if latin_word in ENGLISH_DICT or latin_word_short in ENGLISH_DICT:
        return True
    return False

def classify_word(word):
    word = str(word).strip()
    
    # 1. Rule-Based Filter for obvious typos
    is_invalid, reason = is_invalid_devanagari(word)
    if is_invalid:
        return 'Incorrect', 'High', f'Grammar Error: {reason}'
        
    # 2. Hindi Dictionary Check
    if word in HINDI_DICT:
        return 'Correct', 'High', 'Found in Hindi Dictionary'
        
    # 3. Hinglish Transliteration Check
    if check_hinglish(word):
        return 'Correct', 'Medium', 'Matches English transliteration (Hinglish)'
        
    # 4. Unknown Words (Low Confidence)
    if len(word) > 15:
        return 'Incorrect', 'Medium', 'Word is unnaturally long'
    
    return 'Incorrect', 'Low', 'Not found in dictionaries, but phonetically valid'

# YAHAN SE MAIN EXECUTION SHURU HOTA HAI (Jo pichli baar miss ho gaya tha)
# YAHAN SE MAIN EXECUTION SHURU HOTA HAI
if __name__ == "__main__":
    print("Loading 1.77 Lakh words dataset...")
    
    FILE_PATH = "data/raw/Unique Words Data.xlsx - Sheet1.csv"
    
    try:
        # THE FIX: File is actually an Excel file under the hood, so we force read_excel
        df = pd.read_excel(FILE_PATH, engine='openpyxl')
    except Exception as e:
        print(f"Excel load failed, trying CSV fallback... Error: {e}")
        df = pd.read_csv(FILE_PATH, engine='python', on_bad_lines='skip', quoting=3)
        
    # Get the first column (which contains the words)
    word_column = df.columns[0]
    print(f"Processing column: '{word_column}' with {len(df)} words...")
    
    # Apply classification 
    results = df[word_column].apply(lambda x: pd.Series(classify_word(x)))
    df[['Status', 'Confidence', 'Reason']] = results
    
    # Save the output
    OUTPUT_PATH = "data/processed/classified_words_output.csv"
    df.to_csv(OUTPUT_PATH, index=False, encoding='utf-8-sig')
    
    # Print Summary
    print("\n" + "="*50)
    print("CLASSIFICATION SUMMARY")
    print("="*50)
    print(df['Status'].value_counts())
    print("\nCONFIDENCE BUCKETS")
    print(df['Confidence'].value_counts())
    
    print(f"\nSaved final classified list to: {OUTPUT_PATH}")
    
    # Grab some low confidence words for Q3(c)
    print("\nHere are 10 'Low Confidence' words for your manual review:")
    low_conf = df[df['Confidence'] == 'Low'].head(10)
    print(low_conf[[word_column, 'Status', 'Reason']])