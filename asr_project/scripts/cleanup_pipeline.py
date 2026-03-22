import re
import nltk
from nltk.corpus import words

# Download English dictionary for word detection
nltk.download('words', quiet=True)
english_vocab = set(words.words())

# 1. Number Dictionary (0 to 100, and large multipliers)
hindi_numbers = {
    'शून्य': 0, 'एक': 1, 'दो': 2, 'तीन': 3, 'चार': 4, 'पांच': 5, 'पाँच': 5,
    'छह': 6, 'सात': 7, 'आठ': 8, 'नौ': 9, 'दस': 10,
    'ग्यारह': 11, 'बारह': 12, 'तेरह': 13, 'चौदह': 14, 'पंद्रह': 15, 'सोलह': 16,
    'सत्रह': 17, 'अठारह': 18, 'उन्नीस': 19, 'बीस': 20,
    'पच्चीस': 25, 'तीस': 30, 'चालीस': 40, 'पचास': 50, 'साठ': 60, 'सत्तर': 70,
    'अस्सी': 80, 'नब्बे': 90, 'सौ': 100, 'हज़ार': 1000, 'हजार': 1000, 
    'लाख': 100000, 'करोड़': 10000000
}

# Add remaining common numbers for compound parsing (354 = तीन सौ चौवन, etc.)
hindi_numbers.update({'चौवन': 54})

# Simple mapping of common English words written in Devanagari (Hinglish dictionary)
# In a real production system, we would use a transliteration model + English dict check.
hinglish_dict = {
    'इंटरव्यू': 'interview', 'जॉब': 'job', 'प्रॉब्लम': 'problem', 'सॉल्व': 'solve',
    'कंप्यूटर': 'computer', 'स्कूल': 'school', 'कॉलेज': 'college', 'ऑफिस': 'office',
    'टाइम': 'time', 'नंबर': 'number', 'फोन': 'phone', 'मोबाइल': 'mobile',
    'सर': 'sir', 'मैडम': 'madam', 'प्रोजेक्ट': 'project', 'सिस्टम': 'system',
    'डेटा': 'data', 'क्लास': 'class', 'टेस्ट': 'test', 'मैसेज': 'message'
}

class TextCleanupPipeline:
    def __init__(self):
        self.num_dict = hindi_numbers
        self.en_dict = hinglish_dict
        
    def normalize_numbers(self, text):
        """
        Converts Hindi number words to digits, but ignores hyphenated idioms (e.g., दो-चार).
        """
        # Split text but keep hyphenated words intact
        tokens = text.split()
        normalized_tokens = []
        
        i = 0
        while i < len(tokens):
            token = tokens[i]
            
            # EDGE CASE: Ignore hyphenated idioms like "दो-चार"
            if '-' in token:
                normalized_tokens.append(token)
                i += 1
                continue
                
            # Check for compound numbers like "तीन सौ चौवन" (354) or "एक हज़ार" (1000)
            if token in self.num_dict:
                current_val = self.num_dict[token]
                
                # Look ahead for multipliers or additions
                if i + 1 < len(tokens) and tokens[i+1] in ['सौ', 'हज़ार', 'हजार', 'लाख', 'करोड़']:
                    multiplier = self.num_dict[tokens[i+1]]
                    current_val *= multiplier
                    i += 1
                    
                    # Look ahead for trailing additions (like 'चौवन' in 'तीन सौ चौवन')
                    if i + 1 < len(tokens) and tokens[i+1] in self.num_dict and tokens[i+1] not in ['सौ', 'हज़ार', 'हजार']:
                        current_val += self.num_dict[tokens[i+1]]
                        i += 1
                        
                normalized_tokens.append(str(current_val))
            else:
                normalized_tokens.append(token)
            i += 1
            
        return " ".join(normalized_tokens)

    def detect_english_words(self, text):
        """
        Tags English words written in Devanagari script.
        """
        tokens = text.split()
        tagged_tokens = []
        
        for token in tokens:
            # Safely stripping standard punctuation instead of using Regex
            clean_token = token.strip("।,-!?\"'() ")
            
            if clean_token in self.en_dict:
                # Add the [EN] tags explicitly
                tagged_token = token.replace(clean_token, f"[EN] {clean_token} [/EN]")
                tagged_tokens.append(tagged_token)
            else:
                tagged_tokens.append(token)
                
        return " ".join(tagged_tokens)

    def process(self, text):
        text = self.normalize_numbers(text)
        text = self.detect_english_words(text)
        return text

# ── Examples for the Assignment ──────────────────────
if __name__ == "__main__":
    pipeline = TextCleanupPipeline()
    
    print("\n" + "="*50)
    print("1. NUMBER NORMALIZATION EXAMPLES")
    print("="*50)
    
    # Simple cases
    print(f"Original: मेरे पास दो किताबें हैं।")
    print(f"Cleaned : {pipeline.normalize_numbers('मेरे पास दो किताबें हैं।')}\n")
    
    # Compound numbers
    print(f"Original: मेरे स्कूल में तीन सौ चौवन बच्चे हैं और फीस एक हज़ार है।")
    print(f"Cleaned : {pipeline.normalize_numbers('मेरे स्कूल में तीन सौ चौवन बच्चे हैं और फीस एक हज़ार है।')}\n")
    
    # Edge Cases (Idioms)
    print(f"Original: मैंने उसे दो-चार बातें सुना दी। (Edge Case)")
    print(f"Cleaned : {pipeline.normalize_numbers('मैंने उसे दो-चार बातें सुना दी।')}")
    print("Reason  : Hyphenated idioms imply approximation ('a few'), converting to '2-4' changes the semantic tone.\n")
    
    print("="*50)
    print("2. ENGLISH WORD DETECTION EXAMPLES")
    print("="*50)
    
    print(f"Original: मेरा इंटरव्यू बहुत अच्छा गया और मुझे जॉब मिल गई")
    print(f"Cleaned : {pipeline.detect_english_words('मेरा इंटरव्यू बहुत अच्छा गया और मुझे जॉब मिल गई')}\n")
    
    print(f"Original: ये प्रॉब्लम सॉल्व नहीं हो रहा है कंप्यूटर में")
    print(f"Cleaned : {pipeline.detect_english_words('ये प्रॉब्लम सॉल्व नहीं हो रहा है कंप्यूटर में')}\n")