# ASR Engineering Assignment - Josh Talks

**Candidate:** Harsh Srivastava  
**Role:** AI/ML Engineering Intern (Speech)

---

## 1. Data Preprocessing & Baseline Evaluation

### Approach

- Fixed broken GCS links and consolidated 22 hours of audio/transcripts into a unified `dataset.json`.
- Evaluated `whisper-small` on FLEURS Hindi dataset.
- **Initial WER:** 93.39%
- **Normalized WER:** 92.27% (After stripping punctuation).

### Error Taxonomy

1. **Phonetic Mismatches:** Model confuses similar sounds (e.g., 'अणुओं' vs 'अग्वो').
2. **Hallucinations:** Model generates gibberish when audio is unclear.
3. **Punctuation Bias:** Baseline WER was inflated due to model adding commas/periods not present in ground truth.

---

## 2. Text Cleanup Pipeline

Implemented a robust regex-based pipeline to:

- Normalize Hindi numbers to digits (e.g., 'तीन' -> 3).
- Identify and tag English words within Devanagari script using `[EN]` tags.
- Handle edge cases like common Hindi idioms to prevent over-normalization.

---

## 3. Large-Scale Word Classification (1.77 Lakh Words)

### Methodology

- Used a **Tri-Layer Filtering** approach:
  1. Dictionary Lookup (Standard Hindi).
  2. Transliteration Check (Hinglish/English words).
  3. Character-level N-gram probability (Confidence scoring).
- **Analysis of Low Confidence Bucket:** Words like 'हैं' or 'भी' were flagged because they frequently appear in conversational slangs or specific phonetic contexts not covered by rigid formal dictionaries.

---

## 4. Fine-Tuning Pipeline

- Successfully implemented a `Seq2SeqTrainer` pipeline for Whisper-Small.
- Handled hardware constraints by truncating label sequences to 448 tokens.
- **Result:** Successfully executed a 5-step training run to prove pipeline viability.

---

## 5. Proposed Advanced Evaluation (Lattice-based)

To handle variations like "14" (Digit) vs "चौदह" (Text):

- **Concept:** Instead of 1-Best WER, we should evaluate the ASR Lattice (Graph).
- **Logic:** If the reference word exists in any high-probability path of the lattice, the error is penalized less.
- **Pseudocode:** (Included in the technical submission folder).
