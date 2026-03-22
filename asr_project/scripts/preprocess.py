import pandas as pd
import requests
import json
import os
import librosa
import soundfile as sf
from tqdm import tqdm

# ── Paths ──────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Updated to match the exact file name in your screenshot
EXCEL_PATH = os.path.join(BASE_DIR, "data", "raw", "FT Data.xlsx")
AUDIO_DIR = os.path.join(BASE_DIR, "audio")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
OUTPUT_JSON = os.path.join(PROCESSED_DIR, "dataset.json")

os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

# ── URL Builders (UPDATED) ─────────────────────────────
def fix_url(original_url):
    """
    Replaces the dead GCP bucket path with the new active 'upload_goai/' path.
    """
    if pd.isna(original_url):
        return ""
    return str(original_url).replace("joshtalks-data-collection/hq_data/hi/", "upload_goai/")

# ── Test URLs ──────────────────────────────────────────
def test_urls(df, n=3):
    print(f"\nTesting first {n} URLs...")
    for _, row in df.head(n).iterrows():
        rid = int(row['recording_id'])
        trans_url = fix_url(row['transcription_url_gcp'])
        
        try:
            r = requests.get(trans_url, timeout=10)
            print(f"  rec={rid} → status={r.status_code}")
            if r.status_code == 200:
                data = r.json()
                print(f"    segments: {len(data)}")
                print(f"    first text: {data[0]['text'][:80]}...")
            else:
                print(f"    FAILED: {r.text[:100]}")
        except Exception as e:
            print(f"  rec={rid} → FAILED to connect: {e}")

# ── Download + Process ─────────────────────────────────
def download_and_process(df, max_records=None, download_audio=False):
    records = []
    failed = []

    total = len(df) if max_records is None else max_records

    for idx, (_, row) in enumerate(tqdm(df.iterrows(), total=total, desc="Processing")):
        if max_records and idx >= max_records:
            break

        uid = int(row['user_id'])
        rid = int(row['recording_id'])
        duration = row['duration']

        # Skip very short clips
        if duration < 1:
            continue

        try:
            # Download transcription
            trans_url = fix_url(row['transcription_url_gcp'])
            trans_r = requests.get(trans_url, timeout=15)
            if trans_r.status_code != 200:
                failed.append({'recording_id': rid, 'reason': f'trans_status_{trans_r.status_code}'})
                continue

            segments = trans_r.json()

            # Merge all segment texts
            full_text = " ".join([
                s['text'].strip()
                for s in segments
                if 'text' in s and s['text'].strip()
            ])

            if not full_text.strip():
                failed.append({'recording_id': rid, 'reason': 'empty_transcription'})
                continue

            # Try to get metadata (optional)
            metadata = {}
            try:
                meta_url = fix_url(row['metadata_url_gcp'])
                meta_r = requests.get(meta_url, timeout=10)
                if meta_r.status_code == 200:
                    metadata = meta_r.json()
            except:
                pass

            audio_url = fix_url(row['rec_url_gcp'])

            record = {
                'recording_id': rid,
                'user_id': uid,
                'duration': duration,
                'transcription': full_text,
                'num_segments': len(segments),
                'segments': segments,
                'audio_url': audio_url,
                'audio_path': None,
                'metadata': metadata
            }

            # Download audio 
            if download_audio:
                audio_path = os.path.join(AUDIO_DIR, f"{rid}.wav")
                audio_r = requests.get(
                    audio_url,
                    timeout=60,
                    stream=True
                )
                if audio_r.status_code == 200:
                    with open(audio_path, 'wb') as f:
                        for chunk in audio_r.iter_content(chunk_size=8192):
                            f.write(chunk)
                    # Resample to 16kHz (Whisper requirement)
                    audio, _ = librosa.load(audio_path, sr=16000)
                    sf.write(audio_path, audio, 16000)
                    record['audio_path'] = audio_path
                else:
                    print(f"  Audio download failed for rec={rid}: {audio_r.status_code}")

            records.append(record)

        except Exception as e:
            failed.append({'recording_id': rid, 'reason': str(e)})
            continue

    return records, failed

# ── Summary Stats ──────────────────────────────────────
def print_summary(records, failed):
    print(f"\n{'='*50}")
    print(f"✓ Successfully processed : {len(records)}")
    print(f"✗ Failed                 : {len(failed)}")

    if failed:
        print(f"\nFailed reasons:")
        for f in failed[:5]:
            print(f"  {f}")

    if records:
        total_dur = sum(r['duration'] for r in records) / 3600
        avg_dur = sum(r['duration'] for r in records) / len(records)
        total_segs = sum(r['num_segments'] for r in records)
        print(f"\nDataset stats:")
        print(f"  Total duration   : {total_dur:.2f} hours")
        print(f"  Avg duration     : {avg_dur:.1f} seconds")
        print(f"  Total segments   : {total_segs}")
        print(f"\nSample transcription:")
        print(f"  {records[0]['transcription'][:150]}")
    print(f"{'='*50}")

# ── Main ───────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading CSV...")
    # THE FIX IS HERE: Added encoding='latin1' to handle special Windows characters
    df = pd.read_excel(EXCEL_PATH) 
    
    print(f"Total rows in Data: {len(df)}")
    print(df[['user_id', 'recording_id', 'duration']].head())

    # Step 1: Test URLs
    test_urls(df, n=3)

    # Step 2: Process transcriptions
    print("\nDownloading transcriptions...")
    records, failed = download_and_process(
        df,
        max_records=None,      # process all 104 records
        download_audio=False   # set True to download the actual .wav files
    )

    # Step 3: Print summary
    print_summary(records, failed)

    # Step 4: Save to JSON
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"\nSaved → {OUTPUT_JSON}")