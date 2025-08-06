# c_feature_annotator.py
import pandas as pd
import re
import os
import time
import json
import requests
import zipfile
import io

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import torch
from transformers import pipeline
import google.generativeai as genai
from google.api_core import exceptions as google_exceptions

# --- Lexicon definitions ---
LEXICON_FILE_PATH = 'NRC-Emotion-Lexicon-Wordlevel-v0.92.txt'
PRONOUNS_I = {'i', 'me', 'my', 'mine', 'myself'}
PRONOUNS_WE = {'we', 'us', 'our', 'ours', 'ourselves'}
ABSOLUTIST_WORDS = {
    'always', 'all', 'completely', 'constant', 'constantly', 'definitely', 
    'entire', 'ever', 'every', 'everyone', 'everything', 'full', 'fully',
    'must', 'never', 'nothing', 'totally', 'whole'
}

# --- Lexicon management ---
def download_and_prepare_nrc_lexicon():.
    if os.path.exists(LEXICON_FILE_PATH): print(f"Found existing NRC Lexicon file: '{LEXICON_FILE_PATH}'")
    else:
        print("NRC Lexicon not found. Attempting to download...")
        url = "http://sentiment.nrc.ca/lexicons-for-research/NRC-Emotion-Lexicon-v0.92-In105Languages-Nov2017Translations.zip"
        try:
            response = requests.get(url, stream=True); response.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                english_lexicon_filename = 'NRC-Emotion-Lexicon-v0.92/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt'
                z.extract(english_lexicon_filename); os.rename(english_lexicon_filename, LEXICON_FILE_PATH); os.rmdir('NRC-Emotion-Lexicon-v0.92')
            print("NRC Lexicon downloaded and prepared successfully.")
        except Exception as e:
            print(f"ERROR: Could not download/process lexicon. Error: {e}"); return None
    lexicon_df = pd.read_csv(LEXICON_FILE_PATH, names=["word", "emotion", "association"], sep='\t')
    nrc_lexicon = lexicon_df.pivot(index='word', columns='emotion', values='association').fillna(0)
    print(f"NRC Lexicon loaded with {len(nrc_lexicon)} words.")
    return nrc_lexicon

# --- Annotations functions ---

def annotate_linguistic_features(df, nrc_lexicon):
    print("\n--- Annotating with linguistic & rule-based features ---")
    if df is None or df.empty: return df

    all_features = []
    vader_analyzer = SentimentIntensityAnalyzer()
    
    for text in df['content']:
        words = re.findall(r'\b\w+\b', str(text).lower())
        total_words = len(words) if len(words) > 0 else 1
        features = {}
        
        # VADER sentiment
        features['vader_compound'] = vader_analyzer.polarity_scores(str(text))['compound']
        
        # NRC lexicon
        if nrc_lexicon is not None:
            word_emotions = nrc_lexicon.reindex(words).sum()
            nrc_scores = ((word_emotions / total_words) * 100).to_dict()
            for key, val in nrc_scores.items(): features[f'nrc_{key}'] = val
        
        # Validated markers
        features['i_talk_freq'] = (sum(1 for word in words if word in PRONOUNS_I) / total_words) * 100
        features['we_talk_freq'] = (sum(1 for word in words if word in PRONOUNS_WE) / total_words) * 100
        features['absolutist_freq'] = (sum(1 for word in words if word in ABSOLUTIST_WORDS) / total_words) * 100
        
        all_features.append(features)

    features_df = pd.DataFrame(all_features)
    return pd.concat([df.reset_index(drop=True), features_df.reset_index(drop=True)], axis=1)

def annotate_deep_learning_emotions(df):
    print("\n--- Annotating with deep learning emotion classifier ---")
    if df is None or df.empty: return df
    
    model_name = "SamLowe/roberta-base-go_emotions"
    device = 0 if torch.cuda.is_available() else -1
    try:
        emotion_classifier = pipeline("text-classification", model=model_name, top_k=None, device=device)
    except Exception as e:
        print(f"ERROR: Could not load model '{model_name}'. {e}"); return df

    results = []; batch_size = 32
    for i in range(0, len(df), batch_size):
        batch = [str(item) for item in df['content'][i:i+batch_size].tolist() if item]
        if not batch: continue
        results.extend(emotion_classifier(batch))
        print(f"  Processed DL batch {i//batch_size + 1} of {len(df)//batch_size + 1}")
    
    emotion_data = [{d['label']: d['score'] for d in res_list} for res_list in results]
    emotion_df = pd.DataFrame(emotion_data).add_prefix('goemo_')
    return pd.concat([df.reset_index(drop=True), emotion_df.reset_index(drop=True)], axis=1)

def annotate_llm_perma(df, batch_size=100, max_retries=3):
    print("\n--- Annotating with LLM for PERMA Classification ---")
    if df is None or df.empty: return df

    try:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key: raise ValueError("GOOGLE_API_KEY not found in environment or .env file.")
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.5-flash')
    except Exception as e:
        print(f"ERROR: Could not configure Gemini. Skipping. Error: {e}"); return df
    
    if 'perma_labels' not in df.columns: df['perma_labels'] = "Not Processed"
    df_to_process = df[df['perma_labels'] == "Not Processed"]
    if df_to_process.empty:
        print("All messages have already been processed for PERMA labels. Skipping."); return df
    
    print(f"{len(df_to_process)} messages remaining to be processed by the LLM.")
    for i in range(0, len(df_to_process), batch_size):
        batch_df = df_to_process.iloc[i:i+batch_size]
        prompt_messages = [f'{index}: "{str(row["content"]).replace("\"", "'").replace(os.linesep, " ")}"' for index, row in batch_df.iterrows()]
        prompt = f"""You are a psychology research assistant. Analyze the following batch of text messages based on the PERMA model (Positive Emotion, Engagement, Relationships, Meaning, Accomplishment). Return a single JSON object where each key is the message index (as a string) and the value is a comma-separated string of the PERMA pillars you identify. If no pillars are present for a message, the value should be "None". Example Response Format: {{"101": "Relationships, Positive Emotion", "102": "Accomplishment", "103": "None"}}. Here is the batch of messages to analyze:\n--- BATCH START ---\n{os.linesep.join(prompt_messages)}\n--- BATCH END ---\n\nJSON Response:"""
        
        print(f"  Processing LLM batch {i//batch_size + 1} of {len(df_to_process)//batch_size + 1}...")
        for attempt in range(max_retries):
            try:
                response = model.generate_content(prompt)
                json_text = response.text.strip().lstrip('```json').rstrip('```')
                perma_results = json.loads(json_text)
                for msg_index_str, labels in perma_results.items():
                    df.loc[int(msg_index_str), 'perma_labels'] = labels
                print(f"    -> Successfully processed {len(perma_results)} messages.")
                break
            except google_exceptions.ResourceExhausted:
                wait_time = 60; print(f"    -> RATE LIMIT HIT (Attempt {attempt+1}/{max_retries}). Waiting {wait_time}s."); time.sleep(wait_time)
            except Exception as e:
                print(f"    -> ERROR processing batch. Skipping. Error: {e}"); break
        time.sleep(10) # Inter-batch delay
        
    return df