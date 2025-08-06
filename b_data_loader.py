# b_data_loader.py
import pandas as pd
import os
import re
import sys

def load_and_parse_data(input_file):
    """
    Reads a text archive line-by-line and groups multiline messages.
    
    Args:
        input_file (str): The path to the input text file.
    
    Returns:
        pd.DataFrame: A DataFrame with columns ['timestamp', 'username', 'content'].
    """
    print(f"\nLoading and parsing data from '{input_file}'...")
    if not os.path.exists(input_file):
        print(f"ERROR: Input file not found at '{input_file}'")
        sys.exit(1)
        
    # regex to detect the start of a new message line
    message_start_pattern = re.compile(r'^\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\] (.*?): (.*)')
    
    parsed_messages = []
    current_message = None

    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            match = message_start_pattern.match(line)
            
            if match:
                # this line is the start of a new message.
                # first, save the previous message if it exists.
                if current_message:
                    parsed_messages.append(current_message)
                
                # Now, start the new message object.
                timestamp_str, username, content = match.groups()
                current_message = {
                    'timestamp': pd.to_datetime(timestamp_str),
                    'username': username,
                    'content': content.strip()
                }
            elif current_message:
                # this line is a continuation of the previous message.
                current_message['content'] += '\n' + line.strip()

        # after the loop, save the very last message in the file.
        if current_message:
            parsed_messages.append(current_message)
    
    df = pd.DataFrame(parsed_messages)
    # final cleanup of content
    if not df.empty:
        df['content'] = df['content'].str.strip()

    print(f"Successfully loaded and parsed {len(df)} messages.")
    return df