import json

with open('/data/data2/shiman/R1-Omni/data_json/general.json', 'r') as f:
    data = json.load(f)

base_dir = '/data/data2/shiman/dataset/col_2/'

def extract_caption(general_caption):
    start = general_caption.find('<caption>') + len('<caption>')
    end = general_caption.find('</caption>')
    return general_caption[start:end].strip()

new_json_entries = []

# Process each entry
for entry in data:
    action = entry['action']
    video_id = entry['id']
    video_format = entry['format']  # Get the format dynamically
    general_caption = entry['general_caption']
    
    # Construct video path using id as-is and dynamic format
    video_path = f"{base_dir}{action}/{video_id}.{video_format}"
    
    # Extract caption text
    caption_text = extract_caption(general_caption)
    
    # Build the new entry
    new_entry = {
        "video": video_path,
        "conversations": [
            {
                "from": "human",
                "value": "<video>\nAs an action recognition expert; throughout the video, which action conveyed by the characters is the most obvious to you? Output the thinking process in <think> </think> and final action in <answer> </answer> tags."
            },
            {
                "from": "gpt",
                "value": f"<think>{caption_text}</think>\n<answer>{action}</answer>"
            }
        ]
    }
    
    new_json_entries.append(new_entry)

# Example output for the sample entry
print(json.dumps(new_json_entries[0], indent=4))

with open('/data/data2/shiman/R1-Omni/data_json/cold_start.json', 'w') as f:
    json.dump(new_json_entries, f, indent=4)