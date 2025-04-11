# import json

# # Load the JSON data
# with open('/data/data2/shiman/R1-Omni/data_json/general.json', 'r') as f:
#     data = json.load(f)

# # Base directory for video files
# base_dir = '/data/data2/shiman/dataset/col_2/'

# # Function to extract caption text from general_caption
# def extract_caption(general_caption):
#     start = general_caption.find('<caption>') + len('<caption>')
#     end = general_caption.find('</caption>')
#     return general_caption[start:end].strip()

# # List to hold new JSON entries
# new_json_entries = []

# # Process each entry
# for entry in data:
#     action = entry['action']
#     video_id = entry['id']
#     video_format = entry['format']  # Get the format dynamically
#     general_caption = entry['general_caption']
    
#     # Construct video path using id as-is and dynamic format
#     video_path = f"{base_dir}{action}/{video_id}.{video_format}"
    
#     # Extract caption text
#     caption_text = extract_caption(general_caption)
    
#     # Build the new entry
#     new_entry = {
#         "video": video_path,
#         "conversations": [
#             {
#                 "from": "human",
#                 "value": "<video>\nAs an action recognition expert; throughout the video, which action conveyed by the characters is the most obvious to you?"
#             },
#             {
#                 "from": "gpt",
#                 "value": f"{action}"
#             }
#         ]
#     }
    
#     new_json_entries.append(new_entry)

# # Example output for the sample entry
# print(json.dumps(new_json_entries[0], indent=4))

# # Optionally, save to a new file
# with open('/data/data2/shiman/R1-Omni/data_json/rl.json', 'w') as f:
#     json.dump(new_json_entries, f, indent=4)


import os
import json

base_dir = "/data/data2/shiman/dataset"
output_json = []

# Define a list of valid video extensions (add more as needed)
video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".wmv"}

# Iterate through main category folders (col_1, col_2, etc.)
for category_folder in os.listdir(base_dir):
    category_path = os.path.join(base_dir, category_folder)
    if os.path.isdir(category_path) and category_folder in ['col_1', 'col_2', 'col_3', 'col_4', 'NTU_medical_actions']:
        # Iterate through action subfolders (like shaking_hands)
        for action_folder in os.listdir(category_path):
            action_path = os.path.join(category_path, action_folder)
            if os.path.isdir(action_path):
                # Iterate through video files in each action folder
                for video_file in os.listdir(action_path):
                    # Check if the file has a valid video extension
                    if os.path.splitext(video_file)[1].lower() in video_extensions:
                        video_path = os.path.join(action_path, video_file)
                        entry = {
                            "video": video_path,
                            "conversations": [
                                {
                                    "from": "human",
                                    "value": "<video>\nAs an action recognition expert; throughout the video, which action conveyed by the characters is the most obvious to you?"
                                },
                                {
                                    "from": "gpt",
                                    "value": action_folder  # Use the action folder name as label
                                }
                            ]
                        }
                        output_json.append(entry)

# Save to a JSON file
with open("/data/data2/shiman/R1-Omni/data_json/openrl.json", "w") as f:
    json.dump(output_json, f, indent=4)

print(f"JSON file generated with {len(output_json)} entries")