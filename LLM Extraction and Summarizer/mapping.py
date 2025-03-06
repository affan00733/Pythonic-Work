import os
import json
import re

root_dir = "data/CrisisFACTS/rawdata"

sub_dirs = ["facebook", "news", "reddit", "twitter"]

image_pattern = re.compile(r"(\d+)_\d+\.jpg")


r_tag_pattern = re.compile(r"-r(\d+)")


output_data = []


for sub_dir in sub_dirs:
    sub_dir_path = os.path.join(root_dir, sub_dir)

    if not os.path.exists(sub_dir_path):
        print(f"⚠️ Skipping missing directory: {sub_dir_path}")
        continue


    trecis_folders = [os.path.join(sub_dir_path, folder) for folder in os.listdir(sub_dir_path) if folder.startswith("TRECIS")]

    if not trecis_folders:
        print(f"⚠️ No TRECIS folder found in {sub_dir_path}, skipping...")
        continue


    crisisfacts_folders = [os.path.join(sub_dir_path, folder) for folder in os.listdir(sub_dir_path) if folder.startswith("CrisisFACTS")]

    if not crisisfacts_folders:
        print(f"⚠️ No CrisisFACTS folder found in {sub_dir_path}, skipping...")
        continue


    for trecis_folder in trecis_folders:
        for trecis_subfolder in os.listdir(trecis_folder):
            trecis_subfolder_path = os.path.join(trecis_folder, trecis_subfolder)
            if not os.path.isdir(trecis_subfolder_path):
                continue

            print(f"📂 Processing TRECIS folder: {trecis_subfolder_path}")


            for crisisfacts_folder in crisisfacts_folders:
                for crisis_subfolder in os.listdir(crisisfacts_folder):
                    crisis_subfolder_path = os.path.join(crisisfacts_folder, crisis_subfolder)

                    if not os.path.isdir(crisis_subfolder_path):
                        continue


                    r_tag_match = r_tag_pattern.search(crisis_subfolder)
                    r_tag = r_tag_match.group(1) if r_tag_match else "0"  # Default to "0" if no match


                    metadata_file_paths = [os.path.join(crisis_subfolder_path, file) for file in os.listdir(crisis_subfolder_path) if file.endswith(".json")]

                    if not metadata_file_paths:
                        print(f"⚠️ No JSON metadata found in {crisis_subfolder_path}, skipping...")
                        continue


                    for metadata_file_path in metadata_file_paths:
                        print(f"📝 Loading JSON metadata from: {metadata_file_path}")


                        try:
                            with open(metadata_file_path, "r", encoding="utf-8") as f:
                                metadata_list = json.load(f)

                            if not metadata_list:
                                print(f"⚠️ Skipping empty JSON file: {metadata_file_path}")
                                continue

                        except json.JSONDecodeError:
                            print(f"❌ JSON decoding failed for {metadata_file_path}, skipping...")
                            continue

                        print(f"✅ Loaded {len(metadata_list)} metadata entries.")


                        metadata_mapping = {}
                        for entry in metadata_list:
                            if "source" in entry:
                                try:
                                    if type(entry["source"]) == dict:
                                        mapping_2 = entry["source"]
                                    else:
                                        mapping_2 = json.loads(entry["source"])

                                    metadata_mapping[str(mapping_2["id"])] = entry  # Ensure ID is string
                                except:
                                    print(f"⚠️ Skipping entry with invalid 'source' JSON: {entry}")

                        print(f"✅ Extracted {len(metadata_mapping)} unique IDs from JSON.")

                        for file in os.listdir(trecis_subfolder_path):
                            if file.endswith(".jpg"):
                                match = image_pattern.match(file)
                                if match:
                                    unique_id = match.group(1)

                                    if unique_id in metadata_mapping:
                                        entry = metadata_mapping[unique_id]
                                        output_data.append({
                                            "image_filename": file,
                                            "unique_id": unique_id,
                                            "doc_id": entry["doc_id"],
                                            "event_id": entry["event"],
                                            "unix_timestamp": entry["unix_timestamp"],
                                            "source_type": entry.get("source_type", "Unknown"),
                                            "r_tag": r_tag,
                                            "text": entry["text"]
                                        })
                                        print(f"✅ Matched {file} → {entry['doc_id']}")

output_json_path = os.path.join(root_dir, "image_metadata_mapping.json")
with open(output_json_path, "w", encoding="utf-8") as json_file:
    json.dump(output_data, json_file, indent=4)

print(f"\n✅ Mapping completed! JSON saved at: {output_json_path}")
