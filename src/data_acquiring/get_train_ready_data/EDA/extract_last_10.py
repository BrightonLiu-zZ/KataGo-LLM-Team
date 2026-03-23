import json

input_path = "data/training_data_augmented_shuffled_v3.jsonl"
output_path = "last_10_rows_dataset.json"

with open(input_path) as f:
    lines = f.readlines()

last_10 = [json.loads(line) for line in lines[-10:]]

with open(output_path, "w") as f:
    json.dump(last_10, f, indent=2)

print(f"Wrote {len(last_10)} rows to {output_path}")
