"""
Check whether the augmented/shuffled dataset preserves the
'Valid empty coordinates' list and '[Instruction]' block from the
pre-augmentation training-ready data.
"""
import json
import sys

PRE_AUG_FILE = "data/training_ready_data_v3.jsonl"
POST_AUG_FILE = "data/training_data_augmented_shuffled_v3.jsonl"
QUALITY_FILTERED_FILE = "data/training_data_quality_filtered_v3.jsonl"

NUM_SAMPLES = 5


def analyze_file(filepath, label, num_samples=NUM_SAMPLES):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  File: {filepath}")
    print(f"{'='*60}")

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            total = 0
            has_valid_coords = 0
            has_instruction_block = 0
            has_analyze_line = 0
            has_last_moves = 0

            for i, line in enumerate(f):
                if not line.strip():
                    continue
                rec = json.loads(line)
                prompt = rec.get("user_prompt", "")
                total += 1

                if "Valid empty coordinates" in prompt:
                    has_valid_coords += 1
                if "[Instruction]" in prompt:
                    has_instruction_block += 1
                if "Analyze the board" in prompt:
                    has_analyze_line += 1
                if "Last moves:" in prompt:
                    has_last_moves += 1

                if i < num_samples:
                    print(f"\n--- Sample {i+1} (id: {rec.get('original_id', '?')[:60]}) ---")
                    # Show just the part after the board
                    lines = prompt.split("\n")
                    board_ended = False
                    for pl in lines:
                        stripped = pl.strip()
                        if stripped.startswith("1 ") and board_ended is False:
                            board_ended = True
                            continue
                        if board_ended:
                            print(f"  {pl}")

            print(f"\n--- Summary ({total} records) ---")
            print(f"  Has 'Valid empty coordinates': {has_valid_coords}/{total} ({100*has_valid_coords/total:.1f}%)")
            print(f"  Has '[Instruction]' block:     {has_instruction_block}/{total} ({100*has_instruction_block/total:.1f}%)")
            print(f"  Has 'Analyze the board':       {has_analyze_line}/{total} ({100*has_analyze_line/total:.1f}%)")
            print(f"  Has 'Last moves:':             {has_last_moves}/{total} ({100*has_last_moves/total:.1f}%)")
            print(f"  Total records:                 {total}")

    except FileNotFoundError:
        print(f"  FILE NOT FOUND: {filepath}")
    except Exception as e:
        print(f"  ERROR: {e}")


def main():
    print("Checking prompt format preservation across pipeline stages.\n")
    print("We want to know if 'Valid empty coordinates' and the")
    print("'[Instruction]' block survive the augmentation step.\n")

    analyze_file(PRE_AUG_FILE, "STAGE 1: Pre-augmentation (training_ready_data_v3)")
    analyze_file(QUALITY_FILTERED_FILE, "STAGE 2: After quality filter")
    analyze_file(POST_AUG_FILE, "STAGE 3: After augmentation + shuffle (ACTUAL TRAINING DATA)")


if __name__ == "__main__":
    main()
