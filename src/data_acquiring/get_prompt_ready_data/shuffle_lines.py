import random

def shuffle_lines(input_file, output_file):
    # Read all lines from the input file
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Shuffle the lines randomly
    random.shuffle(lines)
    
    # Write the shuffled lines to the output file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.writelines(lines)

if __name__ == "__main__":
    input_file = r"C:\git_repo\KataGo-LLM-Team\data\training_ready_data.jsonl"
    output_file = r"C:\git_repo\KataGo-LLM-Team\data\training_ready_data_shuffled.jsonl"
    shuffle_lines(input_file, output_file)
    print("Lines shuffled successfully!")
