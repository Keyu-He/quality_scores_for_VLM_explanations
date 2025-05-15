import json
from argparse import ArgumentParser
import random
import os

args = ArgumentParser()
args.add_argument("-d", "--data", default="../data/sample_data.json")
args.add_argument("-n", "--name", default="fm2_llama3.1_8b")
args.add_argument("-i", "--num_instances_per_queue", type=int, default=50)
args.add_argument("-q", "--num_queues", default=1, type=int)
args.add_argument("-s", "--seed", default=0, type=int)
args = args.parse_args()

random.seed(args.seed)

data = json.load(open(args.data))

# Split data into correct and incorrect predictions
correct_data = [item for item in data if item["prediction_is_correct"] == 1]
incorrect_data = [item for item in data if item["prediction_is_correct"] == 0]
print(f"Correct: {len(correct_data)}, Incorrect: {len(incorrect_data)}")

# Calculate the total number of instances required
total_instances_required = args.num_instances_per_queue * args.num_queues
instances_per_class = args.num_instances_per_queue // 2

# Check if there are enough correct and incorrect instances
if instances_per_class * args.num_queues > min(len(correct_data), len(incorrect_data)):
    raise ValueError(
        f"Not enough balanced data to create {args.num_queues} queues with {instances_per_class} correct and {instances_per_class} incorrect instances each."
    )

# Shuffle correct and incorrect data
random.shuffle(correct_data)
random.shuffle(incorrect_data)

# Create output directory
out_dirname = f"web/baked_queues/{args.name}_q{args.num_queues}_i{args.num_instances_per_queue}_s{args.seed}"
os.makedirs(out_dirname, exist_ok=True)

# Generate balanced queues
for uid in range(args.num_queues):
    start_idx = uid * instances_per_class
    end_idx = start_idx + instances_per_class
    
    sampled_correct = correct_data[start_idx:end_idx]
    sampled_incorrect = incorrect_data[start_idx:end_idx]
    
    # Combine and shuffle the sampled data
    sampled_data = sampled_correct + sampled_incorrect
    random.shuffle(sampled_data)
    
    out_file = f"{out_dirname}/{uid:0>3}.json"
    if os.path.exists(out_file):
        print(f"Overwriting {out_file}")
    else:
        print(f"Writing {out_file}")
    
    # Write to JSON file
    json.dump(sampled_data, open(out_file, "w"), indent=2)