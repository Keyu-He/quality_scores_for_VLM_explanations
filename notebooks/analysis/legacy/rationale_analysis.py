import os
import re
import nltk
import torch
import json
import hashlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# load the tokenizer and model only if they are not already defined
if 'nli_tokenizer' not in globals():
    nli_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-xxl")
if 'nli_model' not in globals():
    nli_model = AutoModelForSeq2SeqLM.from_pretrained("soumyasanyal/nli-entailment-verifier-xxl", load_in_8bit=True, device_map="auto")

def calculate_bleu(reference, candidate):
    reference = [reference.split()]
    candidate = candidate.split()
    smoothing_function = SmoothingFunction().method1
    score = sentence_bleu(reference, candidate, smoothing_function=smoothing_function)
    return score

def calculate_rouge(reference, candidate):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, candidate)
    return scores

def calculate_meteor(reference, candidate):
    reference_tokens = reference.split()
    candidate_tokens = candidate.split()
    score = meteor_score([reference_tokens], candidate_tokens)
    return score

def display_scores(reference, candidate):
    bleu_score = calculate_bleu(reference, candidate)
    rouge_scores = calculate_rouge(reference, candidate)
    meteor_score = calculate_meteor(reference, candidate)
    print(f"BLEU Score: {bleu_score:.4f}")
    print("ROUGE Scores:")
    for key, value in rouge_scores.items():
        print(f"  {key}: {value}")
    print(f"METEOR Score: {meteor_score:.4f}")
    
def get_longest_rationale(rationale_list):
    rationales = eval(rationale_list)
    return max(rationales, key=len) if isinstance(rationales, list) else ''

def calc_low_support_score(premise, hypothesis, nli_model=nli_model, nli_tokenizer=nli_tokenizer):
    def get_score(nli_model, nli_tokenizer, input_ids):
        pos_ids = nli_tokenizer('Yes').input_ids
        neg_ids = nli_tokenizer('No').input_ids
        pos_id = pos_ids[0]
        neg_id = neg_ids[0]

        with torch.no_grad():
            logits = nli_model(input_ids, decoder_input_ids=torch.zeros((input_ids.size(0), 1), dtype=torch.long)).logits
            pos_logits = logits[:, 0, pos_id]
            neg_logits = logits[:, 0, neg_id]
            posneg_logits = torch.cat([pos_logits.unsqueeze(-1), neg_logits.unsqueeze(-1)], dim=1)

            # Cast to float before applying softmax
            posneg_logits = posneg_logits.float()
            scores = torch.nn.functional.softmax(posneg_logits, dim=1)
            entail_score = scores[:, 0].item()
            contra_score = scores[:, 1].item()
        
        return entail_score, contra_score
    
    prompt = f"Premise: {premise}\nHypothesis: {hypothesis}\nGiven the premise, is the hypothesis correct?\nAnswer:"
    input_ids = nli_tokenizer(prompt, return_tensors='pt').input_ids
    return get_score(nli_model, nli_tokenizer, input_ids)[1]

def generate_mask(generated_rationale, predicted_answer):
    # Create a regex pattern to match the predicted answer case-insensitively and as a whole word
    pattern = re.compile(r'\b' + re.escape(predicted_answer) + r'\b', re.IGNORECASE)
    return pattern.sub("<mask>", generated_rationale)

def evaluate_support(data, nli_model, nli_tokenizer, hypothesis_col='hypothesis', threshold=0.5):
    support_scores = []
    for idx, row in data.iterrows():
        premise = row['gen_rationale_mask']
        hypothesis = row[hypothesis_col]
        no_entail_prob = calc_low_support_score(premise, hypothesis, nli_model=nli_model, nli_tokenizer=nli_tokenizer)
        support = no_entail_prob < threshold 
        if no_entail_prob > threshold:
            print(f"Premise: {premise}")
            print(f"Hypothesis: {hypothesis}")
            print(f"Probability: {no_entail_prob}")
        support_scores.append({
            'no_entail_prob': no_entail_prob,
            'support': support
        })
    return support_scores

def compute_file_hash(file_path):
    hasher = hashlib.md5()
    with open(file_path, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

# Main
def main(file_path):
    print(f"Processing {file_path}...")
    
    if file_path == '../results/Human Annotation of LLaVA+ Rationales.xlsx':
        model_name = "LLaVA"
    else:
        model_name = file_path.split('results/')[1].split('.xlsx')[0]
    
    spreadsheet = pd.ExcelFile(file_path)
    
    # Read the specified columns from the sheet
    columns_to_read = [
        'question',
        'correct_answer',
        'predicted_answer',
        'is_correct',
        'groundtruth_rationale',
        'generated_rationale'
    ]

    if file_path == '../results/Human Annotation of LLaVA+ Rationales.xlsx':
        data = pd.read_excel(file_path, header=1, usecols=columns_to_read)
    else:
        data = pd.read_excel(file_path, usecols=columns_to_read)
    data['question_no_choice'] = data.apply(lambda row: row['question'].split(' Choices:')[0], axis=1)
    data['longest_groundtruth_rationale'] = data['groundtruth_rationale'].apply(get_longest_rationale)
    
    data['BLEU_score'] = data.apply(lambda row: calculate_bleu(row['longest_groundtruth_rationale'], row['generated_rationale']), axis=1)
    data['ROUGE_scores'] = data.apply(lambda row: calculate_rouge(row['longest_groundtruth_rationale'], row['generated_rationale']), axis=1)
    data['METEOR_score'] = data.apply(lambda row: calculate_meteor(row['longest_groundtruth_rationale'], row['generated_rationale']), axis=1)
    
    input_data = data[['question', 'predicted_answer']].copy()
    input_data['question'] = input_data['question'].apply(lambda x: x.split(' Choices:')[0])
    input_data.rename(columns={'question': 'question_text', 'predicted_answer': 'answer_text'}, inplace=True)
    input_jsonl = f'input_data_{model_name}.jsonl'
    output_jsonl = f'{input_jsonl}.predictions'
    with open(input_jsonl, 'w') as f:
        for index, row in input_data.iterrows():
            # Convert None to null
            row_dict = {k: (v if pd.notna(v) else None) for k, v in row.to_dict().items()}
            json.dump(row_dict, f)
            f.write('\n')
    # Compute hash of input_jsonl
    current_input_hash = compute_file_hash(input_jsonl)
    hash_file = f'{input_jsonl}.hash'
    # Check if output_jsonl exists and input_jsonl hash hasn't changed
    run_bash_command = True
    if os.path.exists(output_jsonl):
        if os.path.exists(hash_file):
            with open(hash_file, 'r') as f:
                saved_input_hash = f.read().strip()
                if current_input_hash == saved_input_hash:
                    run_bash_command = False
    # Save the current input_jsonl hash
    with open(hash_file, 'w') as f:
        f.write(current_input_hash)
    
    # The conversion step
    # Define the full path to the script
    script_path = '/home/<link_hidden>/REV/run_question_converter.sh'
    if run_bash_command:
        # Set PYTHONPATH and run the script
        os.system(f'export PYTHONPATH=/home/<link_hidden>/REV/:$PYTHONPATH && bash {script_path} cqa {input_jsonl} cuda:0')
    else:
        print(f'{output_jsonl} already exists and {input_jsonl} has not changed. Skipping the bash command.')
    
    with open(output_jsonl, 'r') as f:
        predictions = [json.loads(line) for line in f]
    predictions_df = pd.DataFrame(predictions)
    predictions_df.rename(columns={'question_statement_text': 'hypothesis'}, inplace=True)
    
    # Merge datasets based on the 'question' column
    data = pd.merge(data, predictions_df[['question_text', 'hypothesis']], left_on='question_no_choice', right_on='question_text', how='left')
    
    data['gen_rationale_mask'] = data.apply(lambda row: generate_mask(row['generated_rationale'], row['predicted_answer']), axis=1)

    # Evaluate support
    support_results = evaluate_support(data, nli_model, nli_tokenizer)
    support_df = pd.DataFrame(support_results)
    for column in support_df.columns:
        data[column] = support_df[column]
        
    # Plot the distribution of no_entail_prob
    plt.figure(figsize=(8, 3))
    plt.hist(data['no_entail_prob'], bins=50, edgecolor='black')
    plt.title('Distribution of no_entail_prob')
    plt.xlabel('no_entail_prob')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()
    
    support_score = data['support'].mean()
    
    print(f"Support score: {support_score}")
    
if __name__ == '__main__':
    file_paths = [
#                   '../results/Human Annotation of LLaVA+ Rationales.xlsx',
#                   "../results/gpt-4o_inference_one_shot_50_improved_prompt.xlsx",
#                   "../results/gpt-4o_inference_two_steps_50.xlsx",
#                   "../results/instructblip-flan-t5-xxl_inference_one_shot_50_improved_prompt.xlsx",
                    "../results/llava-1.5-7b-hf_inference_no_vision.xlsx",
                 ]
    for file_path in file_paths:
        main(file_path)