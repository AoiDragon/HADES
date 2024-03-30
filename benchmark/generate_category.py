import openai
import argparse
import json
import os

def classify_keywords(api_key, scenario, model):
    openai.api_key = api_key
    background_prompt = "I will provide you with a keyword, and you need to classify it into its semantic category. Your response should be either \"object\", \"behavior\", or \"concept\", using only one lowercase word. The keyword is {}."
    input_file = f'./dataset/keywords/{scenario}.json'
    output_file = f'./dataset/category/{scenario}.json'
    
    # Ensure that the output directory exists before writing the file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(input_file, 'r') as f:
        dataset = json.load(f)
    for line in dataset:
        keywords = line["keywords"]
        prompt = background_prompt.format(keywords)
        message = [{'role': 'user', 'content': prompt}]
        
        response = openai.ChatCompletion.create(
                model=model,
                messages=message,
            )
        answer = response["choices"][0]["message"]["content"].strip()
        line["category"] = answer

    with open(output_file, 'w') as f:
        json.dump(dataset, f, indent=4, ensure_ascii=False)

def parse_args():
    parser = argparse.ArgumentParser(description='Classify keywords into semantic categories using OpenAI.')
    parser.add_argument('--api_key', type=str, help='OpenAI API key', required=True)
    parser.add_argument('--scenario', type=str, help='Scenario for which to generate keywords', choices=['Animal', 'Self-harm', 'Privacy', 'Violence', 'Financial'], default='Violence')
    parser.add_argument('--model', type=str, help='Model to use for classification', default='gpt-4-0314')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    classify_keywords(args.api_key, args.scenario, args.model)
