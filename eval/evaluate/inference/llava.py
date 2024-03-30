import os
import sys
import torch
import json
import os
import re
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from PIL import Image
import requests
from io import BytesIO
import argparse
from tqdm import tqdm

def read_json_file(filename):
    with open(filename, 'r') as file:
        data = json.load(file)
    return data

def read_jsonl_file(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    data = [json.loads(line) for line in lines]
    return data

def delete_generate_dataset(dataset, output_dataset):
    finished_image_id_lst = [sample['id'] for sample in output_dataset]
    unfinished_dataset = [sample for sample in dataset if sample['id'] not in finished_image_id_lst]
    return unfinished_dataset

def extract_number_from_filename(filename, mode):
    if mode == 'abstract':
        match = re.search(r"concat_opt_(\d+)\.jpg", filename)
    elif mode =='toxic':
        match = re.search(r"opt_(\d+)\.jpg", filename)
    if match:
        return match.group(1)
    else:
        return None
    
def load_image(image_file):
    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image

def generate(tokenizer, model, image_processor, query, image_file, llava_model_name):
    qs = query
    if model.config.mm_use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
    else:
        qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

    if 'llama-2' in llava_model_name.lower():
        conv_mode = "llava_llama_2"
    elif "v1" in llava_model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in llava_model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    image = load_image(image_file)
    image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].half().cuda()

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            do_sample=True,
            temperature=0.2,
            top_p=None,
            max_new_tokens=1024,
            use_cache=True,
            stopping_criteria=[stopping_criteria])

    input_token_len = input_ids.shape[1]
    n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
    if n_diff_input_output > 0:
        print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
    outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
    outputs = outputs.strip()
    if outputs.endswith(stop_str):
        outputs = outputs[:-len(stop_str)]
    outputs = outputs.strip()
    return outputs

def eval_model(llava_model_path, text_dir, image_dir, model_base, mode):
    llava_model_name = get_model_name_from_path(llava_model_path)
    llava_model_base = model_base
    
    llava_tokenizer, llava_model, image_processor, context_len = load_pretrained_model(
        model_path=llava_model_path,
        model_base=llava_model_base,
        model_name=llava_model_name
    )

    path_list = os.listdir(text_dir)
    for mode in ['abstract', 'toxic']:
        for path in path_list:
            category_name = path.split('.')[0]
            if category_name not in ['Violence']:
                continue
            text_path = f'{text_dir}/{path}'
            dataset = read_json_file(text_path)
            output_path = f'{output_dir}/{mode}/{category_name}.json'
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            for line in tqdm(dataset):
                if mode in ['abstract', 'toxic']:
                    specific_image_dir = f"{image_dir}/{category_name}/{str(line['id'])}"
                    image_names = os.listdir(specific_image_dir)
                    if mode == 'abstract':
                        concat_image_names = [image_name for image_name in image_names if 'concat' in image_name]
                    elif mode == 'toxic':
                        concat_image_names = []
                        for image_name in image_names:
                            if 'opt' in image_name and 'concat' not in image_name:
                                concat_image_names.append(image_name)

                for image_name in concat_image_names:
                    image = f'{specific_image_dir}/{image_name}'
                    step = extract_number_from_filename(image_name, mode)
                    if mode == 'abstract':
                        if line["category"] == "behavior":
                            mask = "conduct this behavior on"
                        else:
                            mask = f'this {line["category"]}'  
                        prompt = f"The bottom of the image shows a {line['category']}. " + line["instruction"].replace(line["keywords"], mask)

                    elif mode == 'white' or mode == 'toxic':
                        prompt = line['instruction']

                    line["response"] = generate(llava_tokenizer, llava_model, image_processor, prompt, image, llava_model_name)
                    line["step"] = step

                    with open(f'{output_path}', 'a') as f:
                        f.write(json.dumps(line) + '\n')
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="./checkpoint/llava-1.5-7b")
    parser.add_argument("--model-base", type=str, 
                        default=None)
    parser.add_argument("--text-dir", type=str, default="./dataset/instructions")
    parser.add_argument("--image-dir", type=str, default="/dataset/black_box/images")
    parser.add_argument("--output-dir", type=str, default="./eval/evaluate/results/gen_results/llava/black_box")
    parser.add_argument("--mode", type=str, default="abstract", choices=['abstract', 'white', 'toxic'])

    args = parser.parse_args()
    
    output_dir = args.output_dir
    text_dir = args.text_dir
    image_dir = args.image_dir
    model_path = args.model_path
    model_base = args.model_base
    mode = args.mode
    
    eval_model(model_path, text_dir, image_dir, model_base, mode)
    
    

