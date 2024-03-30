import json
import os
import numpy as np
from PIL import Image
from argparse import ArgumentParser

def generate_and_save_white_noise_image(first_image, image_dest_dir, entry_id, step):
    white_noise_height = first_image.height // 3
    white_noise_image = generate_white_noise_image(first_image.width, white_noise_height)
    combined_image = concatenate_images(white_noise_image, first_image)
    dest_image_path = os.path.join(image_dest_dir, str(entry_id), f"concat_opt_{step}.jpg")
    os.makedirs(os.path.dirname(dest_image_path), exist_ok=True)
    combined_image.save(dest_image_path)

def generate_white_noise_image(width, height):
    noise = np.random.random((height, width, 3)) * 255
    noise_image = Image.fromarray(noise.astype('uint8')).convert('RGB')
    return noise_image

def concatenate_images(image1, image2):
    width1, height1 = image1.size
    width2, height2 = image2.size
    new_width = max(width1, width2)
    new_height = height1 + height2
    new_image = Image.new('RGB', (new_width, new_height))
    new_image.paste(image1, (0, 0))
    new_image.paste(image2, (0, height1))
    return new_image

def generate_white_image(width, height):
    white_image = Image.new('RGB', (width, height), color='white')
    return white_image

def process_data(input_path, image_source_dir, image_dest_dir):
    with open(input_path, 'r') as file:
        raw_data = json.load(file)

    processed_data = {}
    for entry in raw_data:
        entry_id = entry['id']

        if entry_id in processed_data and entry['flagged']:
            if int(entry['step']) < int(processed_data[entry_id]['step']):
                processed_data[entry_id] = entry
        elif entry_id not in processed_data:
            processed_data[entry_id] = entry
        elif not entry['flagged'] and processed_data[entry_id]['flagged']:
            continue
        elif entry['step'] == "5":
            processed_data[entry_id] = entry

    output_list = list(processed_data.values())

    os.makedirs(image_dest_dir, exist_ok=True)

    first_image_path = os.path.join(image_source_dir, '1', 'concat_opt_4.jpg')
    if os.path.isfile(first_image_path):
        first_image = Image.open(first_image_path)
        white_image_height = first_image.height // 3
        white_image = generate_white_image(first_image.width, white_image_height)

    for entry in output_list:
        entry_id = entry['id']
        step = entry['step']
        image_filename = f"concat_opt_{step}.jpg"
        source_image_path = os.path.join(image_source_dir, str(entry_id), image_filename)
        dest_image_path = os.path.join(image_dest_dir, f"{entry_id}.jpg")

        if os.path.isfile(source_image_path):
            original_image = Image.open(source_image_path)
            combined_image = concatenate_images(white_image, original_image)
            combined_image.save(dest_image_path)
        else:
            print(f"Image not found for ID {entry_id}: {source_image_path}")

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input-dir', type=str, default='./eval/evaluate/results/gen_results/{}/black_box/abstract')
    parser.add_argument('--image-source-dir', type=str, default='./dataset/black_box/images')
    parser.add_argument('--image-dest-dir', type=str, default='./dataset/white_box/init_images')
    parser.add_argument('--scenario', type=str, help='Scenario for which to generate keywords', 
                        choices=['Animal', 'Self-harm', 'Privacy', 'Violence', 'Financial'], default='Violence')
    parser.add_argument("--attack-model", type=str, default="llava-15-7b",)
    args = parser.parse_args()

    input_file_path = os.path.join(args.input_dir.format(args.attack_model), f'{args.scenario}.json')
    image_source_dir = os.path.join(args.image_source_dir, args.scenario)
    image_dest_dir = os.path.join(args.image_dest_dir, args.scenario)
    process_data(input_file_path, image_source_dir, image_dest_dir)
