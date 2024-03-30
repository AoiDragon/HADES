#!/bin/bash

export OPENAI_API_KEY="your_openai_api_key_here" # Set OpenAI API key as an environment variable
TEXT_DIR="./dataset/instructions" # Assuming this is the directory where instructional texts are stored
OUTPUT_DIR="./dataset/black_box/captions" # Assuming this is the directory where generated captions will be saved
SD_MODEL_PATH="./checkpoint/pixart" # Path to PixArtAlphaPipeline pretrained model
LLAVA_MODEL_PATH="./checkpoint/llava-1.5-7b" # Path to LLaMA pretrained model
SD_IMAGE_DIR_BASE="./dataset/black_box/images" # Base directory to save SD images
LOG_DIR_BASE="./dataset/black_box/logs" # Base directory to save logs

# Array of scenarios
SCENARIOS=('Animal' 'Self-harm' 'Privacy' 'Violence' 'Financial')

for SCENARIO in "${SCENARIOS[@]}"; do
  echo "Processing scenario: $SCENARIO"

  # Generate captions
  python black_box/generate_caption.py \
    --text-dir "$TEXT_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --scenario "$SCENARIO"
  
  # Generate sd_captions
  python black_box/generate_sd_caption.py \
    --text-dir "$OUTPUT_DIR" \
    --scenario "$SCENARIO"
  
  # Optimize harm
  python black_box/optimize_harm.py \
    --sd_model_path "$SD_MODEL_PATH" \
    --llava_model_path "$LLAVA_MODEL_PATH" \
    --input_file "$OUTPUT_DIR" \
    --sd_image_dir_base "$SD_IMAGE_DIR_BASE" \
    --log_dir_base "$LOG_DIR_BASE" \
    --scenario "$SCENARIO"
  
  echo "Finished processing scenario: $SCENARIO"
done
echo "All scenarios processed successfully."
