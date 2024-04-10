#!/bin/bash

# Set common environment variables
# MODE should be one of 'abstract', 'white', 'toxic'
MODE=$1
# MODEL should be 'llava', 'gpt4v', or 'gemini'
MODEL=$2
# The version of HADES
BOX_TYPE=$3

TEXT_DIR="./dataset/instructions"
if [ "$BOX_TYPE" == "hades" ]; then
    SAVE_DIR="./dataset/white_box"
else
    IMAGE_DIR="./dataset/black_box/images"
fi
OUTPUT_DIR="./eval/evaluate/results/gen_results"
API_KEY="<your_api_key_here>" # Replace with your actual API key

# Set model-specific environment variables
LLAVA_MODEL_PATH="./checkpoint/llava-1.5-7b"  # Replace with your actual LLaVA model path
LLAVA_MODEL_BASE="llava_v1_5"  # Replace with your actual LLaVA model base identifier

LLAVA_OUTPUT_DIR="$OUTPUT_DIR/llava/black_box"
GPT4V_OUTPUT_DIR="$OUTPUT_DIR/gpt4v/black_box"
GEMINI_OUTPUT_DIR="$OUTPUT_DIR/gemini/black_box"

# MODE should be one of 'abstract', 'white', 'toxic'
MODE=$1
# MODEL should be 'llava', 'gpt4v', or 'gemini'
MODEL=$2

# Using a case statement to handle different models
case $MODEL in
  llava)
    echo "Starting LLaVA evaluation..."
    python path/to/your/llava_inference.py \
      --model_path "$LLAVA_MODEL_PATH" \
      --model-base "$LLAVA_MODEL_BASE" \
      --text-dir "$TEXT_DIR" \
      --image-dir "$IMAGE_DIR" \
      --output-dir "$LLAVA_OUTPUT_DIR" \
      --mode "$MODE"
    echo "LLaVA evaluation for mode '$MODE' is complete."
    ;;
  
  gpt4v)
    echo "Starting GPT-4v evaluation..."
    python path/to/your/gpt4v_inference.py \
      --text-dir "$TEXT_DIR" \
      --image-dir "$IMAGE_DIR" \
      --output-dir "$GPT4V_OUTPUT_DIR" \
      --mode "$MODE" \
      --api_key "$API_KEY"
    echo "GPT-4v evaluation for mode '$MODE' is complete."
    ;;
  
  gemini)
    echo "Starting Gemini-V Pro evaluation..."
    python path/to/your/gemini_inference.py \
      --text-dir "$TEXT_DIR" \
      --image-dir "$IMAGE_DIR" \
      --output-dir "$GEMINI_OUTPUT_DIR" \
      --mode "$MODE" \
      --api_key "$API_KEY"
    echo "Gemini-V Pro evaluation for mode '$MODE' is complete."
    ;;
  
  *)
    echo "Error: Invalid model selection. Please choose 'llava', 'gpt4v', or 'gemini'."
    exit 1
    ;;
esac

# Set the correct evaluation script based on the box type
if [ "$BOX_TYPE" == "hades" ]; then
    EVALUATION_SCRIPT="evaluate_for_hades.py"
else
    EVALUATION_SCRIPT="evaluate_for_black_box.py"
fi

# Perform evaluation
echo "Starting evaluation for $BOX_TYPE using $EVALUATION_SCRIPT..."
EVAL_DATASET_PATH="$MODEL_OUTPUT_DIR/$MODE/"  # The path where generated results are stored

python eval/evaluate/$EVALUATION_SCRIPT \
  --eval_dataset_path "$EVAL_DATASET_PATH" \
  --model_path "$EVAL_MODEL_PATH" \
  --max_length "$MAX_LENGTH"

echo "Evaluation completed for '$MODEL' model under '$MODE' mode and $BOX_TYPE."
