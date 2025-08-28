# SinLlama v1 - Sinhala-Capable Llama 3-8B for Amazon Bedrock

This project merges a Sinhala LoRA adapter with Llama 3-8B base model to create a Sinhala-capable variant for Amazon Bedrock deployment.

## Credits

This work builds upon the excellent Sinhala language adaptation by the Polyglots team:
- **SinLlama LoRA Adapter**: [polyglots/SinLlama_v01](https://huggingface.co/polyglots/SinLlama_v01)
- **Extended Sinhala Tokenizer**: [polyglots/Extended-Sinhala-LLaMA](https://huggingface.co/polyglots/Extended-Sinhala-LLaMA)

## Benefits of Bedrock Deployment

Importing this model to Amazon Bedrock enables:
- **On-demand inference** - Pay only for what you use, no infrastructure management
- **Serverless scaling** - Automatic scaling based on request volume
- **Enterprise security** - Built-in data protection and compliance features
- **API integration** - Easy integration with existing AWS services

For detailed pricing information, refer to [AWS Bedrock Pricing](https://aws.amazon.com/bedrock/pricing/).

## Prerequisites

- EC2 instance with sufficient memory (r8i.4xlarge recommended - 128GB RAM)
- Python environment with transformers==4.51.3
- AWS CLI configured with Bedrock permissions
- ~60GB free disk space
- **Hugging Face account** with access to Llama models
- Hugging Face token for authentication

## Setup Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install transformers==4.51.3 torch peft safetensors accelerate
```

## Hugging Face Authentication

1. Create a Hugging Face account at https://huggingface.co
2. Request access to Llama models at https://huggingface.co/meta-llama/Meta-Llama-3-8B
3. Generate a token at https://huggingface.co/settings/tokens
4. Login with your token:

```bash
hf auth login
# Enter your HF token when prompted
```

## Cache Management (Optional)

Models download to `~/.cache/huggingface/` (~50GB). To use a different location:

```bash
export TRANSFORMERS_CACHE=/path/to/your/cache
export HF_HOME=/path/to/your/cache
```

## Model Merging

The `combine.py` script merges:
- Base model: `meta-llama/Meta-Llama-3-8B`
- Sinhala adapter: `polyglots/SinLlama_v01`
- Extended tokenizer: `polyglots/Extended-Sinhala-LLaMA` (139,336 tokens)

Run the merge:
```bash
python combine.py
```

This creates `merged_sinllama_8b/` directory with the Sinhala-capable model ready for Bedrock import.

## Upload to S3

```bash
# Create S3 bucket (if needed)
aws s3 mb s3://your-bedrock-models-bucket

# Upload model
aws s3 sync merged_sinllama_8b/ s3://your-bedrock-models-bucket/sinllama-8b/
```

## Import to Bedrock

### Via AWS Console:
1. Go to Amazon Bedrock → Model Import
2. Click "Import model"
3. Configure:
   - **Model name**: `sinllama-8b`
   - **S3 URI**: `s3://your-bedrock-models-bucket/sinllama-8b/`
   - **Model format**: `hugging-face`
   - **Architecture**: `llama`
4. Start import job

### Via AWS CLI:
```bash
aws bedrock create-model-import-job \
    --job-name "sinllama-8b-import" \
    --imported-model-name "sinllama-8b" \
    --role-arn "arn:aws:iam::ACCOUNT:role/BedrockModelImportRole" \
    --model-data-source "s3Uri=s3://your-bedrock-models-bucket/sinllama-8b/"
```

## Model Specifications

- **Architecture**: Llama 3-8B
- **Vocabulary**: 139,336 tokens (expanded for Sinhala)
- **Format**: Sharded SafeTensors (11 shards, ~3GB each)
- **Precision**: float32
- **Transformers version**: 4.51.3

## Usage in Bedrock

Once imported, the model supports English and Sinhala text generation using completion-style prompts.

### Screenshots
The following screenshots demonstrate the complete workflow:

1. **S3 Upload** (`screenshots/1.upload-to-s3.png`) - Uploading merged model to S3 bucket
2. **Model Import** (`screenshots/2.import-model.png`) - Importing model into Bedrock
3. **Playground Access** (`screenshots/3.open-playground.png`) - Opening Bedrock playground
4. **Model Inference** (`screenshots/4.model-inference.png`) - Testing Sinhala text generation

### API Usage
```python
import boto3

bedrock = boto3.client('bedrock-runtime')
response = bedrock.invoke_model(
    modelId='<YOUR_IMPORTED_MODEL_ARN>',
    body=json.dumps({
        "prompt": "ආයුබෝවන්! How can I help you today?\n",
        "max_tokens": 50
    })
)
```

## Troubleshooting

- **Memory issues**: Ensure r8i.4xlarge or similar (128GB+ RAM)
- **Import failures**: Verify S3 permissions and file structure
- **Version conflicts**: Use exact transformers==4.51.3
