import gradio as gr
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer # BitsAndBytesConfig
import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Disable CUDA for Gradio
os.environ["TRANSFORMERS_CACHE"] = "/tmp/huggingface_cache"
os.environ["HF_HOME"] = "/tmp/huggingface_home"


# Model and tokenizer loading
model_name = "Qwen/Qwen2.5-0.5B"  # Replace with your base model name
adapter_path = "grpo_adapter"  # Path to your adapter directory (relative to app.py)

# temporarily commented quantization
# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=torch.float16,
# )

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    # quantization_config=bnb_config,
    trust_remote_code=True,
    torch_dtype=torch.float32,  # Use float16 for faster inference
)
model = PeftModel.from_pretrained(model, adapter_path)
model.config.use_sliding_window_attention = False
model.to(torch.device("cpu"))  # Explicitly move adapter to CPU
model.eval()

# Inference function
def generate_text(user_prompt, max_length=200, temperature=0.7, top_p=0.9):

    messages = [
        {"role": "system", "content": "You are Qwen, a helpful assistant."},
        {"role": "user", "content": user_prompt},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(torch.device("cpu"))

    # Generate output
    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_length,  # Use max_new_tokens instead of max_length
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.pad_token_id,
            do_sample = True
        )

    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    # Decode the generated output
    generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return generated_text

# Sample questions
sample_questions = [
    "What are Large Language Models?",
    "What is 2+2?",
    "Write a Flask App in python to say 'Hello World!'",
    "Give me a short 200-word essay on 'India in AI'",
]

# Gradio interface
iface = gr.Interface(
    fn=generate_text,
    inputs=[
        gr.Textbox(lines=5, label="Prompt"),
        gr.Slider(minimum=50, maximum=500, value=250, label="Max Length"),
        gr.Slider(minimum=0.1, maximum=1.0, value=0.1, label="Temperature"),
        gr.Slider(minimum=0.1, maximum=1.0, value=0.9, label="Top P"),
    ],
    outputs=gr.Textbox(label="Generated Text"),
    title="Qwen 2.5 GRPO RL Model Demo",
    description="Generate text using a base Qwen-2.5-0.5B parameters model fine-tuned with GRPO Trainer dataset. Click a sample question below to get started!",
    examples=[[q, 300, 0.1, 0.9] for q in sample_questions],  # Add examples
)

iface.launch()