"""
Gradio demo for zero-shot TTS using Blueberry model.

Usage:
python gradio_demo.py
"""

import sys
from pathlib import Path

import gradio as gr
import librosa
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, MimiModel

# Add parent directory to path to import utils
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import audio_to_str, resample_audio, str_to_audio

# Constants
MIMI_MODEL_ID = "kyutai/mimi"
MARIN_MODEL_NAME = "WillHeld/blueberry"
SOURCE_SAMPLE_RATE = 16000
TARGET_SAMPLE_RATE = 24000

# Global variables for models (loaded once)
mimi_model = None
marin_model = None
tokenizer = None
device = None


def load_models():
    """Load models once at startup."""
    global mimi_model, marin_model, tokenizer, device
    
    print("Loading models...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load Mimi model
    print(f"Loading Mimi model: {MIMI_MODEL_ID}")
    mimi_model = MimiModel.from_pretrained(MIMI_MODEL_ID)
    mimi_model = mimi_model.to(device)
    
    # Load Marin audio model
    print(f"Loading Marin model: {MARIN_MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MARIN_MODEL_NAME)
    marin_model = AutoModelForCausalLM.from_pretrained(
        MARIN_MODEL_NAME, torch_dtype=torch.float32, device_map="auto"
    )
    marin_model.eval()
    print(f"Number of parameters: {sum(p.numel() for p in marin_model.parameters())}")
    print("Models loaded successfully!")


def run_inference(
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
):
    """Run inference with the model."""
    inputs = tokenizer(prompt, return_tensors="pt").to(marin_model.device)
    with torch.no_grad():
        outputs = marin_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=128259,  # 128259 = <|audio_end|>
        )
    input_len = inputs.input_ids.shape[1]
    generated_audio_str = tokenizer.decode(outputs[0, input_len:], skip_special_tokens=False)
    if "<|audio_end|>" in generated_audio_str:
        generated_audio_str = generated_audio_str.replace("<|audio_end|>", "")
    return generated_audio_str


def generate_speech(
    prompt_text: str,
    prompt_audio_path: str,
    target_text: str,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    seed: int,
):
    """Generate speech using zero-shot TTS."""
    if not prompt_text or not prompt_audio_path or not target_text:
        return None, "Please provide all required inputs: prompt text, prompt audio, and target text."
    
    # Set seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    
    # Load and resample the prompt audio
    audio, sr = librosa.load(prompt_audio_path, sr=None)
    audio_resampled = resample_audio(audio, orig_sr=sr, target_sr=TARGET_SAMPLE_RATE)
    
    # Convert to Mimi audio string
    audio_str = audio_to_str(audio_resampled, mimi_model, device)
    
    # Generate zero-shot TTS
    prompt = (
        f"<|text_start|>{prompt_text}<|text_end|>"
        f"<|audio_start|>{audio_str}<|audio_end|>"
        f"<|text_start|>{target_text}<|text_end|>"
        f"<|audio_start|>"
    )
    
    generated_audio_str = run_inference(
        prompt, max_new_tokens, temperature, top_p
    )
    
    # Ensure the length is divisible by 8
    N = len(generated_audio_str)
    generated_audio_str = generated_audio_str[:(N//8)*8]
    
    # Convert to audio
    audio_numpy = str_to_audio(generated_audio_str, mimi_model, device)
    
    # Gradio expects (sample_rate, audio_array)
    return (TARGET_SAMPLE_RATE, audio_numpy.T), "Generation successful!"


def create_demo():
    """Create Gradio interface."""
    with gr.Blocks(title="Zero-Shot TTS Demo") as demo:
        gr.Markdown("# 🎙️ Zero-Shot Text-to-Speech Demo")
        gr.Markdown("Generate speech in any voice using just a short audio prompt!")
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Input")
                prompt_text = gr.Textbox(
                    label="Prompt Text",
                    placeholder="The text that corresponds to the prompt audio...",
                    lines=3,
                )
                prompt_audio = gr.Audio(
                    label="Prompt Audio",
                    type="filepath",
                    sources=["upload", "microphone"],
                )
                target_text = gr.Textbox(
                    label="Target Text",
                    placeholder="The text you want to synthesize...",
                    lines=3,
                )
                
                gr.Markdown("### Generation Parameters")
                with gr.Row():
                    temperature = gr.Slider(
                        minimum=0.1,
                        maximum=2.0,
                        value=0.9,
                        step=0.1,
                        label="Temperature",
                        info="Higher values = more random"
                    )
                    top_p = gr.Slider(
                        minimum=0.1,
                        maximum=1.0,
                        value=0.8,
                        step=0.05,
                        label="Top-p",
                        info="Nucleus sampling threshold"
                    )
                
                with gr.Row():
                    max_new_tokens = gr.Slider(
                        minimum=100,
                        maximum=3000,
                        value=1500,
                        step=100,
                        label="Max New Tokens",
                        info="Maximum length of generated audio"
                    )
                    seed = gr.Number(
                        value=42,
                        label="Random Seed",
                        precision=0,
                        info="For reproducibility"
                    )
                
                generate_btn = gr.Button("🎵 Generate Speech", variant="primary", size="lg")
            
            with gr.Column():
                gr.Markdown("### Output")
                output_audio = gr.Audio(
                    label="Generated Speech",
                    type="numpy",
                )
                status_text = gr.Textbox(
                    label="Status",
                    interactive=False,
                )
        
        # Examples
        gr.Markdown("### 📝 Example Usage")
        gr.Markdown("""
        1. Upload a short audio clip (prompt audio) of someone speaking
        2. Enter the text that corresponds to that audio (prompt text)
        3. Enter the text you want to synthesize in that voice (target text)
        4. Adjust generation parameters if desired
        5. Click "Generate Speech"
        """)
        
        # Connect the button
        generate_btn.click(
            fn=generate_speech,
            inputs=[
                prompt_text,
                prompt_audio,
                target_text,
                temperature,
                top_p,
                max_new_tokens,
                seed,
            ],
            outputs=[output_audio, status_text],
        )
    
    return demo


def main():
    """Main function to launch the demo."""
    load_models()
    demo = create_demo()
    demo.launch(share=True, server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    main()

