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
from transformers.generation import LogitsProcessor

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


class SuppressTokensLogitsProcessor(LogitsProcessor):
    """Logits processor that suppresses specific tokens by setting their logits to -inf."""
    
    def __init__(self, suppress_tokens: list):
        self.suppress_tokens = suppress_tokens
    
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        scores[:, self.suppress_tokens] = -float('inf')
        return scores


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


def run_inference_no_postprocessing(
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    end_token_id: int, # 128259 = <|audio_end|>, 128257 = <|text_end|>, 128001 = <|end_of_text|>
    min_new_tokens: int = 0,
    suppress_tokens: list = None,
):
    """Run inference with the model."""
    inputs = tokenizer(prompt, return_tensors="pt").to(marin_model.device)
    
    # Set up logits processor if needed
    logits_processor = None
    if suppress_tokens:
        logits_processor = [SuppressTokensLogitsProcessor(suppress_tokens)]
    
    with torch.no_grad():
        outputs = marin_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=end_token_id,
            logits_processor=logits_processor,
        )
    return outputs[0], inputs.input_ids.shape[1]


def extract_text_and_audio_segments(generated_text: str):
    """Extract text and audio segments from generated output."""
    text_segments = []
    audio_segments = []
    
    # Split by text markers
    text_parts = generated_text.split("<|text_start|>")
    for part in text_parts[1:]:  # Skip the first empty part
        if "<|text_end|>" in part:
            text_content = part.split("<|text_end|>")[0]
            text_segments.append(text_content)
    
    # Split by audio markers
    audio_parts = generated_text.split("<|audio_start|>")
    for part in audio_parts[1:]:  # Skip the first part
        if "<|audio_end|>" in part:
            audio_content = part.split("<|audio_end|>")[0]
            audio_segments.append(audio_content)
        else:
            # Handle incomplete audio segment
            audio_content = part.strip()
            if audio_content:
                audio_segments.append(audio_content)
    
    return text_segments, audio_segments


def generate_zero_shot_tts(
    prompt_text: str,
    prompt_audio_path: str,
    target_text: str,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    min_new_tokens: int,
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
    
    output, prompt_length = run_inference_no_postprocessing(
        prompt, max_new_tokens, temperature, top_p, end_token_id=128259, min_new_tokens=min_new_tokens, # <|audio_end|>
    )

    generated_audio_str = tokenizer.decode(output[prompt_length:], skip_special_tokens=False)
    if "<|audio_end|>" in generated_audio_str:
        generated_audio_str = generated_audio_str.replace("<|audio_end|>", "")
    
    # Ensure the length is divisible by 8
    N = len(generated_audio_str)
    generated_audio_str = generated_audio_str[:(N//8)*8]
    
    # Convert to audio
    audio_numpy = str_to_audio(generated_audio_str, mimi_model, device)

    # Gradio expects (sample_rate, audio_array)
    return (TARGET_SAMPLE_RATE, audio_numpy.T), "Generation successful!"


def generate_speech_continuation(
    audio_path: str,
    continuation_method: str,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    min_new_tokens: int,
    seed: int,
    suppress_tokens_str: str,
):
    """Generate speech continuation."""
    if not audio_path:
        return None, "Please provide an audio file."
    
    # Set seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    
    # Parse suppress tokens
    suppress_tokens = None
    if suppress_tokens_str and suppress_tokens_str.strip():
        suppress_tokens = [int(t.strip()) for t in suppress_tokens_str.split(",") if t.strip()]
    
    # Load and resample the audio
    audio, sr = librosa.load(audio_path, sr=None)
    audio_resampled = resample_audio(audio, orig_sr=sr, target_sr=TARGET_SAMPLE_RATE)
    
    # Convert to Mimi audio string
    audio_str = audio_to_str(audio_resampled, mimi_model, device)
    
    # Generate continuation based on method
    if continuation_method == "Method 1 (Direct continuation)":
        prompt = f"<|audio_start|>{audio_str}"
    else:  # Method 2
        prompt = f"<|audio_start|>{audio_str}<|audio_end|><|text_start|>"
    
    output, _ = run_inference_no_postprocessing(
        prompt, max_new_tokens, temperature, top_p, end_token_id=128001, min_new_tokens=min_new_tokens, suppress_tokens=suppress_tokens, # <|end_of_text|>
    )
    generated_text = tokenizer.decode(output, skip_special_tokens=False)
    
    # Extract and concatenate all audio segments
    texts, audios = extract_text_and_audio_segments(generated_text)
    audio_continuation_str = "".join(audios)
    N = len(audio_continuation_str)
    audio_continuation_str = audio_continuation_str[:(N//8)*8]
    
    if not audio_continuation_str:
        return None, f"No audio content generated.\n\nGenerated text:\n{generated_text}"
    
    # Convert to audio
    audio_numpy = str_to_audio(audio_continuation_str, mimi_model, device)
    
    # Format status message with generated text chunks
    status_lines = ["Generation successful!\n"]
    status_lines.append(f"Text segments: {len(texts)}")
    status_lines.append(f"Audio segments: {len(audios)}")
    status_lines.append(f"Audio characters: {len(audio_continuation_str)}\n")
    status_lines.append("=" * 80)
    status_lines.append("Generated text:")
    status_lines.append("=" * 80)
    
    # Split generated text into chunks and print each on one line
    chunks = []
    current_chunk = ""
    in_tag = False
    
    for char in generated_text:
        current_chunk += char
        if char == '<':
            in_tag = True
        elif char == '>':
            in_tag = False
            if any(tag in current_chunk for tag in ['<|text_start|>', '<|text_end|>', '<|audio_start|>', '<|audio_end|>', '<|end_of_text|>']):
                chunks.append(current_chunk)
                current_chunk = ""
    
    if current_chunk:
        chunks.append(current_chunk)
    
    for chunk in chunks:
        status_lines.append(chunk)
    
    status_message = "\n".join(status_lines)
    
    return (TARGET_SAMPLE_RATE, audio_numpy.T), status_message


def generate_transcription(
    audio_path: str,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    min_new_tokens: int,
    seed: int,
):
    """Generate transcription using ASR."""
    if not audio_path:
        return "", "Please provide an audio file."
    
    # Set seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    
    # Load and resample the audio
    audio, sr = librosa.load(audio_path, sr=None)
    audio_resampled = resample_audio(audio, orig_sr=sr, target_sr=TARGET_SAMPLE_RATE)
    
    # Convert to Mimi audio string
    audio_str = audio_to_str(audio_resampled, mimi_model, device)
    
    # Generate transcription
    prompt = f"<|audio_start|>{audio_str}<|audio_end|><|text_start|>"
    output, _ = run_inference_no_postprocessing(
        prompt, max_new_tokens, temperature, top_p, end_token_id=128257, min_new_tokens=min_new_tokens, # <|text_end|>
    )
    generated_text = tokenizer.decode(output, skip_special_tokens=False)

    # Extract transcribed text
    texts, _ = extract_text_and_audio_segments(generated_text)
    transcribed_text = texts[0] if texts else ""
    
    if not transcribed_text:
        return "", "No transcription generated."
    
    return transcribed_text, "Transcription successful!"


def generate_text(
    prompt_text: str,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    min_new_tokens: int,
    seed: int,
    suppress_tokens_str: str,
):
    """Generate text continuation."""
    if not prompt_text:
        return "", "Please provide input text."
    
    # Set seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    
    # Parse suppress tokens
    suppress_tokens = None
    if suppress_tokens_str and suppress_tokens_str.strip():
        suppress_tokens = [int(t.strip()) for t in suppress_tokens_str.split(",") if t.strip()]
    
    # Generate text
    prompt = f"<|text_start|>{prompt_text}"
    output, prompt_length = run_inference_no_postprocessing(
        prompt, max_new_tokens, temperature, top_p, end_token_id=128001, min_new_tokens=min_new_tokens, suppress_tokens=suppress_tokens, # <|end_of_text|>
    )
    
    generated_text = tokenizer.decode(output[prompt_length:], skip_special_tokens=False)
    
    if not generated_text:
        return "", "No text generated."
    
    return generated_text, "Generation successful!"


def create_demo():
    """Create Gradio interface."""
    with gr.Blocks(title="Blueberry Audio Demo") as demo:
        gr.Markdown("# 🎙️ Blueberry Audio Model Demo")
        gr.Markdown("Explore various audio generation and processing tasks!")
        
        with gr.Tabs():
            # Tab 1: Zero-Shot TTS
            with gr.Tab("Zero-Shot TTS"):
                gr.Markdown("### Generate speech in any voice using just a short audio prompt!")
                
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("#### Input")
                        tts_prompt_text = gr.Textbox(
                            label="Prompt Text",
                            placeholder="The text that corresponds to the prompt audio...",
                            lines=3,
                        )
                        tts_prompt_audio = gr.Audio(
                            label="Prompt Audio",
                            type="filepath",
                            sources=["upload", "microphone"],
                        )
                        tts_target_text = gr.Textbox(
                            label="Target Text",
                            placeholder="The text you want to synthesize...",
                            lines=3,
                        )
                        
                        gr.Markdown("#### Generation Parameters")
                        with gr.Row():
                            tts_temperature = gr.Slider(
                                minimum=0.1,
                                maximum=2.0,
                                value=0.9,
                                step=0.1,
                                label="Temperature",
                                info="Higher values = more random"
                            )
                            tts_top_p = gr.Slider(
                                minimum=0.1,
                                maximum=1.0,
                                value=0.8,
                                step=0.05,
                                label="Top-p",
                                info="Nucleus sampling threshold"
                            )
                        
                        with gr.Row():
                            tts_max_new_tokens = gr.Slider(
                                minimum=100,
                                maximum=3000,
                                value=1500,
                                step=100,
                                label="Max New Tokens",
                                info="Maximum length of generated audio"
                            )
                            tts_min_new_tokens = gr.Slider(
                                minimum=0,
                                maximum=1000,
                                value=0,
                                step=50,
                                label="Min New Tokens",
                                info="Minimum length of generated audio"
                            )
                        
                        with gr.Row():
                            tts_seed = gr.Number(
                                value=42,
                                label="Random Seed",
                                precision=0,
                                info="For reproducibility"
                            )
                        
                        tts_generate_btn = gr.Button("🎵 Generate Speech", variant="primary", size="lg")
                    
                    with gr.Column():
                        gr.Markdown("#### Output")
                        tts_output_audio = gr.Audio(
                            label="Generated Speech",
                            type="numpy",
                        )
                        tts_status_text = gr.Textbox(
                            label="Status",
                            interactive=False,
                        )
                
                tts_generate_btn.click(
                    fn=generate_zero_shot_tts,
                    inputs=[
                        tts_prompt_text,
                        tts_prompt_audio,
                        tts_target_text,
                        tts_temperature,
                        tts_top_p,
                        tts_max_new_tokens,
                        tts_min_new_tokens,
                        tts_seed,
                    ],
                    outputs=[tts_output_audio, tts_status_text],
                )
            
            # Tab 2: Speech Continuation
            with gr.Tab("Speech Continuation"):
                gr.Markdown("### Continue speech from an audio prompt!")
                
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("#### Input")
                        cont_audio = gr.Audio(
                            label="Input Audio",
                            type="filepath",
                            sources=["upload", "microphone"],
                        )
                        cont_method = gr.Radio(
                            choices=["Method 1 (Direct continuation)", "Method 2 (Text-guided)"],
                            value="Method 1 (Direct continuation)",
                            label="Continuation Method",
                            info="Method 1: Direct audio continuation | Method 2: Text-guided continuation"
                        )
                        
                        gr.Markdown("#### Generation Parameters")
                        with gr.Row():
                            cont_temperature = gr.Slider(
                                minimum=0.1,
                                maximum=2.0,
                                value=1.0,
                                step=0.1,
                                label="Temperature"
                            )
                            cont_top_p = gr.Slider(
                                minimum=0.1,
                                maximum=1.0,
                                value=0.9,
                                step=0.05,
                                label="Top-p"
                            )
                        
                        with gr.Row():
                            cont_max_new_tokens = gr.Slider(
                                minimum=100,
                                maximum=3000,
                                value=500,
                                step=100,
                                label="Max New Tokens"
                            )
                            cont_min_new_tokens = gr.Slider(
                                minimum=0,
                                maximum=1000,
                                value=100,
                                step=50,
                                label="Min New Tokens",
                                info="Minimum length for speech continuation"
                            )
                        
                        with gr.Row():
                            cont_seed = gr.Number(
                                value=42,
                                label="Random Seed",
                                precision=0
                            )
                        
                        cont_suppress_tokens = gr.Textbox(
                            label="Suppress Tokens (comma-separated token IDs)",
                            placeholder="e.g., 128258, 128259",
                            value="",
                            info="Optional: Enter token IDs to prevent from being generated"
                        )
                        
                        cont_generate_btn = gr.Button("▶️ Continue Speech", variant="primary", size="lg")
                    
                    with gr.Column():
                        gr.Markdown("#### Output")
                        cont_output_audio = gr.Audio(
                            label="Continued Speech",
                            type="numpy",
                        )
                        cont_status_text = gr.Textbox(
                            label="Status",
                            lines=15,
                            interactive=False,
                            show_copy_button=True,
                        )
                
                cont_generate_btn.click(
                    fn=generate_speech_continuation,
                    inputs=[
                        cont_audio,
                        cont_method,
                        cont_temperature,
                        cont_top_p,
                        cont_max_new_tokens,
                        cont_min_new_tokens,
                        cont_seed,
                        cont_suppress_tokens,
                    ],
                    outputs=[cont_output_audio, cont_status_text],
                )
            
            # Tab 3: ASR (Automatic Speech Recognition)
            with gr.Tab("ASR (Transcription)"):
                gr.Markdown("### Transcribe audio to text!")
                
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("#### Input")
                        asr_audio = gr.Audio(
                            label="Input Audio",
                            type="filepath",
                            sources=["upload", "microphone"],
                        )
                        
                        gr.Markdown("#### Generation Parameters")
                        with gr.Row():
                            asr_temperature = gr.Slider(
                                minimum=0.1,
                                maximum=2.0,
                                value=1.0,
                                step=0.1,
                                label="Temperature"
                            )
                            asr_top_p = gr.Slider(
                                minimum=0.1,
                                maximum=1.0,
                                value=0.9,
                                step=0.05,
                                label="Top-p"
                            )
                        
                        with gr.Row():
                            asr_max_new_tokens = gr.Slider(
                                minimum=100,
                                maximum=3000,
                                value=1200,
                                step=100,
                                label="Max New Tokens"
                            )
                            asr_min_new_tokens = gr.Slider(
                                minimum=0,
                                maximum=1000,
                                value=0,
                                step=50,
                                label="Min New Tokens",
                                info="Minimum length of transcription"
                            )
                        
                        with gr.Row():
                            asr_seed = gr.Number(
                                value=42,
                                label="Random Seed",
                                precision=0
                            )
                        
                        asr_generate_btn = gr.Button("📝 Transcribe", variant="primary", size="lg")
                    
                    with gr.Column():
                        gr.Markdown("#### Output")
                        asr_output_text = gr.Textbox(
                            label="Transcription",
                            lines=5,
                            interactive=False,
                        )
                        asr_status_text = gr.Textbox(
                            label="Status",
                            interactive=False,
                        )
                
                asr_generate_btn.click(
                    fn=generate_transcription,
                    inputs=[
                        asr_audio,
                        asr_temperature,
                        asr_top_p,
                        asr_max_new_tokens,
                        asr_min_new_tokens,
                        asr_seed,
                    ],
                    outputs=[asr_output_text, asr_status_text],
                )
            
            # Tab 4: Text Generation
            with gr.Tab("Text Generation"):
                gr.Markdown("### Generate text continuation from a prompt!")
                
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("#### Input")
                        text_input = gr.Textbox(
                            label="Input Text",
                            placeholder="Enter your text prompt here...",
                            lines=5,
                        )
                        
                        gr.Markdown("#### Generation Parameters")
                        with gr.Row():
                            text_temperature = gr.Slider(
                                minimum=0.1,
                                maximum=2.0,
                                value=1.0,
                                step=0.1,
                                label="Temperature"
                            )
                            text_top_p = gr.Slider(
                                minimum=0.1,
                                maximum=1.0,
                                value=0.9,
                                step=0.05,
                                label="Top-p"
                            )
                        
                        with gr.Row():
                            text_max_new_tokens = gr.Slider(
                                minimum=10,
                                maximum=2000,
                                value=200,
                                step=10,
                                label="Max New Tokens"
                            )
                            text_min_new_tokens = gr.Slider(
                                minimum=0,
                                maximum=500,
                                value=0,
                                step=10,
                                label="Min New Tokens"
                            )
                        
                        with gr.Row():
                            text_seed = gr.Number(
                                value=42,
                                label="Random Seed",
                                precision=0
                            )
                        
                        text_suppress_tokens = gr.Textbox(
                            label="Suppress Tokens (comma-separated token IDs)",
                            placeholder="e.g., 128258, 128259",
                            value="",
                            info="Optional: Enter token IDs to prevent from being generated"
                        )
                        
                        text_generate_btn = gr.Button("✍️ Generate Text", variant="primary", size="lg")
                    
                    with gr.Column():
                        gr.Markdown("#### Output")
                        text_output = gr.Textbox(
                            label="Generated Text",
                            lines=10,
                            interactive=False,
                            show_copy_button=True,
                        )
                        text_status = gr.Textbox(
                            label="Status",
                            interactive=False,
                        )
                
                text_generate_btn.click(
                    fn=generate_text,
                    inputs=[
                        text_input,
                        text_temperature,
                        text_top_p,
                        text_max_new_tokens,
                        text_min_new_tokens,
                        text_seed,
                        text_suppress_tokens,
                    ],
                    outputs=[text_output, text_status],
                )
    
    return demo


def main():
    """Main function to launch the demo."""
    load_models()
    demo = create_demo()
    demo.launch(share=True, server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    main()

