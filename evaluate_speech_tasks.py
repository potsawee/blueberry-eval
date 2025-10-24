import json
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import librosa
import numpy as np
import soundfile as sf
import torch
import torchaudio.transforms as T
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, MimiModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
UNICODE_OFFSET: int = 0xE000
NUM_CODEBOOKS: int = 8
CODEBOOK_SIZE: int = 2048
MIMI_MODEL_ID: str = "kyutai/mimi"
MARIN_MODEL_NAME: str = "WillHeld/blueberry"
SOURCE_SAMPLE_RATE: int = 16000
TARGET_SAMPLE_RATE: int = 24000
SPEECH_CONTINUATION_DURATION_SEC: int = 3
MAX_NEW_TOKENS: int = 1200
GENERATION_TEMPERATURE: float = 1.0
GENERATION_TOP_P: float = 0.9


def codes_to_chars(
    codes: Union[List[List[int]], np.ndarray, torch.Tensor],
    codebook_size: int,
    copy_before_conversion: bool = True,
    unicode_offset: int = UNICODE_OFFSET,
) -> str:
    if isinstance(codes, list):
        codes = np.array(codes)
        copy_before_conversion = False
    elif isinstance(codes, torch.Tensor):
        codes = codes.cpu().numpy()
    if len(codes.shape) != 2:
        raise ValueError("codes must be a 2D array of shape (num_codebooks, seq_length).")
    if copy_before_conversion:
        codes = codes.copy()
    for i in range(codes.shape[0]):
        codes[i] += unicode_offset + i * codebook_size
    codes = codes.T.reshape(-1)
    chars = "".join([chr(c) for c in codes])
    return chars


def chars_to_codes(
    chars: str,
    num_codebooks: int,
    codebook_size: int,
    return_tensors: Optional[str] = None,
    unicode_offset: int = UNICODE_OFFSET,
) -> Union[List[List[int]], np.ndarray, torch.Tensor]:
    codes = np.array([ord(c) for c in chars])
    codes = codes.reshape(-1, num_codebooks).T
    for i in range(codes.shape[0]):
        codes[i] -= unicode_offset + i * codebook_size
    if return_tensors is None:
        codes = codes.tolist()
    elif return_tensors == "pt":
        codes = torch.tensor(codes)
    return codes


def audio_to_str(audio_numpy: np.ndarray, mimi_model: MimiModel, device: str) -> str:
    audio_tensor = torch.tensor(audio_numpy).to(device).unsqueeze(0)
    if len(audio_tensor.shape) == 2:
        audio_tensor = audio_tensor.unsqueeze(1)
    
    with torch.no_grad():
        audio_codes = mimi_model.encode(audio_tensor)
    
    codes = audio_codes[0][0].cpu()
    codes = codes[:NUM_CODEBOOKS, :]
    audio_str = codes_to_chars(codes, codebook_size=CODEBOOK_SIZE)
    return audio_str


def str_to_audio(audio_str: str, mimi_model: MimiModel, device: str) -> np.ndarray:
    codes = chars_to_codes(
        audio_str, num_codebooks=NUM_CODEBOOKS, codebook_size=CODEBOOK_SIZE, return_tensors="pt"
    )
    codes = codes.to(device).unsqueeze(0)
    
    with torch.no_grad():
        audio_decoded = mimi_model.decode(codes).audio_values[0]
    
    return audio_decoded.cpu().numpy()


def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return audio
    return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)


def load_librispeech_data(json_path: str) -> List[dict]:
    with open(json_path, "r") as f:
        data = json.load(f)
    return data

def extract_text_and_audio_segments_v1(
    generated_text: str,
) -> Tuple[List[str], List[str]]:
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
            # Handle incomplete audio segment (generation stopped before <|audio_end|>)
            # Extract everything after <|audio_start|> until end of string
            audio_content = part.strip()
            if audio_content:
                audio_segments.append(audio_content)
    
    return text_segments, audio_segments


def run_inference(
    prompt: str, 
    model: AutoModelForCausalLM, 
    tokenizer: AutoTokenizer, 
    max_new_tokens: int = 1000,
):
    # print(f"Prompt: '{prompt}'")
    # by default, tokenizer prepends the <|begin_of_text|> token
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=GENERATION_TEMPERATURE,
            top_p=GENERATION_TOP_P,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
    # print(f"Generated: '{generated_text}'")
    # print("-" * 80 + "\n")
    return generated_text

def task_speech_continuation_v1(
    sample: dict,
    sample_idx: int,
    mimi_model: MimiModel,
    marin_model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    output_dir: Path,
    device: str,
) -> str:
    logger.info(f"Processing speech continuation for sample {sample_idx}")
    
    # Load and resample audio
    audio, sr = librosa.load(sample["file_path"], sr=None)
    audio_resampled = resample_audio(audio, orig_sr=sr, target_sr=TARGET_SAMPLE_RATE)
    
    # Truncate to first 3 seconds
    truncated_length = SPEECH_CONTINUATION_DURATION_SEC * TARGET_SAMPLE_RATE
    audio_truncated = audio_resampled[:truncated_length]
    
    
    # Convert to Mimi audio string
    audio_str = audio_to_str(audio_truncated, mimi_model, device)

    # Generate continuation
    prompt = f"<|audio_start|>{audio_str}"
    generated_text = run_inference(
        prompt, marin_model, tokenizer, MAX_NEW_TOKENS
    )
    # Extract and concatenate all audio segments (handles interleaved audio/text)
    _, audios = extract_text_and_audio_segments_v1(generated_text)
    audio_continuation_str = "".join(audios)
    N = len(audio_continuation_str)
    audio_continuation_str = audio_continuation_str[:(N//8)*8]

    with open(output_dir / "speech_continuation_v1.txt", "w") as f:
        f.write(generated_text)

    if audio_continuation_str:
        logger.info(f"Extracted {len(audio_continuation_str)} audio characters from continuation")
        audio_numpy = str_to_audio(audio_continuation_str, mimi_model, device)
        output_path = output_dir / "speech_continuation_v1.wav"
        sf.write(output_path, audio_numpy.T, TARGET_SAMPLE_RATE)
        logger.info(f"Saved speech continuation v1 to {output_path}")
    else:
        logger.warning("No audio content found in speech continuation v1 generation")
    return generated_text

def task_speech_continuation_v2(
    sample: dict,
    sample_idx: int,
    mimi_model: MimiModel,
    marin_model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    output_dir: Path,
    device: str,
) -> str:
    logger.info(f"Processing speech continuation for sample {sample_idx}")
    
    # Load and resample audio
    audio, sr = librosa.load(sample["file_path"], sr=None)
    audio_resampled = resample_audio(audio, orig_sr=sr, target_sr=TARGET_SAMPLE_RATE)
    
    # Truncate to first 3 seconds
    truncated_length = SPEECH_CONTINUATION_DURATION_SEC * TARGET_SAMPLE_RATE
    audio_truncated = audio_resampled[:truncated_length]
    
    
    # Convert to Mimi audio string
    audio_str = audio_to_str(audio_truncated, mimi_model, device)
    
    # Generate continuation
    prompt = f"<|audio_start|>{audio_str}<|audio_end|><|text_start|>"
    generated_text = run_inference(
        prompt, marin_model, tokenizer, MAX_NEW_TOKENS
    )
    # Extract and concatenate all audio segments (handles interleaved audio/text)
    _, audios = extract_text_and_audio_segments_v1(generated_text)
    audio_continuation_str = "".join(audios)
    N = len(audio_continuation_str)
    audio_continuation_str = audio_continuation_str[:(N//8)*8]

    with open(output_dir / "speech_continuation_v2.txt", "w") as f:
        f.write(generated_text)

    if audio_continuation_str:
        logger.info(f"Extracted {len(audio_continuation_str)} audio characters from continuation")
        audio_numpy = str_to_audio(audio_continuation_str, mimi_model, device)
        output_path = output_dir / "speech_continuation_v2.wav"
        sf.write(output_path, audio_numpy.T, TARGET_SAMPLE_RATE)
        logger.info(f"Saved speech continuation v2 to {output_path}")
    else:
        logger.warning("No audio content found in speech continuation v2 generation")
    
    return generated_text


def task_asr(
    sample: dict,
    sample_idx: int,
    mimi_model: MimiModel,
    marin_model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    output_dir: Path,
    device: str,
) -> str:
    logger.info(f"Processing ASR for sample {sample_idx}")
    
    # Load and resample audio
    audio, sr = librosa.load(sample["file_path"], sr=None)
    audio_resampled = resample_audio(audio, orig_sr=sr, target_sr=TARGET_SAMPLE_RATE)
    
    # Convert to Mimi audio string
    audio_str = audio_to_str(audio_resampled, mimi_model, device)
    
    # Generate transcription
    prompt = f"<|audio_start|>{audio_str}<|audio_end|><|text_start|>"
    generated_text = run_inference(
        prompt, marin_model, tokenizer, MAX_NEW_TOKENS
    )
    
    # Extract transcribed text
    texts, _ = extract_text_and_audio_segments_v1(generated_text)
    transcribed_text = texts[0]
    
    output_path = output_dir / "asr.txt"
    with open(output_path, "w") as f:
        f.write(transcribed_text)
    logger.info(f"Saved ASR result to {output_path}")
    
    return generated_text


def task_zero_shot_tts(
    sample: dict,
    sample_idx: int,
    mimi_model: MimiModel,
    marin_model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    output_dir: Path,
    device: str,
) -> str:
    logger.info(f"Processing zero-shot TTS for sample {sample_idx}")
    
    # Load and resample audio
    audio, sr = librosa.load(sample["file_path"], sr=None)
    audio_resampled = resample_audio(audio, orig_sr=sr, target_sr=TARGET_SAMPLE_RATE)
    
    # Convert to Mimi audio string
    audio_str = audio_to_str(audio_resampled, mimi_model, device)
    
    # Get transcript and zero-shot text
    transcript = sample["transcript"].lower()
    zero_shot_text = sample.get("zero_shot_tts", sample.get("zero_shot_text", ""))
    
    # Generate zero-shot TTS
    prompt = (
        f"<|text_start|>{transcript}<|text_end|>"
        f"<|audio_start|>{audio_str}<|audio_end|>"
        f"<|text_start|>{zero_shot_text}<|text_end|>"
        f"<|audio_start|>"
    )
    generated_text = run_inference(
        prompt, marin_model, tokenizer, MAX_NEW_TOKENS
    )
    
    # Extract the last audio segment (the generated zero-shot TTS)
    texts, audios = extract_text_and_audio_segments_v1(generated_text)
    audio_generated_str = audios[1] # audios[1]
    N = len(audio_generated_str)
    audio_generated_str = audio_generated_str[:(N//8)*8]
    
    if audio_generated_str:
        audio_numpy = str_to_audio(audio_generated_str, mimi_model, device)
        output_path = output_dir / "zero_shot_tts.wav"
        sf.write(output_path, audio_numpy.T, TARGET_SAMPLE_RATE)
        logger.info(f"Saved zero-shot TTS to {output_path}")
    else:
        logger.warning("No audio content found in zero-shot TTS generation")
    
    return generated_text


def main() -> None:
    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Create output directories
    output_dir = Path("generated_outputs")
    output_dir.mkdir(exist_ok=True)
    
    # Load Mimi model
    logger.info(f"Loading Mimi model: {MIMI_MODEL_ID}")
    mimi_model = MimiModel.from_pretrained(MIMI_MODEL_ID)
    mimi_model = mimi_model.to(device)
    
    # Load Marin audio model
    logger.info(f"Loading Marin model: {MARIN_MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MARIN_MODEL_NAME)
    marin_model = AutoModelForCausalLM.from_pretrained(
        MARIN_MODEL_NAME, torch_dtype=torch.float32, device_map="auto"
    )
    marin_model.eval()
    logger.info(f"Number of parameters: {sum(p.numel() for p in marin_model.parameters())}")
    
    # Load dataset
    data_path = "data/librispeech_test_clean_sample_100_with_zero_shot.json"
    logger.info(f"Loading dataset from {data_path}")
    dataset = load_librispeech_data(data_path)
    logger.info(f"Loaded {len(dataset)} samples")
    
    # Process each sample
    for idx, sample in enumerate(tqdm(dataset, desc="Processing samples")):
        logger.info(f"\n{'=' * 80}")
        logger.info(f"Processing sample {idx}/{len(dataset)}")
        
        # Extract file ID from file path (e.g., "2830-3979-0010" from "path/to/2830-3979-0010.flac")
        file_id = Path(sample["file_path"]).stem
        
        # Create subdirectory for this sample using file ID
        sample_output_dir = output_dir / file_id
        sample_output_dir.mkdir(exist_ok=True)
        logger.info(f"Output directory: {sample_output_dir}")
        
        # Copy original audio
        audio, sr = librosa.load(sample["file_path"], sr=None)
        audio_resampled = resample_audio(audio, orig_sr=sr, target_sr=TARGET_SAMPLE_RATE)
        original_audio_path = sample_output_dir / "original_audio.wav"
        sf.write(original_audio_path, audio_resampled.T, TARGET_SAMPLE_RATE)
        logger.info(f"Saved original audio to {original_audio_path}")
        
        # Run all tasks and collect raw generated outputs
        generation_outputs: Dict[str, str] = {}
        
        # Task 1a: Speech Continuation v1
        speech_continuation_v1_output = task_speech_continuation_v1(
            sample, idx, mimi_model, marin_model, tokenizer, sample_output_dir, device
        )
        generation_outputs["speech_continuation_v1"] = speech_continuation_v1_output
        
        # Task 1b: Speech Continuation v2
        speech_continuation_v2_output = task_speech_continuation_v2(
            sample, idx, mimi_model, marin_model, tokenizer, sample_output_dir, device
        )
        generation_outputs["speech_continuation_v2"] = speech_continuation_v2_output
        
        # Task 2: ASR
        asr_output = task_asr(sample, idx, mimi_model, marin_model, tokenizer, sample_output_dir, device)
        generation_outputs["asr"] = asr_output
        
        # Task 3: Zero-shot TTS
        zero_shot_tts_output = task_zero_shot_tts(
            sample, idx, mimi_model, marin_model, tokenizer, sample_output_dir, device
        )
        generation_outputs["zero_shot_tts"] = zero_shot_tts_output
        
        # Save all raw generation outputs to JSON
        generation_json_path = sample_output_dir / "generation.json"
        with open(generation_json_path, "w") as f:
            json.dump(generation_outputs, f, indent=2)
        logger.info(f"Saved generation outputs to {generation_json_path}")
    
    logger.info(f"\n{'=' * 80}")
    logger.info(f"Evaluation complete! Results saved to {output_dir}")


if __name__ == "__main__":
    main()

