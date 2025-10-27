import logging
from typing import List, Optional, Union

import librosa
import numpy as np
import torch
from transformers import MimiModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
UNICODE_OFFSET: int = 0xE000
NUM_CODEBOOKS: int = 8
CODEBOOK_SIZE: int = 2048


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

