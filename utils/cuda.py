import gc 
import torch

def claim_memory():
    gc.collect()
    torch.cuda.empty_cache()


# https://github.com/huggingface/transformers/issues/28188
def supports_flash_attention():
    """Check if a GPU supports FlashAttention."""

    print("cuda is available", torch.cuda.is_available())
    
    DEVICE = 0 if torch.cuda.is_available() else "cpu"
    if DEVICE == "cpu":
        return False
    
    major, minor = torch.cuda.get_device_capability(DEVICE)
    
    # Check if the GPU architecture is Ampere (SM 8.x) or newer (SM 9.0)
    is_sm8x = major == 8 and minor >= 0
    is_sm90 = major == 9 and minor == 0
    
    return is_sm8x or is_sm90