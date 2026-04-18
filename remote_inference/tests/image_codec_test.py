# to run the tests: pytest remote_inference/tests/image_codec_test.py

import numpy as np
import torch
from remote_inference.image_codec import encode_image, decode_image_to_tensor


def test_image_codec_round_trip():
    """Encoding and then decoding an image should yield a tensor with the same shape and similar content."""
    height, width, channels = 480, 640, 3
    # Create a random image (HWC, uint8)
    img = np.random.randint(0, 256, size=(height, width, channels), dtype=np.uint8)

    # encode to base64 string using our encoder function
    b64_str = encode_image(img)

    # Decode back to tensor
    img_tensor = decode_image_to_tensor(b64_str)

    # Check shape (should be 1 x C x H x W)
    assert img_tensor.shape[0] == 1, f"Batch dimension should be 1, got {img_tensor.shape[0]}"
    assert img_tensor.shape[1] == 3, f"Channel dimension should be 3, got {img_tensor.shape[1]}"
    assert img_tensor.shape[2] <= 224 and img_tensor.shape[3] <= 224, f"Height and width should be <= 224, got {img_tensor.shape[2:]}"
    assert img_tensor.shape[2] / img_tensor.shape[3] == height / width, f"Aspect ratio should be preserved, got {img_tensor.shape[2] / img_tensor.shape[3]} vs {height / width}"

    # Check value range
    assert torch.all((img_tensor >= 0) & (img_tensor <= 1)), "Tensor values should be in [0, 1]"