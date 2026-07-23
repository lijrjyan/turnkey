from __future__ import annotations

import importlib
from typing import Any

INTERNVL_IMAGE_SIZE = 448
INTERNVL_MAX_IMAGE_TILES = 12


def _internvl_query_with_image_tokens(prompt: str, *, model: Any, tokenizer: Any, num_patches: int) -> str:
    img_context_token = "<IMG_CONTEXT>"
    model.img_context_token_id = tokenizer.convert_tokens_to_ids(img_context_token)
    num_image_tokens = int(getattr(model, "num_image_token", 0)) * int(num_patches)
    image_tokens = "<img>" + (img_context_token * num_image_tokens) + "</img>"
    if "<image>" in prompt:
        return prompt.replace("<image>", image_tokens, 1)
    return f"{image_tokens}\n{prompt}"


def _internvl_pixel_values_from_image_path(path: str, *, device: Any, dtype: Any | None) -> Any:
    torch = importlib.import_module("torch")
    image_mod = importlib.import_module("PIL.Image")
    transforms = importlib.import_module("torchvision.transforms")
    interpolation = importlib.import_module("torchvision.transforms.functional").InterpolationMode

    image = _load_image_from_path_or_url(path, image_mod=image_mod)
    tiles = _internvl_dynamic_preprocess(
        image,
        image_size=INTERNVL_IMAGE_SIZE,
        max_num=INTERNVL_MAX_IMAGE_TILES,
    )
    transform = transforms.Compose(
        [
            transforms.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            transforms.Resize((INTERNVL_IMAGE_SIZE, INTERNVL_IMAGE_SIZE), interpolation=interpolation.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    pixel_values = torch.stack([transform(tile) for tile in tiles])
    if dtype is not None:
        pixel_values = pixel_values.to(dtype=dtype)
    return pixel_values.to(device=device)


def _load_image_from_path_or_url(path: str, *, image_mod: Any) -> Any:
    if path.startswith(("http://", "https://")):
        requests = importlib.import_module("requests")
        BytesIO = importlib.import_module("io").BytesIO
        response = requests.get(path)
        response.raise_for_status()
        return image_mod.open(BytesIO(response.content)).convert("RGB")
    return image_mod.open(path).convert("RGB")


def _internvl_dynamic_preprocess(
    image: Any,
    *,
    min_num: int = 1,
    max_num: int = 12,
    image_size: int = 448,
    use_thumbnail: bool = True,
) -> list[Any]:
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = sorted(
        {
            (i, j)
            for n in range(min_num, max_num + 1)
            for i in range(1, n + 1)
            for j in range(1, n + 1)
            if i * j <= max_num and i * j >= min_num
        },
        key=lambda x: x[0] * x[1],
    )
    target_aspect_ratio = _internvl_find_closest_aspect_ratio(
        aspect_ratio,
        target_ratios,
        orig_width,
        orig_height,
        image_size,
    )
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    resized = image.resize((target_width, target_height))
    processed = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        processed.append(resized.crop(box))
    if use_thumbnail and len(processed) != 1:
        processed.append(image.resize((image_size, image_size)))
    return processed


def _internvl_find_closest_aspect_ratio(
    aspect_ratio: float,
    target_ratios: list[tuple[int, int]],
    width: int,
    height: int,
    image_size: int,
) -> tuple[int, int]:
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio
