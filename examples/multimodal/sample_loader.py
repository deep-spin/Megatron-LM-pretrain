# Sample loader for text-images & text-only datasets
# To use, copy to the dataset folder and/or edit the sample_loader.py
# import torch

def sample_loader(raw: dict) -> dict:    # Note: Images are already decoded to tensors
    # TODO: Set the correct values for all (required) fields
    images = raw["png.jpg"] if "png.jpg" in raw else []
    images = [images] if not isinstance(images, list) else images
    return dict(
        images=images,  # expected type: typing.List[torch.Tensor]
        texts=raw["png.json"],  # expected type: typing.List[str] (not true, it actually contains conversation)
        similarity_matrix=None,  # expected type: typing.Optional[torch.Tensor], default: None
        matched_text_indices=None,  # expected type: typing.Optional[typing.List[int]], default: None
    )

def part_filter(part: str) -> bool:
    # TODO: Filter for parts required by the sample_loader
    # E.g. if your dataset contains jpeg, txt and json, but you won't use json,
    # remove it from the list, such that it is not decoded. If you need all, keep as is
    return part in ('png.jpg', 'png.json')