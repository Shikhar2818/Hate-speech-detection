import numpy as np
from PIL import Image

from experiments.train_meme_pilot import grouped_split
from hatexai.models.meme_pilot import occluded_images


def test_meme_templates_stay_in_one_split():
    templates = np.repeat(np.arange(20), 2)
    labels = np.tile([0, 1], 20)
    splits = grouped_split(labels, templates)
    groups = [set(templates[indices]) for indices in splits.values()]
    assert sum(len(group) for group in groups) == len(set(templates))
    assert set.union(*groups) == set(templates)


def test_meme_occlusion_preserves_input_and_handles_small_images():
    image = Image.new("RGB", (9, 9))
    masks = list(occluded_images(image))
    assert len(masks) == 9
    assert masks[0].getpixel((0, 0)) == (128, 128, 128)
    assert masks[0].getpixel((8, 8)) == image.getpixel((8, 8))
    assert image.getpixel((0, 0)) == (0, 0, 0)
    assert len(list(occluded_images(Image.new("RGB", (1, 1))))) == 9
