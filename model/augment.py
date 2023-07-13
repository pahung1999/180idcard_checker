import albumentations as A
import numpy as np
import cv2
import random
from PIL import Image
from typing import Tuple, Callable
from functools import partial
from dataclasses import dataclass

def expanded_image(image_to_paste: Image, background_image: np.ndarray, x_expand_rate: float = 0.1, y_expand_rate: float = 0.1) -> Image:
    """
    Expand the size of an image and paste it onto a background image. Image_to_paste can be a transparent background image.

    Args:
        image_to_paste (PIL.Image.Image): The image to be pasted onto the background image.
        background_image (np.ndarray): The background image represented as a NumPy array.
        x_expand_rate (float): The rate of horizontal expansion. Default is 0.1.
        y_expand_rate (float): The rate of vertical expansion. Default is 0.1.

    Returns:
        PIL.Image.Image: The final image with the expanded image pasted onto the background image.

    Raises:
        ValueError: If the input image or background image is not valid.

    """

    w, h = image_to_paste.size

    new_w = int(w*(1+random.uniform(0, x_expand_rate)))
    new_h = int(h*(1+random.uniform(0, x_expand_rate)))

    parse_x = random.randint(0, new_w-w)
    parse_y = random.randint(0, new_h-h)

    background_image = cv2.resize(background_image, (new_w, new_h))

    background_image = Image.fromarray(background_image)

    # Create a new image with the same size as the background image
    new_image = Image.new('RGBA', background_image.size)

    # Paste the transparent image onto the new image using the alpha channel
    new_image.paste(image_to_paste, (parse_x, parse_y), mask=image_to_paste)

    # Paste the new image onto the background image
    final_image = Image.alpha_composite(background_image.convert('RGBA'), new_image)

    # Save the modified image
    return final_image

def rotate_img(image: Image, degree_rotate: int):
    """
    Rotate an image by a specified angle. And the result image is a transparent background image.

    Args:
        image (PIL.Image.Image): The image to be rotated.
        degree_rotate (int): The angle in degrees by which to rotate the image.

    Returns:
        PIL.Image.Image: The rotated image.

    """

    # Rotate the image with expand=True
    rotated_image = image.rotate(degree_rotate, expand=True)

    # Create a transparent image of the same size
    transparent_image = Image.new('RGBA', rotated_image.size, (0, 0, 0, 0))

    # Paste the rotated image onto the transparent image
    transparent_image.paste(rotated_image, (0, 0), mask=rotated_image)

    return transparent_image

#===========================================================================
def CoarseDropout(*a, **kw):
    orig = A.CoarseDropout(*a, **kw)
    patched = A.Lambda(name="CoarseDropoutImg", image=orig.apply)
    return patched

#===========================================================================
@dataclass
class ShaderBasicLight:
    """
    Dataclass representing parameters for the ShaderBasicLight shader.

    Args:
        min_deg_x (int): The minimum value for deg_x. Default is 0.
        min_deg_y (int): The minimum value for deg_y. Default is 0.
        max_deg_x (int): The maximum value for deg_x. Default is 3.
        max_deg_y (int): The maximum value for deg_y. Default is 3.
        red (Tuple): The range of values for the red channel. Default is (0, 255).
        green (Tuple): The range of values for the green channel. Default is (0, 255).
        blue (Tuple): The range of values for the blue channel. Default is (0, 255).

    """
    min_deg_x: int = 0
    min_deg_y: int = 0
    max_deg_x: int = 3
    max_deg_y: int = 3
    red: Tuple = (0, 255)
    green: Tuple = (0, 255)
    blue: Tuple = (0, 255)

    def __post_init__(self):
        self.deg_x = random.randint(self.min_deg_x, self.max_deg_x)
        self.deg_y = random.randint(self.min_deg_y, self.max_deg_y)
        self.flip_x = random.choice((True, False))
        self.flip_y = random.choice((True, False))
        self.r = random.choice(self.red)
        self.g = random.choice(self.green)
        self.b = random.choice(self.blue)

    def __call__(self, x, y, w, h):
        deg_x = self.deg_x
        deg_y = self.deg_y
        px = x**deg_x / w**deg_x
        py = y**deg_y / h**deg_y
        if self.flip_x:
            px = 1 - px
        if self.flip_y:
            py = 1 - py
        r = self.r * (1 - px * py)
        g = self.g * (1 - px * py)
        b = self.b * (1 - px * py)
        return (int(r), int(g), int(b))

def fake_light(
    image: np.ndarray, shader_factory: Callable, tile_size: int, alpha: int, **opts
):
    """
    Apply a fake light effect to an image using a shader function.

    Args:
        image (np.ndarray): The input image represented as a NumPy array.
        shader_factory (Callable): The shader function used to generate the lighting effect.
        tile_size (int): The size of each tile used in the effect.
        alpha (int): The alpha value of the effect.
        **opts: Additional options to be passed to the shader function.

    Returns:
        np.ndarray: The image with the fake light effect applied.

    """
    # Prepare
    H, W = image.shape[:2]
    shader_fn = shader_factory()
    if isinstance(tile_size, tuple):
        tile_size = random.randint(*tile_size)
    if isinstance(alpha, tuple):
        alpha = random.uniform(*alpha)

    # Convert image to RGB if it's gray
    if image.ndim == 2:
        image = np.stack([image, image, image], axis=-1)

    # Tiles
    canvas = np.zeros((H, W, 3))
    for x in range(0, W, tile_size):
        for y in range(0, H, tile_size):
            br = shader_fn(x, y, W, H)
            x2 = min(x + tile_size, W)
            y2 = min(y + tile_size, H)
            cv2.rectangle(canvas, (x, y), (x2, y2), br, -1)

    # alpha composite
    image = (image * (1 - alpha) + canvas * alpha).round().astype(image.dtype)
    return image


def FakeLight(tile_size=(20, 50), alpha=(0.2, 0.6), **kw):
    """
    Augmentation function for applying the FakeLight effect.

    Args:
        tile_size (Tuple or int): The range of tile sizes to be used in the effect. Default is (20, 50).
        alpha (Tuple or float): The range of alpha values to be used in the effect. Default is (0.2, 0.6).
        **kw: Additional keyword arguments to be passed to the augmentation function.

    Returns:
        Callable: The augmentation function for applying the FakeLight effect.

    """
    fn = partial(
        fake_light, shader_factory=ShaderBasicLight, tile_size=tile_size, alpha=alpha
    )
    return A.Lambda(image=fn, **kw)

#=====================================================================================


def get_transform(prob = 0.5):
    """
    Get a composition of image transformations.

    Args:
        prob (float): The probability of applying each transformation. Default is 0.5.

    Returns:
        albumentations.Compose: The composition of image transformations.

    """
    transformations = [
            # Shader based
            A.OneOf([FakeLight()], p=prob),
            # Color effects
            A.OneOf(
                [
                    A.RandomBrightnessContrast(),
                    A.ToGray(),
                    A.Equalize(),
                    A.ChannelDropout(),
                    A.ChannelShuffle(),
                    A.FancyPCA(),
                    A.ToSepia(),
                    A.ColorJitter(),
                    A.RandomGamma(),
                    A.RGBShift(),
                ],
                p=prob,
            ),
            # Degrade
            A.OneOf(
                [
                    A.PixelDropout(),
                    A.OneOf(
                        [
                            CoarseDropout(fill_value=color, max_width=32, max_height=32)
                            for color in range(0, 255)
                        ]
                    ),
                    A.Downscale(interpolation=cv2.INTER_LINEAR),
                    A.Blur(),
                    A.MedianBlur(),
                    A.Posterize(),
                    A.Spatter(),
                    A.ISONoise(),
                    A.MultiplicativeNoise(),
                    A.ImageCompression(quality_lower=50),
                    A.GaussNoise(),
                ],
                p=prob,
            ),
            A.RandomBrightnessContrast(p=prob),
            A.GaussNoise(p=prob),
            A.Blur(p=prob),
            A.PixelDropout(dropout_prob = random.uniform(0.1, 0.5), p = prob),
        ]
    return A.Compose(transformations)