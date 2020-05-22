from matplotlib import pyplot as plt
import numpy as np
from statistics import mean
from PIL import Image
from os.path import join


def show_images(images, titles=None, cols=10, figSize=(15, 15)):
    """
    Shows images with its labels. Expected PIL Image.
    """
    figure = plt.figure(figsize=figSize)
    num_of_images = len(images)
    rows = np.ceil(num_of_images / float(cols))
    for index in range(0, num_of_images):
        plt.subplot(rows, cols, index + 1)
        plt.axis('off')
        if titles is not None:
            plt.title(titles[index])
        plt.imshow(np.asarray(images[index]), cmap="gray")


def get_stats(images, name):
    widths = [img.size[0] for img in images]
    heights = [img.size[1] for img in images]

    max_width = max(widths)
    max_height = max(heights)

    min_width = min(widths)
    min_height = min(heights)

    avg_width = mean(widths)
    avg_height = mean(heights)

    print(f"Stats of {name}\n")
    print(f"Number of images: {len(images)}")
    print(f"Min width, Max width, Min Height, Max height: {min_width}, {max_width}, {min_height}, {max_height}")
    print(f"Average width and height {avg_width}, {avg_height}")


def paste_fg_on_bg(bg, fg, fg_mask, randomPasteCount):
    """
    Pastes the fore ground image to background and creates a mask
    """
    overlayed_images = []
    masked_images = []
    for i in range(randomPasteCount):
        bg_width, bg_height = bg.size
        fg_width, fg_height = fg.size
        offset_x = np.random.randint(0, bg_width - fg_width)
        offset_y = np.random.randint(0, bg_height - fg_height)
        bg_copy = bg.copy()
        bg_copy.paste(fg, (offset_x, offset_y), fg)
        overlayed_images.append(bg_copy)

        bg_mask = Image.new('1', bg.size)
        bg_mask.paste(fg_mask, (offset_x, offset_y), fg_mask)
        masked_images.append(bg_mask)

    return overlayed_images, masked_images


def load_image(folder, file, channel):
    with open(join(folder, file), 'rb') as f:
        x = Image.open(f)
        x = x.convert(channel)
    return x
