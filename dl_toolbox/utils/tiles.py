from itertools import product
from rasterio.windows import Window
import numpy as np
import torch

def gaussian_weight(window_size):
    center = window_size // 2
    y, x = np.ogrid[:window_size, :window_size]
    distance = np.sqrt((x - center)**2 + (y - center)**2)
    sigma = window_size / 2
    weights = np.exp(-distance**2 / (2 * sigma**2))
    return weights

def initialize_count_map(height, width):
    return torch.zeros((height, width), dtype=torch.float32)

def increment_count_map(count_map, get_tiles, gaussian_weight, size=224, step=128):
    for col_off, row_off, width, height in get_tiles(count_map.shape[1], count_map.shape[0], size=size, step=step):
        print(col_off, row_off, width, height)
        count_map[row_off:row_off + height, col_off:col_off + width] += gaussian_weight(size)


def get_tiles(nols, nrows, size, size2=None, step=None, step2=None, col_offset=0, row_offset=0):
    
    if step is None: step = size
    if size2 is None: size2 = size
    if step2 is None: step2 = step

    max_col_offset = int(np.ceil((nols-size)/step))
    # Remove all offsets such that offset+size > nols and add one offset to
    # reach nols
    col_offsets = list(range(col_offset, col_offset + nols, step))[:max_col_offset+1]
    col_offsets[max_col_offset] = col_offset + nols - size

    max_row_offset = int(np.ceil((nrows-size2)/step2))
    # Remove all offsets such that offset+size > nols and add one offset to
    # reach nols
    row_offsets = list(range(row_offset, row_offset + nrows, step2))[:max_row_offset+1]
    row_offsets[max_row_offset] = row_offset + nrows - size2

    offsets = product(col_offsets, row_offsets)
    big_window = Window(col_off=col_offset, row_off=row_offset, width=nols, height=nrows)
    for col_off, row_off in offsets:
        window = Window(col_off=col_off, row_off=row_off, width=size,
                        height=size2).intersection(big_window)
        yield window


def get_tiles_numpy(nols, nrows, size, size2=None, step=None, step2=None, col_offset=0, row_offset=0):
    if step is None: step = size
    if size2 is None: size2 = size
    if step2 is None: step2 = step

    max_col_offset = int(np.ceil((nols-size)/step))
    col_offsets = list(range(col_offset, col_offset + nols, step))[:max_col_offset+1]
    col_offsets[max_col_offset] = col_offset + nols - size

    max_row_offset = int(np.ceil((nrows-size2)/step2))
    row_offsets = list(range(row_offset, row_offset + nrows, step2))[:max_row_offset+1]
    row_offsets[max_row_offset] = row_offset + nrows - size2

    offsets = product(col_offsets, row_offsets)
    for col_off, row_off in offsets:
        yield (col_off, row_off, size, size2)
        
def extract_tiles(image, tile_size, step=None):
    nrows, ncols, channels = image.shape
    tiles = []
    for col_off, row_off, width, height in get_tiles_numpy(ncols, nrows, tile_size[1], tile_size[0], step):
        print(col_off, row_off, width, height)
        tile = image[row_off:row_off + height, col_off:col_off + width, :]
        tiles.append(tile)
    return torch.stack(tiles)


def main():

    for tile in get_tiles(1000,1500,412, size2=397, step=400, col_offset=10):
        print(tile)


if __name__ == "__main__":

    main()
