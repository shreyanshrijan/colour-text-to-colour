from __future__ import annotations

from skimage import color
from functools import wraps
import numpy as np

# TODO - in principle we could do input validation for each method.  Might be
# a good project when you need an easy project for the afternoon.


class ColourConverter():

    def elementwise(func):
        @wraps(func)
        def wrapper(input):
            if isinstance(input, list):
                output = []
                for input_i in input:
                    output.append(wrapper(input_i))
            else:
                output = func(input)
            return output

        return wrapper

    @staticmethod
    @elementwise
    def hex_to_rgb(h):
        return tuple(int(h[i:i+2], base=16) for i in (1, 3, 5))

    @staticmethod
    @elementwise
    def hex_to_lab(h):
        return color.rgb2lab(
            np.array(ColourConverter.hex_to_rgb(h))[None, None, :]/255
        ).flatten()

    @staticmethod
    @elementwise
    def lab_to_rgb(lab):
        return tuple(
                np.around(
                    color.lab2rgb(
                        lab[None, None, :]
                    ).squeeze()*255
                ).astype(int)
            )

    @staticmethod
    @elementwise
    def rgb_to_hex(rgb):
        return '#{:02x}{:02x}{:02x}'.format(*rgb)

    @staticmethod
    @elementwise
    def rgb_to_lab(rgb):
        return ColourConverter.hex_to_lab(ColourConverter.rgb_to_hex(rgb))

    @staticmethod
    @elementwise
    def lab_to_hex(lab):
        return ColourConverter.rgb_to_hex(ColourConverter.lab_to_rgb(lab))

    @staticmethod
    def average_hex_values_from_pandas_groupby(h):
        h = h.tolist()
        lab = ColourConverter.hex_to_lab(h)
        return ColourConverter.lab_to_hex(np.mean(lab, axis=0))
