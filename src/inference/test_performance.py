from __future__ import annotations

import io
import matplotlib.patches
import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib
import matplotlib.pyplot as plt

from src.utils import ColourConverter
# MAKE A FUNCTION TO PLOT L2 NORM FOR TEST DATA AND THE COLOR PALLETE

plt.switch_backend('AGG')

def plot_and_compare(x):
    # Find the L2 norm and plot
    l2_norm = []
    model_name = []
    for name, df in x.items():
        a = ColourConverter.hex_to_lab(list(df['actual_LAB_code'].values))
        b = df['predicted_LAB_code'].values
        dist = {}
        for element in range(len(a)):
            dist[element] = np.linalg.norm(a[element] - b[element])
        l2_norm.append(sum(dist.values()) / len(a))
        model_name.append(name)

    model_l2_norm_data = pd.DataFrame({
        'model_name': model_name,
        'l2_norm': l2_norm
    })

    fig = px.scatter(model_l2_norm_data, x='model_name', y='l2_norm')
    fig.show()


def draw_color_palletes(predicted_colour):

    # Draw color pallete for both true and predicted values to compare visually

    pred_hex_codes = ColourConverter.lab_to_hex(predicted_colour)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    square_1 = matplotlib.patches.Rectangle((0, 0), 200, 200, color=f"{pred_hex_codes}")
    ax.add_patch(square_1)
    plt.xlim([0, 200])
    plt.ylim([0, 200])

    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png')
    plt.close(fig)

    return img_buf
