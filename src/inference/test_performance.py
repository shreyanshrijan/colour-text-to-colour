from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns

from src.utils import ColourConverter
# MAKE A FUNCTION TO PLOT L2 NORM FOR TEST DATA AND THE COLOR PALLETE


def plot_and_compare(x):
    # Find the L2 norm and plot
    l2_norm = []
    model_name = []
    for name, df in x.items():
        a = ColourConverter.hex_to_lab(list(df['actual_LAB_code'].values))
        b = df['predicted_LAB_code'].values
        dist = {}
        for element in range(len(a)):
            dist[element] = np.linalg.norm(a[element]-b[element])
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
    return sns.palplot(sns.color_palette(pred_hex_codes))
