from plotnine import ggplot, aes, geom_point, geom_smooth, coord_cartesian, scale_color_identity, theme
import pandas as pd
from tqdm import tqdm

import os

from config import DATA_PATH, PLOTS_PATH, SPAN, LOWER_QUANTILE, HIGHER_QUANTILE
from utils import create_directory_if_missing




filename = "2024-03-25_15-42-30_simulation"
df = pd.read_csv(DATA_PATH / f"{filename}.csv")
df.simulation_id = df.simulation_id.astype(object)
variables_to_plot = ["order_fullfilment_time","service_time","waiting_time","assignment_cost"]

plots_directory = PLOTS_PATH / filename
create_directory_if_missing(plots_directory)

for var in tqdm(variables_to_plot):

      # Calculate quantiles
      q_low, q_high = df[var].quantile((LOWER_QUANTILE, HIGHER_QUANTILE))

      p9 = (ggplot(df)
            + geom_point(aes(x="activation_time", y=var, color="simulation_id"), alpha=0.05)   # Add points with transparency
            + geom_smooth(aes(x="activation_time", y=var, color="simulation_id"), se=False, method="loess", span=SPAN)
            + geom_smooth(aes(x="activation_time", y=var), se=False, method="loess", span=SPAN)
            + theme(legend_position='none')
            + coord_cartesian(ylim=(q_low, q_high)) 
            )

      # Save the plot as an image file (e.g., PNG)
      p9.save(PLOTS_PATH / filename / f"{var}_{filename}.png", dpi=300)

print("Plot saved successfully as 'plot.png'")