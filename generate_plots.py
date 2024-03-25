from config import DATA_PATH, PLOTS_PATH

import os

from plotnine import ggplot, aes, geom_point, geom_smooth, coord_cartesian, scale_color_identity, theme
import pandas as pd
from tqdm import tqdm

span = 0.4
qh = 0.95
ql = 0.05
filename = "2024-03-25_15-42-30_simulation"
df = pd.read_csv(DATA_PATH / f"{filename}.csv")
df.simulation_id = df.simulation_id.astype(object)
variables_to_plot = ["order_fullfilment_time","service_time","waiting_time","assignment_cost"]

directory = PLOTS_PATH / filename
if not os.path.exists(directory):
        os.makedirs(directory)

for var in tqdm(variables_to_plot):

      # Calculate quantiles
      q_low, q_high = df[var].quantile((ql,qh))

      p9 = (ggplot(df)
            + geom_point(aes(x="activation_time", y=var, color="simulation_id"), alpha=0.05)   # Add points with transparency
            + geom_smooth(aes(x="activation_time", y=var, color="simulation_id"), se=False, method="loess", span=span)
            + geom_smooth(aes(x="activation_time", y=var), se=False, method="loess", span=span)
            + theme(legend_position='none')
            + coord_cartesian(ylim=(q_low, q_high)) 
            )

      # Save the plot as an image file (e.g., PNG)
      p9.save(PLOTS_PATH / filename / f"{var}_{filename}.png", dpi=300)

print("Plot saved successfully as 'plot.png'")