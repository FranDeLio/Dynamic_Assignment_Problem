from config import DATA_PATH, PLOTS_PATH

from plotnine import ggplot, aes, geom_point, geom_smooth
import pandas as pd

filename = "2024-03-24_01-56-42_simulation.csv"
df=pd.read_csv(DATA_PATH / filename).loc[lambda x: x.orders_is_served==True].reset_index(drop=True)

# Assuming df is your DataFrame containing data
p9 = (ggplot(data=df)
      + geom_point(aes(x="activation_time", y="order_fullfilment_time", color="simulation_id"), alpha=0.01)   # Add points with transparency
      + geom_smooth(aes(x="activation_time", y="order_fullfilment_time", color="simulation_id"), se=False, method="loess")
      )  # Add loess smoothing

# Save the plot as an image file (e.g., PNG)
p9.save(PLOTS_PATH / f"{filename}.png", dpi=300)

p9 = (ggplot(data=df)
      + geom_point(aes(x="activation_time", y="service_time", color="simulation_id"), alpha=0.01)   # Add points with transparency
      + geom_smooth(aes(x="activation_time", y="service_time", color="simulation_id"), se=False, method="loess")
      )  # Add loess smoothing

# Save the plot as an image file (e.g., PNG)
p9.save(PLOTS_PATH / f"service_{filename}.png", dpi=300)

print("Plot saved successfully as 'plot.png'")