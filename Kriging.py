import pyinterpolate
from pyinterpolate import Blocks, read_txt, calc_point_to_point_distance, VariogramCloud
from pyinterpolate.variogram.empirical.experimental_variogram import calculate_semivariance

import rasterio
import numpy as np

# Read dataset
src = rasterio.open('shiyan.tif')
data = src.read(1)
num_rows = data.shape[0]
num_columns = data.shape[1]

# 1. 1000 random samples

# 1.1 create random index of 1500 to make sure at least 1000 valid points are included
rows = np.random.randint(0, num_rows, 1500)
cols = np.random.randint(0, num_columns, 1500)
z_vals = data[rows, cols]

# 1.2 filter 1000 points with valid z values
filter_idx = np.where(z_vals != src.nodatavals[0])[0]
selected_indices = np.random.choice(filter_idx, size=1000, replace=False)

rows = rows[selected_indices]
cols = cols[selected_indices]
xs, ys = rasterio.transform.xy(src.transform, rows, cols)
z_vals = z_vals[selected_indices]

# 1.3 Adding noise
x_with_noise = np.random.normal(xs, 10)
y_with_noise = np.random.normal(ys, 10)
z_with_noise = np.random.normal(z_vals, 16)
random_1k = np.vstack((x_with_noise, y_with_noise, z_with_noise)).T

# 2. Random 10000 ground truth points
rows = np.random.randint(0, num_rows, 15000)
cols = np.random.randint(0, num_columns, 15000)

z_vals = data[rows, cols]

# 1.2 filter 10000 points with valid z values
filter_idx = np.where(z_vals != src.nodatavals[0])[0]
selected_indices = np.random.choice(filter_idx, size=10000, replace=False)

rows = rows[selected_indices]
cols = cols[selected_indices]
xs, ys = rasterio.transform.xy(src.transform, rows, cols)
z_vals = z_vals[selected_indices]
random_10k = np.vstack((xs, ys, z_vals)).T

# 3. theoretical variogram
distances = calc_point_to_point_distance(random_1k[:, :-1])
maximum_range = np.max(distances) / 2

number_of_lags = 16
step_size = maximum_range / number_of_lags
vc = VariogramCloud(input_array=random_1k, step_size=step_size, max_range=maximum_range)
for idx, _lag in enumerate(vc.lags):
    print(f'Lag {_lag} has {vc.points_per_lag[idx]} point pairs.')
vc.plot('scatter')
breakpoint()
print(src)