import numpy as np

#Converting the cyclical continuous features into sine and cosine

def transform_sine_cosine (data_frame, input_col_h, input_col_d,
                           new_col_h_sin, new_col_h_cos,
                           new_col_d_sin, new_col_d_cos):
  
  data_frame = data_frame.drop([ 'DOY', 'Hour '], axis=1)

  hour_sin = np.sin (2 * np.pi * data_frame['Hour ']/24.0)
  hour_cos = np.cos (2 * np.pi * data_frame['Hour ']/24.0)
  doy_sin = np.sin (2 * np.pi * data_frame['DOY']/365.25)
  doy_cos = np.cos (2 * np.pi * data_frame['DOY']/365.25)

  data_frame[new_col_h_sin] = hour_sin
  data_frame[new_col_h_cos] = hour_cos
  data_frame[new_col_d_sin] = doy_sin
  data_frame[new_col_d_cos] = doy_cos
