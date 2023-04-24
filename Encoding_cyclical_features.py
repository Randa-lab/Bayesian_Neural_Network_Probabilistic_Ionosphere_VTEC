import numpy as np

#Encoding the cyclical continuous features Hour and Day of year (DOY) into sine and cosine

def encode_cyclical (data_frame, input_col_hour, input_col_doy):
  
  value_hour = 24
  value_doy = 365.25

  data_frame [input_col_hour + ' (sin)'] = np.sin (2 * np.pi * data_frame[input_col_hour] / value_hour)
  data_frame [input_col_hour + ' (cos)'] = np.cos (2 * np.pi * data_frame[input_col_hour] / value_hour)
  data_frame [input_col_doy + ' (sin)'] = np.sin (2 * np.pi * data_frame[input_col_doy] / value_doy)
  data_frame [input_col_doy + ' (cos)'] = np.cos (2 * np.pi * data_frame[input_col_doy] / value_doy)
  
  data_frame = data_frame.drop([input_col_hour, input_col_doy], axis=1)
  
  
  
  ### encode_cyclical (data_df_2, 'Hour ', 'DOY')
