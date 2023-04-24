#Calculation of the exponential moving average over the last 30 days and 4 days

def compute_exponential_moving_average (data_frame, input_col, new_col_EMA_30d, new_col_EMA_96h):
  ema_30d = data_frame[input_col].ewm(span=(30*24), adjust=False).mean()
  ema_96h = data_frame[input_col].ewm(span=(4*24), adjust=False).mean()
  data_frame[new_col_EMA_30d] = ema_30d
  data_frame[new_col_EMA_96h] = ema_96h
