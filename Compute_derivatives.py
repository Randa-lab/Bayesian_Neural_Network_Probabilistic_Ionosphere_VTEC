#Calculation of the first and second VTEC derivatives

def compute_derivatives (data_frame, input_col, new_col_der1, new_col_der2):
  der1 = data_frame[input_col].diff()
  der2 = data_frame[input_col].diff().diff()
  data_frame[new_col_der1] = der1
  data_frame[new_col_der2] = der2
