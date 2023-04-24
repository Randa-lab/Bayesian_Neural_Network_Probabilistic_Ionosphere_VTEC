import numpy as np

#Calculation of percentage of ground truth (GT) within CI

def compute_percentage_in_CI (y_test, y_pred_upper, y_pred_lower):
  diff_up = y_pred_upper - y_test
  diff_low = y_pred_lower - y_test
  
  in_values=0
  out_values=0
  
  for i in range(diff_up.size):
    if np.any((diff_up[i] >= 0) & (diff_low[i] <= 0)):
      in_values = in_values + 1
    else:
      out_values = out_values + 1
        
  whole = in_values + out_values
  
  percent_in_CI = in_values/whole * 100
  percent_out_CI = out_values/whole * 100

  print('GT  within the CI (%):', round(percent_in_CI,2))
  print('GT outside the CI (%):', round(percent_out_CI,2))
