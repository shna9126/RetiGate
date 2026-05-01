import pandas as pd
import numpy as np

# Adjust the path to your actual CSV location
df = pd.read_csv('results/06_Evidence_Archive/table4_per_frame.csv') 
df_valid = df.dropna(subset=['roi_x1', 'roi_y1', 'roi_x2', 'roi_y2']).copy()

df_valid['roi_w']    = df_valid['roi_x2'] - df_valid['roi_x1']
df_valid['roi_h']    = df_valid['roi_y2'] - df_valid['roi_y1']
df_valid['roi_area'] = df_valid['roi_w'] * df_valid['roi_h']
df_valid['area_frac']= df_valid['roi_area'] / (1242 * 375)

print("="*50)
print("REAL RETIGATE ROI STATISTICS")
print("="*50)
print(f"Mean ROI:    {df_valid['roi_w'].mean():.0f} x {df_valid['roi_h'].mean():.0f} px")
print(f"Median ROI:  {df_valid['roi_w'].median():.0f} x {df_valid['roi_h'].median():.0f} px")
print(f"Mean area:   {df_valid['area_frac'].mean()*100:.1f}% of frame")
print(f"Median area: {df_valid['area_frac'].median()*100:.1f}% of frame")
print("="*50)