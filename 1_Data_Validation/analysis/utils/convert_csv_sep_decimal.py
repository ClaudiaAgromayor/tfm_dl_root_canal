import pandas as pd
from pathlib import Path
p = Path(r"c:\Users\clauu\OneDrive - Universidad Pontificia Comillas\Documentos\ICAI\Beca IIT\1_Data_Validation\results\canales\hd95_diente_completo\hd95_global_summary.csv")
backup = p.with_name(p.stem + "_backup.csv")
print('Reading:', p)
# Read current CSV (comma sep, dot decimal)
df = pd.read_csv(p, sep=',', decimal='.', dtype=str)
# Convert numeric-like columns where possible
for col in ['gt_voxels','manual_voxels','dice_3d','hd95_3d']:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col].replace('', pd.NA), errors='coerce')
# Save backup (overwrite if exists)
if p.exists():
    p.replace(backup)
    print('Backup saved as:', backup)
# Write CSV with semicolon sep and comma decimal
# Use float_format to keep reasonable precision; NaN will be empty
p_out = p
# pandas to_csv will use locale if decimal specified
with p_out.open('w', encoding='utf-8', newline='') as f:
    df.to_csv(f, sep=';', decimal=',', index=False, float_format='%.6f')
print('Rewrote CSV with sep=";" and decimal="," at:', p_out)
print(df.head(5).to_string(index=False))
