from pathlib import Path
from src.data import read_dat

raw_dir = Path('hetrec_data')
df = read_dat(raw_dir, 'movie_tags.dat')
print(f'Tag file shape: {df.shape}')
if len(df.columns) > 0:
    print(f'Columns: {df.columns.tolist()}')
    print(f'First few rows:')
    print(df.head())
    print(f'Total tag assignments: {len(df)}')
else:
    print('No columns - file may be empty or malformed')

