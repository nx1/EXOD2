import shutil
from exod.utils.path import data_plots, docs
from pathlib import Path

source_dir = data_plots
target_dir = docs / 'plots'
png_files = list(data_plots.glob('*png'))
print(f'png_files: {png_files}')

for files_path in png_files:
    shutil.copy(files_path, target_dir)
    print(f'Copying {files_path} to {target_dir}')

png_files = list(target_dir.glob('*png'))
with open(docs / 'plots.md', 'w') as f:
    f.write('# Plots\n\n')
    for file_path in png_files:
        f.write(f'## {file_path.name}\n\n')
        f.write(f'![{file_path.name}](plots/{file_path.name})\n\n')
        print(f'Writing {file_path.name} to {docs / "plots.md"}')


