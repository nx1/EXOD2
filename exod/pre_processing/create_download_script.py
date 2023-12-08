import os
from pathlib import Path

from exod.utils import path
from exod.utils.logger import logger

def read_observation_ids(file_path):
    """
    Read observation IDs from file.
    Each line should be a single obsid.
    """
    with open(file_path, 'r') as file:
        obs_ids = [line.strip() for line in file.readlines()]
    return obs_ids

obs_list_path  = path.data / 'observations.txt'
output_sh_path = path.data / 'download_obs.sh'
save_dir       = path.data_downloaded

logger.info(f'Observations List Path: {obs_list_path}')
logger.info(f'Download Script Path: {output_sh_path}')
logger.info(f'Save Directory: {save_dir}')

logger.info(f'Reading observations from {obs_list_path}')
observation_ids = read_observation_ids(obs_list_path)
logger.info(f'Found {len(observation_ids)} observations ids')

lines = []
for obs in observation_ids:
    lines.append(f'wget -nc -O {save_dir}/{obs}.tar https://nxsa.esac.esa.int/nxsa-sl/servlet/data-action-aio?obsno={obs}&level=PPS')

# Add newline and & for running concuirrently
lines = [o+'& \n' for o in lines]

# Hack to get Ctrl + C to kill all processes
killcmd = """
# Function to clean up and exit
cleanup_and_exit() {
    echo "Terminating script..."
    # Terminate all child processes (wget commands)
    pkill -O $$
    exit 1
}

# Trap Ctrl+C (SIGINT) to run cleanup_and_exit function
trap cleanup_and_exit SIGINT
"""

with open(output_sh_path, 'w+') as f:
    f.write(killcmd)
    f.writelines(lines)
    f.write('wait\n')
    f.write('echo "Script completed successfully!"')

logger.info(f'{len(lines)} lines written to {output_sh_path}')

