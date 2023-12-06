import os
from pathlib import Path
from logger import logger

def read_observation_ids(file_path):
    with open(file_path, 'r') as file:
        obs_ids = [line.strip() for line in file.readlines()]
    return obs_ids

script_path = Path(__file__).resolve()

obs_list_path = script_path.parent.parent / 'data' / 'observations.txt'
output_sh_path = script_path.parent.parent / 'data' / 'download_obs.sh'
save_dir = script_path.parent.parent / 'data' / 'observations'

logger.info(f'Script Path: {script_path}')
logger.info(f'Observations List Path: {obs_list_path}')
logger.info(f'Download Script Path: {output_sh_path}')
logger.info(f'Save Directory: {save_dir}')

logger.info(f'Reading observations from {obs_list_path}')
observation_ids = read_observation_ids(obs_list_path)
logger.info(f'Found {len(observation_ids)} observations ids')

lines = []
for obs in observation_ids:
    os.makedirs(f'{save_dir}/{obs}', exist_ok=True)
    lines.append(f'wget -nc -O {save_dir}/{obs}/P{obs}PNS001PIEVLI.FTZ http://nxsa.esac.esa.int/nxsa-sl/servlet/data-action-aio?obsno={obs}&instname=PN&level=PPS&name=PIEVLI')
    lines.append(f'wget -nc -O {save_dir}/{obs}/P{obs}PNS001FBKTSR0000.FTZ http://nxsa.esac.esa.int/nxsa-sl/servlet/data-action-aio?obsno={obs}&name=FBKTSR&instname=PN&level=PPS&extension=FTZ')
    lines.append(f'wget -nc -O {save_dir}/{obs}/P{obs}M1S002MIEVLI.FTZ http://nxsa.esac.esa.int/nxsa-sl/servlet/data-action-aio?obsno={obs}&instname=M1&level=PPS&name=MIEVLI')
    lines.append(f'wget -nc -O {save_dir}/{obs}/P{obs}M1S002FBKTSR0000.FTZ http://nxsa.esac.esa.int/nxsa-sl/servlet/data-action-aio?obsno={obs}&name=FBKTSR&instname=M1&level=PPS&extension=FTZ')
    lines.append(f'wget -nc -O {save_dir}/{obs}/P{obs}M2S003MIEVLI.FTZ http://nxsa.esac.esa.int/nxsa-sl/servlet/data-action-aio?obsno={obs}&instname=M2&level=PPS&name=MIEVLI')
    lines.append(f'wget -nc -O {save_dir}/{obs}/P{obs}M2S003FBKTSR0000.FTZ http://nxsa.esac.esa.int/nxsa-sl/servlet/data-action-aio?obsno={obs}&name=FBKTSR&instname=M2&level=PPS&extension=FTZ')
    lines.append(f'wget -nc -O {save_dir}/{obs}/{obs}.tar.gz http://nxsa.esac.esa.int/nxsa-sl/servlet/data-action-aio?obsno={obs}&level=ODF&extension=SAS')

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

