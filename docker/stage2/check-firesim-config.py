#!/usr/bin/env python3
"""Parse the bundled manager configuration without contacting FPGA hardware."""
from pathlib import Path
import sys


def main():
    deploy = Path('/opt/chipyard/sims/firesim/deploy')
    recipes_path = deploy / 'config_build_recipes.yaml'
    hwdb_path = Path('/opt/stage2-metadata/config_hwdb.yaml')
    for path in (recipes_path, hwdb_path):
        if not path.is_file() or not path.stat().st_size:
            raise RuntimeError('missing or empty FireSim configuration: ' + str(path))
    sys.path.insert(0, str(deploy))
    from runtools.runtime_config import RuntimeBuildRecipes, RuntimeHWDB
    recipes = RuntimeBuildRecipes(str(recipes_path), 'verilator', '', '')
    hardware = RuntimeHWDB(str(hwdb_path))
    if not recipes.hwconf_dict or not hardware.hwconf_dict:
        raise RuntimeError('FireSim configuration must contain recipes and hardware entries')
    print('STAGE2_FIRESIM_CONFIG_STATUS=PASS', flush=True)


if __name__ == '__main__':
    main()
