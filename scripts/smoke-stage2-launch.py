#!/usr/bin/env python3
"""Select direct Docker access or the administrator-installed smoke profile."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import pwd
import subprocess
import sys
from urllib.parse import unquote, urlparse

REPO = Path(__file__).resolve().parents[1]
CONFIG = Path('/etc/pytorch-chipyard-stage2-local.json')
ENTRY = Path('/usr/local/sbin/run-stage2-local-smoke')
BACKENDS = (('rvv', 4), ('gemmini', 4), ('scalar', 16))


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def check_inputs(source, installed):
    for backend, cores in BACKENDS:
        directory = source / backend
        names = {f'model-{cores}core.elf', 'input.bin', 'weights.bin',
                 'model_spec.json', '.pytorch-chipyard-workload-rel'}
        names.update(path.name for path in directory.glob('.pytorch-chipyard-*'))
        for name in sorted(names):
            current = directory / name
            target = installed / backend / name
            if current.is_symlink() or not current.is_file() or not current.stat().st_size:
                raise RuntimeError('missing Stage 1 smoke input: ' + str(current))
            if target.is_symlink() or not target.is_file() or sha(current) != sha(target):
                raise RuntimeError('Stage 1 inputs differ from the installed smoke profile; '
                                   'an administrator must refresh the profile: ' + str(current))


def restricted_command(args, env, config, image_id):
    if config['user'] != pwd.getpwuid(os.getuid()).pw_name:
        raise RuntimeError('the installed smoke profile belongs to another account')
    if image_id != config['image_id']:
        raise RuntimeError('the selected Stage 2 image differs from the installed smoke profile; '
                           'an administrator must register the new image')
    for variable, key in (('FIRESIM_DB', 'fpga_db'), ('FIRESIM_SSH_HOST', 'ssh_host'),
                          ('FIRESIM_SSH_USER', 'user'), ('FIRESIM_VIVADO_DIR', 'vivado')):
        value = env.get(variable)
        if value and value != config[key]:
            raise RuntimeError(variable + ' differs from the installed smoke profile')
    if env.get('FIRESIM_HWDB_PATH'):
        raise RuntimeError('unset FIRESIM_HWDB_PATH to use the installed smoke profile')
    if args.rerun_completed or args.rvv_panic_retries:
        raise RuntimeError('the restricted profile supports fresh runs, --resume, and --preflight-only')
    if not args.ssh_agent:
        raise RuntimeError('provide --ssh-agent PATH or SSH_AUTH_SOCK')
    source = Path(env.get('STAGE2_SMOKE_ARTIFACT_DIR',
                          str(REPO / 'results/smoke-test/artifacts/pytorch-chipyard')))
    check_inputs(source, Path(config['smoke_inputs']))
    command = ['sudo', '-n', str(ENTRY), '--ssh-agent', args.ssh_agent]
    run_id = env.get('RUN_ID')
    if args.resume:
        if not run_id:
            raise RuntimeError('--resume requires RUN_ID')
        command.extend(['--resume', run_id])
    elif run_id:
        command.extend(['--run-id', run_id])
    if args.preflight_only:
        command.append('--preflight-only')
    return command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ssh-agent', default=os.environ.get('SSH_AUTH_SOCK'))
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument('--resume', action='store_true')
    selection.add_argument('--rerun-completed', action='store_true')
    parser.add_argument('--preflight-only', action='store_true')
    parser.add_argument('--print-command', action='store_true')
    parser.add_argument('--rvv-panic-retries', type=int, default=0)
    args = parser.parse_args()
    endpoint = urlparse(os.environ.get('STAGE2_DOCKER_HOST', 'unix:///var/run/docker.sock'))
    socket = unquote(endpoint.path)
    if (args.print_command or os.geteuid() == 0 or endpoint.scheme != 'unix'
            or socket not in ('/var/run/docker.sock', '/run/docker.sock')
            or os.access(socket, os.R_OK | os.W_OK)):
        os.execvp('bash', ['bash', str(REPO / 'scripts/run-stage2-docker.sh'),
                          '--smoke-test', *sys.argv[1:]])
    if not CONFIG.is_file() or not ENTRY.is_file():
        raise RuntimeError('system Docker access is unavailable; an administrator must install '
                           'the restricted smoke profile with install-stage2-local-smoke-launcher.py')
    config = json.loads(CONFIG.read_text())
    env = dict(os.environ)
    image = env.get('STAGE2_IMAGE', 'pytorch-chipyard:stage2')
    env.setdefault('DOCKER_HOST', f'unix:///run/user/{os.getuid()}/docker.sock')
    result = subprocess.run(['docker', 'image', 'inspect', '--format', '{{.Id}}', image],
                            env=env, text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError('cannot inspect the Stage 2 image in the build Docker engine: '
                           + result.stderr.strip())
    command = restricted_command(args, env, config, result.stdout.strip())
    print('[smoke-stage2] using installed restricted profile: ' + config['image_id'], flush=True)
    os.execvp(command[0], command)


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        print('[smoke-stage2][error] ' + str(error), file=sys.stderr)
        sys.exit(1)
