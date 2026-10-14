#!/usr/bin/env python3
"""Exercise restricted smoke dispatch without Docker, sudo, or FPGA access."""
import importlib.util
import json
import os
from pathlib import Path
import pwd
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location('smoke_stage2_launch',
                                            Path(__file__).with_name('smoke-stage2-launch.py'))
launcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launcher)


class SmokeDispatchTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / 'inputs'
        for backend, cores in launcher.BACKENDS:
            directory = self.source / backend
            directory.mkdir(parents=True)
            for name in (f'model-{cores}core.elf', 'input.bin', 'weights.bin', 'model_spec.json',
                         '.pytorch-chipyard-workload-rel', '.pytorch-chipyard-build-cores'):
                (directory / name).write_text('fixture')
        self.installed = self.root / 'installed'
        shutil.copytree(self.source, self.installed)
        self.config = {'user': pwd.getpwuid(os.getuid()).pw_name, 'image_id': 'sha256:' + 'a' * 64,
                       'smoke_inputs': str(self.installed), 'fpga_db': '/opt/card.json',
                       'ssh_host': 'fpga.example.org', 'vivado': '/opt/Vivado/2021.1'}
        self.args = types.SimpleNamespace(ssh_agent='/tmp/agent.sock', resume=False,
                                         rerun_completed=False, rvv_panic_retries=0,
                                         preflight_only=False)
        self.env = {'STAGE2_SMOKE_ARTIFACT_DIR': str(self.source)}

    def command(self):
        return launcher.restricted_command(self.args, self.env, self.config, self.config['image_id'])

    def test_current_inputs_use_only_the_fixed_sudo_entrypoint_and_preserve_resume(self):
        command = self.command()
        self.assertEqual(command, ['sudo', '-n', str(launcher.ENTRY), '--ssh-agent', '/tmp/agent.sock'])
        self.env['RUN_ID'] = 'smoke-fixture'
        self.args.resume = self.args.preflight_only = True
        self.assertEqual(self.command()[-3:], ['--resume', 'smoke-fixture', '--preflight-only'])

    def test_changed_image_inputs_and_hardware_never_use_the_old_profile(self):
        with self.assertRaisesRegex(RuntimeError, 'new image'):
            launcher.restricted_command(self.args, self.env, self.config, 'sha256:' + 'b' * 64)
        self.env['FIRESIM_DB'] = '/tmp/other-card.json'
        with self.assertRaisesRegex(RuntimeError, 'FIRESIM_DB differs'):
            self.command()
        self.env.pop('FIRESIM_DB')
        (self.source / 'rvv/model-4core.elf').write_text('newly compiled ELF')
        with self.assertRaisesRegex(RuntimeError, 'inputs differ'):
            self.command()

    def test_denied_system_socket_dispatches_to_sudo_and_inspects_only_the_build_engine(self):
        config = self.root / 'profile.json'
        config.write_text(json.dumps(self.config))
        entry = self.root / 'run-smoke'
        entry.touch()
        env = dict(self.env, DOCKER_HOST='unix:///build/docker.sock')
        result = subprocess.CompletedProcess([], 0, self.config['image_id'] + '\n', '')
        with patch.object(launcher, 'CONFIG', config), patch.object(launcher, 'ENTRY', entry), \
                patch.dict(os.environ, env, clear=True), patch.object(sys, 'argv', ['launcher', '--ssh-agent', '/tmp/agent.sock']), \
                patch.object(launcher.os, 'access', return_value=False), patch.object(launcher.os, 'geteuid', return_value=1000), \
                patch.object(launcher.subprocess, 'run', return_value=result) as docker, \
                patch.object(launcher.os, 'execvp') as execute:
            launcher.main()
        self.assertEqual(docker.call_args.args[0][:3], ['docker', 'image', 'inspect'])
        self.assertEqual(docker.call_args.kwargs['env']['DOCKER_HOST'], 'unix:///build/docker.sock')
        self.assertEqual(execute.call_args.args, ('sudo', ['sudo', '-n', str(entry), '--ssh-agent', '/tmp/agent.sock']))

    def test_direct_and_print_command_paths_do_not_require_a_restricted_profile(self):
        for arguments, access in (([], True), (['--print-command'], False)):
            with self.subTest(arguments=arguments), patch.dict(os.environ, {}, clear=True), \
                    patch.object(sys, 'argv', ['launcher', *arguments]), \
                    patch.object(launcher.os, 'access', return_value=access), \
                    patch.object(launcher.os, 'geteuid', return_value=1000), \
                    patch.object(launcher.subprocess, 'run') as docker, \
                    patch.object(launcher.os, 'execvp', side_effect=SystemExit) as execute:
                with self.assertRaises(SystemExit):
                    launcher.main()
                self.assertEqual(execute.call_args.args[0], 'bash')
                self.assertIn('--smoke-test', execute.call_args.args[1])
                docker.assert_not_called()


if __name__ == '__main__':
    unittest.main()
