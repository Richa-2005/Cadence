import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch

from src.training.metrics import verification_metrics
from src.data.normalization import fit_normalizer
from src.data.pair_generation import load_pair_index, generate_fixed_pairs, write_pairs


class TrainingTests(unittest.TestCase):
    def test_metrics_direction_and_ties(self):
        self.assertEqual(verification_metrics([.1, .2, .8, .9], [1, 1, 0, 0])['eer'], 0)
        self.assertEqual(verification_metrics([.5]*4, [1, 0, 1, 0])['eer'], .5)
        self.assertEqual(verification_metrics([.9, .8, .1, .2], [1, 1, 0, 0])['eer'], 1)
        with self.assertRaises(ValueError):
            verification_metrics([.1], [1])

    def test_two_epoch_training_and_checkpoints(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            rng = np.random.default_rng(42)
            subjects = np.repeat(['a', 'b', 'c', 'd'], 4)
            x = rng.uniform(.01, .3, (16, 70, 3)).astype(np.float32)
            x[:, 5:] = 0
            x[:, 0, 1:] = 0
            mask = np.arange(70)[None, :] < np.full((16, 1), 5)
            np.savez(root / 'aalto_shard_0000.npz', features=x, mask=mask,
                     seq_len=np.full(16, 5), subject_id=subjects,
                     sample_id=np.array([f's{i}' for i in range(16)]), session=np.ones(16))
            split = root / 'split.json'
            split.write_text(json.dumps(dict(train=['a', 'b'], val=['c', 'd'], test=[])))
            train = load_pair_index('aalto', 'train', root, split)
            normalizer, _ = fit_normalizer([train])
            normalizer.save(root / 'norm.json')
            val = load_pair_index('aalto', 'val', root, split)
            write_pairs(generate_fixed_pairs(val, 16), root / 'val.npz')
            result = subprocess.run([sys.executable, '-m', 'src.training.train',
                '--processed-dir', str(root), '--split-path', str(split),
                '--normalizer-path', str(root / 'norm.json'),
                '--val-pairs-path', str(root / 'val.npz'), '--output-dir', str(root / 'run'),
                '--dataset-version', 'synthetic-test', '--epochs', '2',
                '--pairs-per-epoch', '16', '--batch-size', '8', '--device', 'cpu'],
                capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            history = json.loads((root / 'run/history.json').read_text())
            self.assertEqual(len(history), 2)
            self.assertEqual(history[1]['val']['pairs'], 16)
            # Own freshly generated checkpoint, including Python/NumPy RNG state.
            checkpoint = torch.load(root / 'run/last.pt', weights_only=False)
            best = torch.load(root / 'run/best.pt', weights_only=False)
            self.assertEqual(checkpoint['epoch'], 2)
            self.assertEqual(best['best_val_eer'], min(row['val']['eer'] for row in history))
            self.assertIn('normalizer', checkpoint)
            self.assertIn('artifact_hashes', checkpoint['config'])

            command = list(result.args)
            command[command.index('--output-dir') + 1] = str(root / 'continued')
            command[command.index('--epochs') + 1] = '1'
            command += ['--resume', str(root / 'run/last.pt'), '--lr', '0.0003']
            resumed = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(resumed.returncode, 0, resumed.stderr)
            continued = torch.load(root / 'continued/last.pt', weights_only=False)
            self.assertEqual(continued['epoch'], 3)
            self.assertEqual(len(continued['history']), 3)
            self.assertEqual(continued['optimizer']['param_groups'][0]['lr'], .0003)
            self.assertTrue((root / 'continued/best.pt').exists())
            self.assertIn('resume_sha256', continued['config'])
            bad_command = list(command)
            bad_command[bad_command.index('--output-dir') + 1] = str(root / 'bad')
            bad_command += ['--seed', '99']
            rejected = subprocess.run(bad_command, capture_output=True, text=True)
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn('Resume mismatch: seed', rejected.stderr)

            mining_command = list(result.args)
            mining_command[mining_command.index('--output-dir') + 1] = str(root / 'mined')
            mining_command += ['--negative-sampling', 'hard']
            mined = subprocess.run(mining_command, capture_output=True, text=True)
            self.assertEqual(mined.returncode, 0, mined.stderr)
            mining_checkpoint = torch.load(root / 'mined/last.pt', weights_only=False)
            self.assertEqual(mining_checkpoint['config']['negative_sampling'], 'hard')
            self.assertEqual(mining_checkpoint['history'][-1]['train']['pairs'], 16)
            self.assertGreater(mining_checkpoint['history'][-1]['train']['candidates'], 16)
            resumed_command = list(mining_command)
            resumed_command[resumed_command.index('--output-dir') + 1] = str(root / 'mined_resume')
            resumed_command[resumed_command.index('--epochs') + 1] = '1'
            resumed_command += ['--resume', str(root / 'mined/last.pt')]
            mined_resume = subprocess.run(resumed_command, capture_output=True, text=True)
            self.assertEqual(mined_resume.returncode, 0, mined_resume.stderr)
            continued_mining = torch.load(root / 'mined_resume/last.pt', weights_only=False)
            self.assertEqual(continued_mining['epoch'], 3)
