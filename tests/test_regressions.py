"""Small regression checks; run with python -m unittest discover -s tests -v."""
import importlib.util
from pathlib import Path
import runpy
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.io import savemat
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'simulation/train_code'))
from utils import LoadTest, LoadTraining, shuffle_crop


class DataRegressionTests(unittest.TestCase):
    def test_single_item_batch_and_exact_crop_size(self):
        image = np.full((256, 256, 28), 0.25, dtype=np.float32)
        with patch.object(torch.Tensor, 'cuda', lambda tensor: tensor):
            for batch_size in (1, 2, 3):
                with self.subTest(batch_size=batch_size):
                    result = shuffle_crop([image], batch_size)
                    self.assertEqual(tuple(result.shape), (batch_size, 28, 256, 256))
                    self.assertTrue(torch.all(result == 0.25))

    def test_splicing_uses_each_source_image_size(self):
        images = [np.ones((256, 256, 28), dtype=np.float32),
                  np.ones((384, 320, 28), dtype=np.float32)]
        with patch.object(torch.Tensor, 'cuda', lambda tensor: tensor):
            for seed in range(10):
                np.random.seed(seed)
                self.assertTrue(torch.all(shuffle_crop(images, 4) == 1))

    def test_loaders_ignore_metadata_and_accept_directory_without_slash(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            image = np.ones((256, 256, 28), dtype=np.float32)
            savemat(path / 'scene1.mat', {'img': image})
            (path / '.DS_Store').write_text('metadata')
            self.assertEqual(len(LoadTraining(directory)), 1)
            self.assertEqual(tuple(LoadTest(directory).shape), (1, 28, 256, 256))
            spec = importlib.util.spec_from_file_location(
                'simulation_test_utils', ROOT / 'simulation/test_code/utils.py')
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self.assertEqual(tuple(module.LoadTest(directory).shape), (1, 28, 256, 256))


class OptionRegressionTests(unittest.TestCase):
    def parse(self, path, *args):
        with patch.object(sys, 'argv', ['option.py', *args]):
            return runpy.run_path(str(ROOT / path))['opt']

    def test_simulation_defaults_use_supported_model_and_gpu_zero(self):
        opt = self.parse('simulation/train_code/option.py')
        self.assertEqual(opt.method, 'bisrnet')
        self.assertEqual(opt.gpu_id, '0')
        self.assertEqual(opt.data_root, '../../datasets/')

    def test_real_data_root_and_sample_count_are_respected(self):
        opt = self.parse('real/train_code/option.py', '--data_root', '/tmp/data',
                         '--epoch_sam_num', '2', '--size', '48')
        self.assertEqual(opt.trainset_num, 2)
        self.assertEqual(opt.data_path_CAVE, '/tmp/data/CAVE_512_28')
        self.assertEqual(opt.data_path_KAIST, '/tmp/data/KAIST_CVPR2021')
        self.assertEqual(opt.mask_path, '/tmp/data/TSA_real_data/mask.mat')

    def test_explicit_real_paths_override_data_root(self):
        opt = self.parse('real/train_code/option.py', '--data_root', '/tmp/data',
                         '--data_path_CAVE', '/tmp/cave', '--mask_path', '/tmp/mask.mat')
        self.assertEqual(opt.data_path_CAVE, '/tmp/cave')
        self.assertEqual(opt.mask_path, '/tmp/mask.mat')

    def test_real_default_epoch_length_is_preserved(self):
        self.assertEqual(self.parse('real/train_code/option.py').trainset_num, 20000)


if __name__ == '__main__':
    unittest.main()
