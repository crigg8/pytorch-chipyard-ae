from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

import plot_cnn_result


class NormalizedPlotSelectionTest(unittest.TestCase):
    def test_excludes_model_without_baseline(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "device": "Rocket",
                    "model": "ResNet",
                    "4-core": np.nan,
                    "8-core": 5.0,
                    "16-core": np.nan,
                },
                {
                    "device": "Rocket",
                    "model": "AlexNet",
                    "4-core": 10.0,
                    "8-core": 5.0,
                    "16-core": np.nan,
                },
            ]
        )

        models, configs, ratios = plot_cnn_result.select_normalized_data(
            frame,
            "Rocket",
            plot_cnn_result.ROCKET_CONFIGS,
            "4-core",
        )

        self.assertEqual(models, ["AlexNet"])
        self.assertEqual([config[0] for config in configs], ["4-core", "8-core"])
        self.assertEqual(ratios["8-core"].loc["AlexNet"], 2.0)
        self.assertTrue(np.isnan(ratios["8-core"].loc["ResNet"]))

    def test_returns_no_models_when_all_baselines_are_missing(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "device": "Gemmini",
                    "model": "SqueezeNet",
                    "Dual Gemmini": np.nan,
                    "Quad Gemmini": 4.0,
                }
            ]
        )

        models, configs, _ratios = plot_cnn_result.select_normalized_data(
            frame,
            "Gemmini",
            plot_cnn_result.GEMMINI_CONFIGS,
            "Dual Gemmini",
        )

        self.assertEqual(models, [])
        self.assertEqual(configs, [])


if __name__ == "__main__":
    unittest.main()
