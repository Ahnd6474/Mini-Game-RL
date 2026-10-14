import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from Reversi import ReversiEnv
from Reversi.export_submission import export_dpn_submission
from Reversi.model import DPN, ReversiTransformerConfig


class SubmissionExportTest(unittest.TestCase):
    def test_pure_python_inference_matches_pytorch_and_protocol(self) -> None:
        torch.manual_seed(5)
        config = ReversiTransformerConfig(
            d_model=16,
            num_layers=1,
            num_heads=2,
            dim_feedforward=32,
            dropout=0.0,
        )
        model = DPN(config).eval()
        observation = ReversiEnv().observe()
        board = torch.tensor([observation.board])
        player = torch.tensor([observation.current_player])
        mask = torch.tensor([observation.legal_action_mask]).bool()
        with torch.inference_mode():
            expected = model(board, player, mask).policy_logits[0].numpy()

        with tempfile.TemporaryDirectory() as directory:
            path = export_dpn_submission(model, Path(directory) / "submission.py")
            spec = importlib.util.spec_from_file_location("generated_submission", path)
            module = importlib.util.module_from_spec(spec)
            assert spec.loader is not None
            spec.loader.exec_module(module)
            legal = module.legal_actions(list(observation.board), observation.current_player)
            actual = module.PurePythonDPN()(
                observation.board, observation.current_player, legal
            )
            np.testing.assert_allclose(
                np.asarray(actual)[legal], expected[legal], rtol=2e-2, atol=2e-2
            )
            self.assertIn(module.choose_action(list(observation.board), 1, legal), legal)
            self.assertIsInstance(module.MODEL, module.PurePythonDPN)

            request = (
                "READY FIRST\n"
                f"TURN 1 {ReversiEnv().board_string()}\n"
            )
            process = subprocess.run(
                # -S prevents site-packages (including NumPy and PyTorch) from
                # being imported, matching the restricted submission runtime.
                [sys.executable, "-S", str(path)],
                input=request,
                text=True,
                capture_output=True,
                timeout=10,
                check=True,
            )
            lines = process.stdout.strip().splitlines()
            self.assertEqual(lines[0], "OK")
            self.assertTrue(lines[1].startswith("MOVE "))


if __name__ == "__main__":
    unittest.main()
