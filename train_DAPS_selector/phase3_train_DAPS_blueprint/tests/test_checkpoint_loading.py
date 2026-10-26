from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import train_and_eval_blueprint as phase3


def make_model(num_nodes: int, num_types: int) -> phase3.PremiseSelectorModel:
    encoder = phase3.RelGTPremiseEncoder(
        num_nodes=num_nodes,
        max_neighbor_hop=2,
        node_type_map={f"type_{i}": i for i in range(num_types)},
        feature_dim=8,
        channels=8,
        local_num_layers=1,
        heads=2,
        conv_type="full",
        num_centroids=8,
        sample_node_len=3,
        select_top_k=3,
    )
    return phase3.PremiseSelectorModel(encoder, query_dim=8, proj_dim=4)


class CheckpointLoadingTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(42)
        self.source = make_model(num_nodes=17, num_types=4)
        self.target = make_model(num_nodes=7, num_types=2)
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / "checkpoint.pt"

    def assert_state_equal(self, model, expected) -> None:
        for key, tensor in model.state_dict().items():
            self.assertTrue(torch.equal(tensor, expected[key]), key)

    def test_transfer_preserves_shared_weights_and_fresh_graph_state(self) -> None:
        before = {key: value.clone() for key, value in self.target.state_dict().items()}
        torch.save({"model_state": self.source.state_dict(), "epoch": 80}, self.path)
        phase3.load_checkpoint(self.target, str(self.path), torch.device("cpu"))
        source = self.source.state_dict()
        for key, tensor in self.target.state_dict().items():
            expected = before[key] if key in phase3.PHASE2_GRAPH_SPECIFIC_KEYS else source[key]
            self.assertTrue(torch.equal(tensor, expected), key)

    def test_graph_state_is_reset_even_when_shapes_match(self) -> None:
        self.target = make_model(num_nodes=17, num_types=4)
        before = {key: self.target.state_dict()[key].clone() for key in phase3.PHASE2_GRAPH_SPECIFIC_KEYS}
        torch.save({"model_state": self.source.state_dict()}, self.path)
        phase3.load_checkpoint(self.target, str(self.path), torch.device("cpu"))
        for key, tensor in before.items():
            self.assertTrue(torch.equal(self.target.state_dict()[key], tensor), key)

    def test_supported_wrappers_and_plain_state(self) -> None:
        state = self.source.state_dict()
        for wrapper in (None, "model", "model_state", "state_dict", "model_state_dict"):
            with self.subTest(wrapper=wrapper):
                data = state if wrapper is None else {wrapper: state, "epoch": 80}
                loaded = phase3._validated_checkpoint_state(
                    self.target, data, "fixture", transfer_from_phase2=True,
                )
                self.assertEqual(set(loaded), set(state) - set(phase3.PHASE2_GRAPH_SPECIFIC_KEYS))

    def test_invalid_checkpoints_fail_before_changing_the_model(self) -> None:
        original = self.source.state_dict()
        before = {key: value.clone() for key, value in self.target.state_dict().items()}
        key = "premise_encoder.feature_proj.weight"
        wrong_shape = dict(original)
        wrong_shape[key] = wrong_shape[key][:-1]
        missing_key = dict(original)
        missing_key.pop(key)
        non_tensor = dict(original)
        non_tensor[key] = "invalid"
        unexpected_key = dict(original)
        unexpected_key["unknown.weight"] = torch.ones(1)
        for state, error in (
            (wrong_shape, RuntimeError), (missing_key, ValueError),
            (non_tensor, ValueError), (unexpected_key, ValueError),
            ({"epoch": 80}, ValueError),
        ):
            with self.subTest(keys=len(state)):
                torch.save({"model_state": state}, self.path)
                with self.assertRaises(error):
                    phase3.load_checkpoint(self.target, str(self.path), torch.device("cpu"))
                self.assert_state_equal(self.target, before)

    def test_resume_restores_graph_weights_optimizer_and_scheduler(self) -> None:
        optimizer = torch.optim.AdamW(self.source.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
        sum(p.sum() for p in self.source.parameters()).backward()
        optimizer.step()
        scheduler.step()
        torch.save({
            "model": self.source.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "epoch": 3, "best_val": 0.75,
        }, self.path)
        self.target = make_model(num_nodes=17, num_types=4)
        target_optimizer = torch.optim.AdamW(self.target.parameters(), lr=0.1)
        target_scheduler = torch.optim.lr_scheduler.StepLR(target_optimizer, step_size=1)
        epoch, best = phase3.load_resume_checkpoint(
            self.target, target_optimizer, target_scheduler, str(self.path), torch.device("cpu"),
        )
        self.assertEqual((epoch, best), (4, 0.75))
        self.assert_state_equal(self.target, self.source.state_dict())
        self.assertEqual(target_scheduler.state_dict(), scheduler.state_dict())
        self.assertEqual(target_optimizer.param_groups[0]["lr"], optimizer.param_groups[0]["lr"])
        for source_param, target_param in zip(self.source.parameters(), self.target.parameters()):
            for key, value in optimizer.state[source_param].items():
                self.assertTrue(torch.equal(value, target_optimizer.state[target_param][key]))

    def test_resume_rejects_different_graph(self) -> None:
        before = {key: value.clone() for key, value in self.target.state_dict().items()}
        torch.save({"model": self.source.state_dict()}, self.path)
        optimizer = torch.optim.AdamW(self.target.parameters())
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        with self.assertRaisesRegex(RuntimeError, "shape mismatch"):
            phase3.load_resume_checkpoint(self.target, optimizer, scheduler, str(self.path), torch.device("cpu"))
        self.assert_state_equal(self.target, before)

    def test_random_fallback_requires_explicit_option(self) -> None:
        before = {key: value.clone() for key, value in self.target.state_dict().items()}
        with self.assertRaises(FileNotFoundError):
            phase3.load_checkpoint(self.target, str(self.path), torch.device("cpu"))
        phase3.load_checkpoint(self.target, str(self.path), torch.device("cpu"), allow_random_init=True)
        self.assert_state_equal(self.target, before)


if __name__ == "__main__":
    unittest.main()
