import unittest

import torch

from GAPS_Project.src.models.tree_rec_features import apply_input_ablation


class TreeRecInputAblationTest(unittest.TestCase):
    def setUp(self):
        self.node = torch.arange(24, dtype=torch.float32).reshape(3, 8)
        self.graph = torch.arange(90, dtype=torch.float32).reshape(2, 45)

    def test_node_only_removes_event_features(self):
        node, graph = apply_input_ablation(self.node, self.graph, 'node_only')
        self.assertIs(node, self.node)
        self.assertIsNone(graph)

    def test_event_only_removes_node_features(self):
        node, graph = apply_input_ablation(self.node, self.graph, 'event_only')
        self.assertTrue(torch.equal(node, torch.zeros_like(self.node)))
        self.assertIs(graph, self.graph)

    def test_no_energy_masks_all_energy_derived_inputs(self):
        node, graph = apply_input_ablation(self.node, self.graph, 'no_energy')
        self.assertTrue(torch.equal(node[:, 3], torch.zeros(3)))
        self.assertTrue(torch.equal(node[:, 5], torch.zeros(3)))
        self.assertTrue(torch.equal(graph[:, 1:36], torch.zeros(2, 35)))
        self.assertTrue(torch.equal(node[:, 4], self.node[:, 4]))
        self.assertTrue(torch.equal(graph[:, 36:], self.graph[:, 36:]))

    def test_no_time_masks_node_and_tof_time(self):
        node, graph = apply_input_ablation(self.node, self.graph, 'no_time')
        self.assertTrue(torch.equal(node[:, 4], torch.zeros(3)))
        self.assertTrue(torch.equal(graph[:, 38:45], torch.zeros(2, 7)))
        self.assertTrue(torch.equal(node[:, 3], self.node[:, 3]))
        self.assertTrue(torch.equal(graph[:, :38], self.graph[:, :38]))


if __name__ == '__main__':
    unittest.main()
