import json
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROUTING_CONFIG_PATH = PROJECT_ROOT / "config" / "routing_config.json"
MODEL_CONFIG_PATH = PROJECT_ROOT / "models" / "distilbert_final" / "config.json"


class RoutingConfigTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        with ROUTING_CONFIG_PATH.open("r", encoding="utf-8") as file:
            cls.routing_config = json.load(file)

        with MODEL_CONFIG_PATH.open("r", encoding="utf-8") as file:
            cls.model_config = json.load(file)

    def test_expected_number_of_classes(self):
        self.assertEqual(self.routing_config["num_classes"], 77)
        self.assertEqual(len(self.routing_config["department_mapping"]), 77)

    def test_every_model_label_has_department_mapping(self):
        model_labels = set(self.model_config["id2label"].values())
        routing_labels = set(self.routing_config["department_mapping"].keys())

        self.assertEqual(model_labels, routing_labels)

    def test_thresholds_are_valid_probabilities(self):
        threshold = self.routing_config["confidence_threshold"]
        target_accuracy = self.routing_config["target_auto_accuracy"]

        self.assertGreaterEqual(threshold, 0.0)
        self.assertLessEqual(threshold, 1.0)
        self.assertGreaterEqual(target_accuracy, 0.0)
        self.assertLessEqual(target_accuracy, 1.0)

    def test_max_length_is_positive(self):
        self.assertGreater(self.routing_config["max_length"], 0)

    def test_department_names_are_non_empty(self):
        departments = self.routing_config["department_mapping"].values()

        self.assertTrue(all(isinstance(name, str) and name.strip() for name in departments))


if __name__ == "__main__":
    unittest.main()
