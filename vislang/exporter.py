"""DatasetExporter: HuggingFace datasets-compatible JSONL export."""

import json
import os


class DatasetExporter:
    """Export dataset pairs to HuggingFace-compatible JSONL format."""

    def __init__(self, output_dir):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def export_jsonl(self, pairs, filename="train.jsonl"):
        """Export pairs to JSONL. Returns file path."""
        path = os.path.join(self.output_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            for pair in pairs:
                record = {
                    "image_url": pair.get("image_url", ""),
                    "text": pair.get("sentence", pair.get("text", "")),
                    "language": pair.get("language", "en"),
                    "score": pair.get("score", 0.0),
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return path

    def export_metadata(self, stats, filename="dataset_info.json"):
        """Export stats dict to JSON. Returns file path."""
        path = os.path.join(self.output_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            serializable = {k: list(v) if isinstance(v, set) else v for k, v in stats.items()}
            json.dump(serializable, f, indent=2, ensure_ascii=False)
        return path

    def load_jsonl(self, path):
        """Load a JSONL file and return a list of dicts."""
        records = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records
