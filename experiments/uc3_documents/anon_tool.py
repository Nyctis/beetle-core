"""Anonymize the UC3 corpus with NVIDIA NeMo Anonymizer, all models local.

Detection uses the pinned GLiNER2 model on a loopback server; the validator and
the substitution roles use Bolt on the loopback vLLM. Nothing leaves the node and
telemetry is off. Writes the `tool` variant into data/{corpus}.json.

Usage (tools venv): python anon_tool.py final
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path

import pandas as pd
from anonymizer import AnonymizerConfig, AnonymizerInput, Detect, Substitute
from anonymizer.interface.anonymizer import Anonymizer
from data_designer.config import ModelProvider

HERE = Path(__file__).resolve().parent
MODELS = """
selected_models:
  detection:
    entity_detector: gliner-pii-detector
    entity_validator: bolt-local
    entity_augmenter: bolt-local
    latent_detector: bolt-local
  replace:
    replacement_generator: bolt-local
model_configs:
  - alias: gliner-pii-detector
    model: fastino/gliner2-privacy-filter-PII-multi
    provider: local-gliner2
    skip_health_check: true
    inference_parameters:
      max_parallel_requests: 8
      timeout: 120
  - alias: bolt-local
    model: bolt
    provider: local-vllm
    skip_health_check: true
    inference_parameters:
      max_parallel_requests: 6
      max_tokens: 3000
      temperature: 0.2
      timeout: 300
"""


def main() -> None:
    corpus, limit = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 0
    path = HERE / f"data/{corpus}.json"
    b = json.loads(path.read_text())
    texts = b["variants"]["raw"][:limit] if limit else b["variants"]["raw"]
    providers = [ModelProvider(name="local-gliner2", endpoint="http://127.0.0.1:8001/v1", provider_type="openai", api_key="EMPTY"),
                 ModelProvider(name="local-vllm", endpoint="http://127.0.0.1:8000/v1", provider_type="openai", api_key="EMPTY")]
    tmp = Path(tempfile.mkdtemp())
    source = tmp / "docs.csv"
    pd.DataFrame({"id": [str(i) for i in range(len(texts))], "text": texts}).to_csv(source, index=False)
    models = tmp / "models.yaml"
    models.write_text(MODELS)
    anon = Anonymizer(model_configs=str(models), model_providers=providers, artifact_path=str(tmp / "art"))
    config = AnonymizerConfig(detect=Detect(gliner_threshold=0.3), replace=Substitute(), emit_telemetry=False)
    t0 = time.time()
    result = anon.run(config=config, data=AnonymizerInput(source=str(source), text_column="text", id_column="id",
                                                          data_summary="Internal documents of a fictional insurance institution."))
    frame = result.dataframe if hasattr(result, "dataframe") else result
    print("columns", list(frame.columns), "rows", len(frame), "seconds", round(time.time() - t0, 1), flush=True)
    # The replacement map stays with the data owner; it is what makes local re-mapping possible.
    frame = frame.assign(replacement_map=list(result.trace_dataframe["_replacement_map"]),
                         failed=len(getattr(result, "failed_records", []) or []))
    frame.to_json(HERE / f"data/{corpus}_tool_raw.jsonl", orient="records", lines=True)


if __name__ == "__main__":
    main()
