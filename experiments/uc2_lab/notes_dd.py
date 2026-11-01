"""Write messy lab notes with NeMo Data Designer, pointed at the local vLLM.

Structured runs are the seed dataset; one LLM text column turns each into a
note. Runs in the tools venv. Telemetry stays off, and the only provider is the
loopback vLLM, so nothing leaves the node.

Usage: NEMO_TELEMETRY_ENABLED=false python notes_dd.py WORLD [LIMIT]
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd
from data_designer.config import (ChatCompletionInferenceParams, DataDesignerConfigBuilder, DataFrameSeedSource,
                                  LLMTextColumnConfig, ModelConfig, ModelProvider)
from data_designer.interface import DataDesigner

HERE = Path(__file__).resolve().parent
STYLES = {"Mei": "very terse, fragments, lots of abbreviations", "Tomas": "telegraphic, slashes between fields",
          "Priya": "short complete sentences", "Diego": "lowercase, casual, some shorthand"}
HELDOUT_STYLE = "a new rotation student: longer, careful sentences, spells things out"
PROMPT = """Rewrite this lab record as a notebook entry in the style: {{ style }}.
Keep every value and every lot, strain, column, instrument and construct name exactly as written. Add no new facts and
give no opinion about why the run turned out as it did. Output only the entry, 40 to 90 words.

Record: {{ facts }}"""


def main() -> None:
    world, limit = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 0
    out = HERE / "data" / world
    rows = [json.loads(l) for l in open(out / "records.jsonl")]
    rows = rows[:limit] if limit else rows
    df = pd.DataFrame([{"id": r["id"], "facts": r["facts"],
                        "style": HELDOUT_STYLE if r["split"] == "heldout" else STYLES[r["operator"]]} for r in rows])
    provider = ModelProvider(name="local-vllm", endpoint="http://127.0.0.1:8000/v1", provider_type="openai", api_key="EMPTY")
    model = ModelConfig(alias="bolt", model="bolt", provider="local-vllm", skip_health_check=True,
                        inference_parameters=ChatCompletionInferenceParams(max_tokens=220, temperature=0.8, max_parallel_requests=8))
    builder = DataDesignerConfigBuilder(model_configs=[model])
    builder.with_seed_dataset(DataFrameSeedSource(df=df))
    builder.add_column(LLMTextColumnConfig(name="note", prompt=PROMPT, model_alias="bolt"))
    t0 = time.time()
    dd = DataDesigner(artifact_path=str(Path.home() / "beetle-exp/tools/dd_artifacts"), model_providers=[provider])
    result = dd.create(builder, num_records=len(df), dataset_name=f"uc2_{world}")
    table = result.load_dataset()
    with open(out / "notes.jsonl", "w") as f:
        for _, row in table.iterrows():
            f.write(json.dumps({"id": int(row["id"]), "note": row["note"]}, ensure_ascii=False) + "\n")
    print("notes", len(table), "seconds", round(time.time() - t0, 1), flush=True)


if __name__ == "__main__":
    main()
