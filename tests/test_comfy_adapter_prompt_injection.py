from __future__ import annotations

from sd_optim.gen_adapters import ComfyUIAdapter


def test_injects_payload_prompts_for_samplercustom_workflow() -> None:
    workflow = {
        "1": {
            "class_type": "SamplerCustom",
            "inputs": {
                "positive": ["2", 0],
                "negative": ["3", 0],
            },
        },
        "2": {
            "class_type": "A1111Prompt",
            "inputs": {"text": "old positive"},
        },
        "3": {
            "class_type": "A1111PromptNegative",
            "inputs": {"text": "old negative"},
        },
    }
    payload = {
        "prompt": "new positive",
        "negative_prompt": "new negative",
    }

    adapter = ComfyUIAdapter("http://localhost:8188")
    adapter._inject_parameters(workflow, payload)

    assert workflow["2"]["inputs"]["text"] == "new positive"
    assert workflow["3"]["inputs"]["text"] == "new negative"
