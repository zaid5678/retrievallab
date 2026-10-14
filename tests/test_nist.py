import json
from pathlib import Path

import pytest

from src.retrievallab.nist import (
    CATALOG_PATH,
    control_chunks,
    load_controls,
    word_chunks,
)


CATALOG = {
    "catalog": {
        "groups": [
            {
                "id": "ac",
                "title": "Access Control",
                "controls": [
                    {
                        "id": "ac-2",
                        "title": "Account Management",
                        "props": [{"name": "label", "value": "AC-2"}],
                        "params": [
                            {"id": "ac-02_odp.01", "label": "time period"},
                            {
                                "id": "ac-02_odp.02",
                                "select": {"choice": ["daily", "weekly"]},
                            },
                        ],
                        "parts": [
                            {
                                "id": "ac-2_smt",
                                "name": "statement",
                                "parts": [
                                    {
                                        "name": "item",
                                        "props": [{"name": "label", "value": "a."}],
                                        "prose": "Review accounts every {{ insert: param, ac-02_odp.01 }};",
                                    },
                                    {
                                        "name": "item",
                                        "props": [{"name": "label", "value": "b."}],
                                        "prose": "Report {{ insert: param, ac-02_odp.02 }}.",
                                    },
                                ],
                            },
                            {
                                "name": "guidance",
                                "prose": "Accounts include individual and shared accounts.",
                            },
                        ],
                        "controls": [
                            {
                                "id": "ac-2.1",
                                "title": "Automated Account Management",
                                "props": [{"name": "label", "value": "AC-2(1)"}],
                                "parts": [
                                    {
                                        "name": "statement",
                                        "prose": "Automate account management.",
                                    }
                                ],
                            },
                            {
                                "id": "ac-2.2",
                                "title": "Removed",
                                "props": [
                                    {"name": "label", "value": "AC-2(2)"},
                                    {"name": "status", "value": "withdrawn"},
                                ],
                            },
                        ],
                    }
                ],
            }
        ]
    }
}


@pytest.fixture
def controls(tmp_path):
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps(CATALOG), encoding="utf-8")
    return load_controls(path)


def test_load_controls_includes_enhancements_and_skips_withdrawn(controls):
    assert [c["control_id"] for c in controls] == ["AC-2", "AC-2(1)"]
    assert controls[0]["family"] == "Access Control"


def test_load_controls_fills_parameters(controls):
    statement = controls[0]["statement"]

    assert "a. Review accounts every [Assignment: organization-defined time period];" in statement
    assert "[Selection: daily; weekly]" in statement
    assert "{{" not in statement


def test_control_chunks_one_per_control(controls):
    chunks = control_chunks(controls)

    assert [c["control_ids"] for c in chunks] == [["AC-2"], ["AC-2(1)"]]
    assert "Discussion: Accounts include" in chunks[0]["text"]
    assert "Discussion" not in control_chunks(controls, include_guidance=False)[0]["text"]


def test_word_chunks_track_every_overlapped_control(controls):
    chunks = word_chunks(controls, chunk_size=200, overlap=0)

    assert len(chunks) == 1
    assert chunks[0]["control_ids"] == ["AC-2", "AC-2(1)"]


@pytest.mark.skipif(not CATALOG_PATH.exists(), reason="run scripts/download_nist.py")
def test_evaluation_labels_reference_active_controls():
    known = {c["control_id"] for c in load_controls()}
    dataset = json.loads(Path("data/eval/nist_questionnaire.json").read_text())

    unknown = {
        control_id
        for item in dataset
        for control_id in item["expected_controls"]
        if control_id not in known
    }
    assert not unknown
