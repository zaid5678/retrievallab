import json
import re
from pathlib import Path


CATALOG_URL = (
    "https://raw.githubusercontent.com/usnistgov/oscal-content/main/"
    "nist.gov/SP800-53/rev5/json/NIST_SP-800-53_rev5_catalog.json"
)
CATALOG_PATH = Path("data/raw/nist_800_53_rev5_catalog.json")

PARAM_PATTERN = re.compile(r"\{\{\s*insert:\s*param,\s*([\w.\-]+)\s*\}\}")


def _prop(item: dict, name: str) -> str | None:
    """Return the first unclassed prop value with the given name."""
    for prop in item.get("props", []):
        if prop["name"] == name and "class" not in prop:
            return prop["value"]
    return None


def _describe_param(param: dict) -> str:
    if "select" in param:
        choices = "; ".join(param["select"].get("choice", []))
        return f"[Selection: {choices}]"

    label = param.get("label") or param["id"]
    return f"[Assignment: organization-defined {label}]"


def _fill_params(text: str, params: dict[str, str]) -> str:
    # Selection choices can themselves contain params, so substitute twice.
    for _ in range(2):
        text = PARAM_PATTERN.sub(
            lambda match: params.get(match.group(1), "[Assignment]"),
            text,
        )
    return text


def _flatten_parts(part: dict, depth: int = 0) -> list[str]:
    """Flatten a nested statement part into labelled lines."""
    lines = []
    label = _prop(part, "label")
    prose = part.get("prose", "")

    if prose:
        prefix = f"{label} " if label else ""
        lines.append("  " * depth + prefix + prose)

    for child in part.get("parts", []):
        child_depth = depth + 1 if prose else depth
        lines.extend(_flatten_parts(child, child_depth))

    return lines


def _iter_controls(controls: list[dict]):
    for control in controls:
        yield control
        yield from _iter_controls(control.get("controls", []))


def load_controls(path: str | Path = CATALOG_PATH) -> list[dict]:
    """Load active SP 800-53 controls and enhancements from an OSCAL catalog."""
    catalog = json.loads(Path(path).read_text(encoding="utf-8"))["catalog"]

    params = {}
    for group in catalog["groups"]:
        for control in _iter_controls(group.get("controls", [])):
            for param in control.get("params", []):
                params[param["id"]] = _describe_param(param)

    controls = []
    for group in catalog["groups"]:
        for control in _iter_controls(group.get("controls", [])):
            if _prop(control, "status") == "withdrawn":
                continue

            parts = {part["name"]: part for part in control.get("parts", [])}
            statement = "\n".join(_flatten_parts(parts.get("statement", {})))
            guidance = parts.get("guidance", {}).get("prose", "")

            controls.append(
                {
                    "control_id": _prop(control, "label"),
                    "title": control["title"],
                    "family": group["title"],
                    "statement": _fill_params(statement, params),
                    "guidance": _fill_params(guidance, params),
                }
            )

    return controls


def render_control(control: dict, include_guidance: bool = True) -> str:
    text = f"{control['control_id']} {control['title']}\n{control['statement']}"

    if include_guidance and control["guidance"]:
        text += f"\n\nDiscussion: {control['guidance']}"

    return text


def control_chunks(
    controls: list[dict],
    include_guidance: bool = True,
) -> list[dict]:
    """Structure-aware chunking: one chunk per control or enhancement."""
    return [
        {
            "chunk_id": control["control_id"],
            "source": control["control_id"],
            "control_ids": [control["control_id"]],
            "text": render_control(control, include_guidance),
        }
        for control in controls
    ]


def word_chunks(
    controls: list[dict],
    chunk_size: int = 100,
    overlap: int = 20,
) -> list[dict]:
    """Baseline chunking: fixed word windows over each family's full text.

    This mirrors chunking a PDF of the catalog. Each chunk records every
    control whose words it overlaps, so it can still be scored by control ID.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and < chunk_size")

    families: dict[str, list[dict]] = {}
    for control in controls:
        families.setdefault(control["family"], []).append(control)

    chunks = []
    step = chunk_size - overlap

    for family, members in families.items():
        words = []
        owners = []
        for control in members:
            control_words = render_control(control).split()
            words.extend(control_words)
            owners.extend([control["control_id"]] * len(control_words))

        for index, start in enumerate(range(0, len(words), step)):
            window = slice(start, start + chunk_size)
            if not words[window]:
                break

            chunks.append(
                {
                    "chunk_id": f"{family}:{index}",
                    "source": family,
                    "control_ids": list(dict.fromkeys(owners[window])),
                    "text": " ".join(words[window]),
                }
            )

    return chunks
