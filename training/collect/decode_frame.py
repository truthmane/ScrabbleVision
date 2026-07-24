"""Decodes a captured video-frame data URL into a real image file.

Written for one specific wrinkle: capturing a frame via a browser-automation
tool sometimes returns a result too large to inline, saving it instead to a
JSON file shaped like `[{"type": "text", "text": "..."}]` where the text
field is itself a JSON-double-encoded string containing the
`data:image/jpeg;base64,...` data URL (not just the raw data URL) --
`json.load` alone doesn't unwrap that inner layer, hence the regex extraction
below rather than a second `json.loads`. If you already have a plain data
URL (e.g. from a browser console session), just base64-decode it directly;
this script's specific job is unwrapping that saved-tool-result format.
"""
from __future__ import annotations

import argparse
import base64
import json
import re
from pathlib import Path


def decode_frame(tool_result_path: Path, out_path: Path) -> int:
    with open(tool_result_path) as f:
        data = json.load(f)

    text = data[0]["text"]
    match = re.search(r"data:image/jpeg;base64,([A-Za-z0-9+/=]+)", text)
    if not match:
        raise ValueError(f"no data:image/jpeg;base64,... payload found in {tool_result_path}")

    raw = base64.b64decode(match.group(1))
    out_path.write_bytes(raw)
    return len(raw)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tool_result_path", type=Path)
    parser.add_argument("out_path", type=Path)
    args = parser.parse_args()

    size = decode_frame(args.tool_result_path, args.out_path)
    print(f"wrote {args.out_path} ({size} bytes)")


if __name__ == "__main__":
    main()
