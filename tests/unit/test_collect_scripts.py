import base64
import io
import json

import cv2
import numpy as np
import pytest
from PIL import Image

from training.collect.crop_rack import crop_rack, parse_groups
from training.collect.decode_frame import decode_frame


def test_parse_groups_splits_triples_correctly():
    groups = parse_groups(["95", "450", "B,A,T,E", "515", "785", "E,L,S"])
    assert groups == [(95, 450, ["B", "A", "T", "E"]), (515, 785, ["E", "L", "S"])]


def test_parse_groups_rejects_incomplete_triples():
    with pytest.raises(ValueError):
        parse_groups(["95", "450"])


def test_decode_frame_extracts_data_url(tmp_path):
    img = Image.new("RGB", (8, 8), (100, 150, 200))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    data_url = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()

    tool_result_path = tmp_path / "tool_result.json"
    tool_result_path.write_text(json.dumps([{"type": "text", "text": data_url}]))

    out_path = tmp_path / "out.jpg"
    size = decode_frame(tool_result_path, out_path)

    assert size > 0
    assert out_path.exists()
    decoded = Image.open(out_path)
    assert decoded.size == (8, 8)


def test_crop_rack_writes_one_file_per_letter(tmp_path):
    img = np.full((50, 700, 3), 200, dtype=np.uint8)
    image_path = tmp_path / "rack.png"
    cv2.imwrite(str(image_path), img)

    out_dir = tmp_path / "out"
    count = crop_rack(image_path, out_dir, "test", [(0, 300, ["A", "B", "C"]), (400, 700, ["D", "E"])])

    assert count == 5
    assert sorted(p.name.split("_")[0] for p in out_dir.glob("*.png")) == ["A", "B", "C", "D", "E"]
