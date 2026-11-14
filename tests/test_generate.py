import json
import re
from collections import defaultdict

from claims.generate import main, write_dataset


def snapshot(root):
    return {
        str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()
    }


def test_same_seed_gives_byte_identical_output(tmp_path):
    main(["--claims", "40", "--seed", "7", "--out", str(tmp_path / "a")])
    main(["--claims", "40", "--seed", "7", "--out", str(tmp_path / "b")])
    first, second = snapshot(tmp_path / "a"), snapshot(tmp_path / "b")
    assert first.keys() == second.keys()
    assert first == second


def test_rerun_into_same_folder_is_identical(tmp_path):
    write_dataset(12, 3, tmp_path)
    before = snapshot(tmp_path)
    write_dataset(12, 3, tmp_path)
    assert snapshot(tmp_path) == before


def test_different_seed_gives_different_data(tmp_path):
    write_dataset(10, 1, tmp_path / "a")
    write_dataset(10, 2, tmp_path / "b")
    assert snapshot(tmp_path / "a") != snapshot(tmp_path / "b")


def test_layout_matches_assignment(dataset, labels):
    folders = sorted((dataset / "claims").iterdir())
    assert len(folders) == len(labels) == 40
    for folder, label in zip(folders, labels, strict=True):
        files = sorted(folder.glob("*.txt"))
        assert 2 <= len(files) <= 4
        assert folder.name == label["claim_id"]
        for f in files:
            assert re.fullmatch(r"[0-9a-f]{8}\.txt", f.name), "file names must be opaque"
            assert "SPECIMEN" in f.read_text()

    policy_keys = {
        "certificate_no", "participant_name", "participant_ic", "product", "start_date",
        "end_date", "sum_covered",
    }  # fmt: skip
    for line in (dataset / "policies.jsonl").read_text().splitlines():
        policy = json.loads(line)
        assert set(policy) == policy_keys
        assert policy["product"] in {"MOTOR", "MEDICAL"}


def test_coverage_of_types_layouts_noise_and_scenarios(dataset, labels):
    docs = [d for lab in labels for d in lab["documents"]]
    titles = defaultdict(set)
    for lab in labels:
        for d in lab["documents"]:
            text = (dataset / "claims" / lab["claim_id"] / f"{d['doc_id']}.txt").read_text()
            if not d["ocr_noise"]:
                titles[d["doc_type"]].add(text.splitlines()[0])
    assert set(titles) == {
        "claim_form", "hospital_bill", "discharge_summary", "repair_estimate", "police_report",
    }  # fmt: skip
    assert all(len(t) >= 2 for t in titles.values()), "at least two layouts per type"
    assert sum(d["ocr_noise"] for d in docs) / len(docs) >= 0.25

    scenarios = {(lab["scenario"], lab["product"]) for lab in labels}
    assert {("clean", "MEDICAL"), ("clean", "MOTOR")} <= scenarios
    codes = {c for lab in labels for c in lab["expected_reason_codes"]}
    assert {"CERT_NOT_FOUND", "MISSING_DOCUMENT", "DATE_ORDER_INVALID", "AMOUNT_MISMATCH"} <= codes


def test_answers_are_not_in_document_text(dataset, labels):
    for lab in labels:
        for d in lab["documents"]:
            text = (dataset / "claims" / lab["claim_id"] / f"{d['doc_id']}.txt").read_text()
            assert lab["scenario"] not in text
            assert d["doc_type"] not in text
