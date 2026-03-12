from __future__ import annotations

import json
import random
import re
import unicodedata as ud
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# Conservative normalization: unify spacing and punctuation variants while preserving meaning.
REPLACEMENTS = {
    "\u00A0": " ",  # NBSP
    "\u2019": "ꞌ",  # right single quote -> saltillo style apostrophe
    "\u2018": "ꞌ",  # left single quote -> saltillo style apostrophe
    "\u02BC": "ꞌ",  # modifier apostrophe -> saltillo style apostrophe
    "'": "ꞌ",       # ascii apostrophe -> saltillo style apostrophe
    "\t": " ",
}


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    if text.lower() == "nan":
        return ""
    text = text.strip()
    if not text:
        return ""
    for src, dst in REPLACEMENTS.items():
        text = text.replace(src, dst)
    text = re.sub(r"\s+", " ", text)
    text = ud.normalize("NFC", text)
    return text.strip()


def clean_word_headword(text: str) -> str:
    # Remove dictionary indexing markers like "(1/6)" or "(1)" at end.
    cleaned = re.sub(r"\s*\([^)]*\)\s*$", "", text).strip()
    return cleaned or text


def load_dictionary_rows() -> list[dict[str, Any]]:
    dictionary_file = ROOT / "Dictionnaire Ncobnkn.xlsx"
    dataframe = pd.read_excel(dictionary_file, sheet_name="Feuil1")

    output: list[dict[str, Any]] = []
    for _, row in dataframe.iterrows():
        mot_raw = normalize_text(row.get("Mot", ""))
        fr_raw = normalize_text(row.get("Francais", ""))
        if not mot_raw or not fr_raw:
            continue

        record = {
            "source": "dictionary",
            "medumba_raw": mot_raw,
            "medumba_norm": clean_word_headword(mot_raw),
            "medumba_alt": normalize_text(row.get("Autre forme", "")),
            "french_raw": fr_raw,
            "french_norm": fr_raw,
            "english_raw": normalize_text(row.get("Anglais", "")),
            "english_norm": normalize_text(row.get("Anglais", "")),
            "pos_raw": normalize_text(row.get("Nature du mot", "")),
            "definition_raw": normalize_text(row.get("Definition", "")),
        }
        output.append(record)
    return output


def load_expression_rows() -> list[dict[str, Any]]:
    expressions_file = ROOT / "Expressions Medumba.xlsx"

    # header=None to avoid accidentally taking first sentence as header.
    dataframe = pd.read_excel(expressions_file, header=None)
    if dataframe.shape[1] < 2:
        return []

    dataframe = dataframe.iloc[:, :2]
    dataframe.columns = ["french", "medumba"]

    output: list[dict[str, Any]] = []
    for _, row in dataframe.iterrows():
        french_raw = normalize_text(row.get("french", ""))
        medumba_raw = normalize_text(row.get("medumba", ""))
        if not french_raw or not medumba_raw:
            continue
        output.append(
            {
                "source": "expressions",
                "medumba_raw": medumba_raw,
                "medumba_norm": medumba_raw,
                "medumba_alt": "",
                "french_raw": french_raw,
                "french_norm": french_raw,
                "english_raw": "",
                "english_norm": "",
                "pos_raw": "",
                "definition_raw": "",
            }
        )
    return output


def deduplicate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str, str]] = set()
    deduped: list[dict[str, Any]] = []

    for row in rows:
        key = (row["source"], row["medumba_norm"], row["french_norm"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped


def build_training_pairs(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    pairs: list[dict[str, str]] = []
    pair_id = 1

    for row in rows:
        medumba = row["medumba_norm"]
        french = row["french_norm"]

        if medumba and french:
            pairs.append(
                {
                    "id": f"pair_{pair_id:06d}",
                    "task": "fr_to_medumba",
                    "instruction": "Translate French to Medumba.",
                    "input": french,
                    "output": medumba,
                    "source": row["source"],
                }
            )
            pair_id += 1

            pairs.append(
                {
                    "id": f"pair_{pair_id:06d}",
                    "task": "medumba_to_fr",
                    "instruction": "Translate Medumba to French.",
                    "input": medumba,
                    "output": french,
                    "source": row["source"],
                }
            )
            pair_id += 1

    return pairs


def split_dataset(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    rows_copy = rows[:]
    random.shuffle(rows_copy)

    n = len(rows_copy)
    n_train = int(n * 0.85)
    n_val = int(n * 0.10)

    train = rows_copy[:n_train]
    val = rows_copy[n_train : n_train + n_val]
    test = rows_copy[n_train + n_val :]
    return train, val, test


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def char_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    text = "\n".join(
        [
            row["medumba_norm"]
            for row in rows
            if row.get("medumba_norm")
        ]
        + [
            row["french_norm"]
            for row in rows
            if row.get("french_norm")
        ]
    )

    non_ascii = Counter(ch for ch in text if ord(ch) > 127)
    top = [
        {
            "char": ch,
            "codepoint": f"U+{ord(ch):04X}",
            "name": ud.name(ch, "UNKNOWN"),
            "count": count,
        }
        for ch, count in non_ascii.most_common(80)
    ]
    return {
        "total_chars": len(text),
        "unique_non_ascii": len(non_ascii),
        "top_non_ascii": top,
    }


def suspicious_character_samples(rows: list[dict[str, Any]]) -> dict[str, list[str]]:
    suspicious_patterns = {
        "greek_alpha": re.compile(r"[αὰ]"),
        "dotted_circle": re.compile(r"◌"),
        "euro_sign": re.compile(r"€"),
        "en_dash": re.compile(r"–"),
    }

    samples: dict[str, list[str]] = {name: [] for name in suspicious_patterns}

    for row in rows:
        for field in ["medumba_norm", "french_norm", "definition_raw"]:
            value = row.get(field, "")
            if not value:
                continue
            for name, pattern in suspicious_patterns.items():
                if pattern.search(value) and len(samples[name]) < 25:
                    samples[name].append(value)

    return samples


def main() -> None:
    dict_rows = load_dictionary_rows()
    expr_rows = load_expression_rows()

    all_rows = deduplicate_rows(dict_rows + expr_rows)
    train_rows, val_rows, test_rows = split_dataset(all_rows)

    # Full cleaned tables
    write_jsonl(OUT_DIR / "corpus_clean_all.jsonl", all_rows)
    write_jsonl(OUT_DIR / "corpus_clean_train.jsonl", train_rows)
    write_jsonl(OUT_DIR / "corpus_clean_val.jsonl", val_rows)
    write_jsonl(OUT_DIR / "corpus_clean_test.jsonl", test_rows)

    # Instruction-style training pairs
    train_pairs = build_training_pairs(train_rows)
    val_pairs = build_training_pairs(val_rows)
    test_pairs = build_training_pairs(test_rows)

    write_jsonl(OUT_DIR / "pairs_train.jsonl", train_pairs)
    write_jsonl(OUT_DIR / "pairs_val.jsonl", val_pairs)
    write_jsonl(OUT_DIR / "pairs_test.jsonl", test_pairs)

    report = {
        "seed": RANDOM_SEED,
        "counts": {
            "dictionary_rows_raw": len(dict_rows),
            "expressions_rows_raw": len(expr_rows),
            "rows_after_dedup": len(all_rows),
            "train_rows": len(train_rows),
            "val_rows": len(val_rows),
            "test_rows": len(test_rows),
            "train_pairs": len(train_pairs),
            "val_pairs": len(val_pairs),
            "test_pairs": len(test_pairs),
        },
        "normalization": {
            "unicode_normalization": "NFC",
            "replacements": REPLACEMENTS,
            "note": "Apostrophe-like characters are unified to U+A78C for consistency.",
        },
        "char_report": char_report(all_rows),
        "suspicious_char_samples": suspicious_character_samples(all_rows),
        "files_written": [
            "outputs/corpus_clean_all.jsonl",
            "outputs/corpus_clean_train.jsonl",
            "outputs/corpus_clean_val.jsonl",
            "outputs/corpus_clean_test.jsonl",
            "outputs/pairs_train.jsonl",
            "outputs/pairs_val.jsonl",
            "outputs/pairs_test.jsonl",
            "outputs/prep_report.json",
        ],
    }

    (OUT_DIR / "prep_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("Prepared corpus successfully.")
    print(json.dumps(report, ensure_ascii=False, indent=2)[:10000])


if __name__ == "__main__":
    main()
