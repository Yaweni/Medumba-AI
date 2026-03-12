from __future__ import annotations

from pathlib import Path
import json

ROOT = Path(__file__).resolve().parent

xlsx_files = sorted(ROOT.glob("*.xlsx"))
pdf_files = sorted(ROOT.glob("*.pdf"))

report: dict = {
    "root": str(ROOT),
    "xlsx_files": [],
    "pdf_files": [],
}

try:
    import pandas as pd
except Exception as e:
    raise SystemExit(f"pandas import failed: {e}")

for file in xlsx_files:
    file_info: dict = {"file": file.name, "sheets": []}
    try:
        excel = pd.ExcelFile(file)
        for sheet in excel.sheet_names:
            try:
                df = pd.read_excel(file, sheet_name=sheet)
                columns = [str(c) for c in df.columns.tolist()]
                missing = df.isna().sum().to_dict()
                sample = df.head(3).fillna("").astype(str).to_dict(orient="records")
                file_info["sheets"].append(
                    {
                        "sheet": sheet,
                        "rows": int(len(df)),
                        "cols": int(len(df.columns)),
                        "columns": columns,
                        "missing_by_column": {str(k): int(v) for k, v in missing.items()},
                        "sample_rows": sample,
                    }
                )
            except Exception as e:
                file_info["sheets"].append({"sheet": sheet, "error": str(e)})
    except Exception as e:
        file_info["error"] = str(e)
    report["xlsx_files"].append(file_info)

for file in pdf_files:
    info: dict = {"file": file.name}
    try:
        from pypdf import PdfReader  # type: ignore

        reader = PdfReader(str(file))
        info["pages"] = len(reader.pages)
        snippets = []
        for idx, page in enumerate(reader.pages[:3]):
            text = page.extract_text() or ""
            snippets.append(
                {
                    "page": idx + 1,
                    "chars": len(text),
                    "snippet": text[:400].replace("\n", " "),
                }
            )
        info["sample_pages"] = snippets
    except Exception as e:
        info["error"] = str(e)
    report["pdf_files"].append(info)

out_path = ROOT / "corpus_profile_report.json"
out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

print(f"Wrote report to: {out_path}")
print(json.dumps(report, ensure_ascii=False, indent=2)[:12000])
