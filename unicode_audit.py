from pathlib import Path
from collections import Counter
import unicodedata as ud
import pandas as pd

root = Path(__file__).resolve().parent
files = ["Dictionnaire Ncobnkn.xlsx", "Expressions Medumba.xlsx"]
texts: list[str] = []

for filename in files:
    path = root / filename
    if "Expressions" in filename:
        dataframe = pd.read_excel(path, header=None)
    else:
        dataframe = pd.read_excel(path, sheet_name="Feuil1")

    for column in dataframe.columns:
        values = dataframe[column].dropna().astype(str).tolist()
        texts.extend(values)

joined = "\n".join(texts)
joined.encode("utf-8").decode("utf-8")

non_ascii_chars = sorted({character for character in joined if ord(character) > 127})
counter = Counter(character for character in joined if ord(character) > 127)

print(f"Total text chars: {len(joined)}")
print(f"Unique non-ASCII chars: {len(non_ascii_chars)}")
print("Sample non-ASCII:", "".join(non_ascii_chars[:120]))
print("Top code points:")
for character, count in counter.most_common(40):
    name = ud.name(character, "UNKNOWN")
    print(f"{character}\tU+{ord(character):04X}\t{name}\t{count}")

nfc_diff = sum(1 for text in texts if text != ud.normalize("NFC", text))
nfd_diff = sum(1 for text in texts if text != ud.normalize("NFD", text))
print(f"Rows differing from NFC: {nfc_diff}")
print(f"Rows differing from NFD: {nfd_diff}")
print("UTF-8 roundtrip: OK")
