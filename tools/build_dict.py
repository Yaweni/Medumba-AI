#!/usr/bin/env python3
"""Build site/data/dictionary.json from outputs/pairs_train.jsonl

Usage: python tools/build_dict.py /absolute/path/to/outputs/pairs_train.jsonl
If no arg provided, script will try to read from '../outputs/pairs_train.jsonl' relative to repo root.
"""
import sys, json, os

def load_jsonl(path):
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line=line.strip()
            if not line: continue
            yield json.loads(line)

def main():
    if len(sys.argv) > 1:
        inp = sys.argv[1]
    else:
        inp = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'outputs', 'pairs_train.jsonl'))

    outdir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'site', 'data'))
    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir, 'dictionary.json')

    seen = set()
    entries = []
    for obj in load_jsonl(inp):
        task = obj.get('task','')
        if task == 'fr_to_medumba':
            frm, to = 'fr', 'md'
            inp_text = obj.get('input','').strip()
            out_text = obj.get('output','').strip()
        elif task == 'medumba_to_fr':
            frm, to = 'md', 'fr'
            inp_text = obj.get('input','').strip()
            out_text = obj.get('output','').strip()
        else:
            continue
        key = (frm, inp_text.lower())
        if key in seen: continue
        seen.add(key)
        entries.append({ 'from': frm, 'to': to, 'input': inp_text, 'output': out_text })

    with open(outpath, 'w', encoding='utf-8') as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)

    print(f'Wrote {len(entries)} entries to {outpath}')

if __name__ == '__main__':
    main()
