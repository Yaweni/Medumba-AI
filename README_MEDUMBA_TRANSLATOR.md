# Medumba Translator (RAG + Gemini optional)

This repo contains a translator UI and a lightweight server that performs retrieval-augmented translation using your Medumba dictionary. The server can optionally call Gemini for fluent translations.

Files:
- `site/index.html`, `site/styles.css`, `site/app.js` - frontend
- `server/genai_proxy.js` - RAG server + Gemini proxy
- `tools/build_dict.py` - converter script that creates `site/data/dictionary.json` from `outputs/pairs_train.jsonl`

Quick start:

1. Build the JSON dictionary:

```bash
python tools/build_dict.py "c:\\Users\\yaweh\\OneDrive\\Documents\\Medumba AI\\outputs\\pairs_train.jsonl"
```

This writes `site/data/dictionary.json`.

2. Run the server (serves the UI and API):

```bash
cd server
npm install
# copy server/.env.example to server/.env and fill in GEMINI_API_KEY
npm start
```

Then open http://localhost:5000 in your browser.

Notes:
- If you do not set `GEMINI_API_KEY`, the server still returns dictionary suggestions and exact matches, but LLM translations are disabled.
- Never embed API keys in client-side code. Keep the key in server environment variables.
- GitHub Pages cannot host server proxies. To deploy with LLM enabled, use a provider that supports serverless functions or a Node server, and store `GEMINI_API_KEY` as a secret there.
