// RAG-assisted Medumba <-> French translator (Gemini-backed)
// Usage: set GEMINI_API_KEY and run `node server/genai_proxy.js`

const fs = require('fs');
const path = require('path');
const dotenv = require('dotenv');
const express = require('express');
const cors = require('cors');
const { GoogleGenAI } = require('@google/genai');

// Load root .env first, then server/.env overrides everything (including existing env vars).
dotenv.config({ path: path.join(__dirname, '..', '.env') });
dotenv.config({ path: path.join(__dirname, '.env'), override: true });

const PORT = process.env.PORT || 5000;
const MODEL = process.env.GEMINI_MODEL || 'gemini-2.5-flash';
const DICT_PATH = process.env.DICT_PATH || path.join(__dirname, '..', 'site', 'data', 'dictionary.json');
const PAIR_PATH = process.env.PAIR_PATH || path.join(__dirname, '..', 'outputs', 'pairs_train.jsonl');
const MAX_SUGGESTIONS = 12;
const LLM_SUGGESTION_LIMIT = 6;
const LLM_EXAMPLE_LIMIT = 6;
const SYSTEM_INSTRUCTION =
  'You are a translation engine for French and Medumba. ' +
  'Return only valid JSON with a single key "translation". ' +
  'Do not include explanations, prefixes, or extra keys.';

const GENERATION_CONFIG = {
  temperature: 0.2,
  topP: 0.9,
  maxOutputTokens: 120,
  responseMimeType: 'application/json',
  responseJsonSchema: {
    type: 'object',
    properties: {
      translation: { type: 'string' },
    },
    required: ['translation'],
    additionalProperties: false,
  },
};

const app = express();
app.use(cors());
app.use(express.json({ limit: '1mb' }));
app.use(express.static(path.join(__dirname, '..', 'site')));

const apiKeyRaw = process.env.GEMINI_API_KEY || process.env.GOOGLE_API_KEY || '';
const apiKey = apiKeyRaw.trim();
if (!apiKey) {
  console.warn('GEMINI_API_KEY not set. LLM translation will be disabled.');
} else if (apiKeyRaw !== apiKey) {
  console.warn('GEMINI_API_KEY had leading/trailing whitespace. It was trimmed.');
}

const ai = apiKey ? new GoogleGenAI({ apiKey }) : null;

let dictionary = [];
let index = { 'fr-md': new Map(), 'md-fr': new Map() };
let entries = { 'fr-md': [], 'md-fr': [] };
let trainPairs = { 'fr-md': [], 'md-fr': [] };
let pairIndex = { 'fr-md': new Map(), 'md-fr': new Map() };
let wordIndex = { 'fr-md': new Map(), 'md-fr': new Map() };

function normalize(text) {
  if (!text) return '';
  return text
    .toString()
    .normalize('NFD')
    .replace(/[\u0300-\u036f]/g, '')
    .replace(/[^\p{L}\p{N}\s'-]/gu, ' ')
    .replace(/\s+/g, ' ')
    .trim()
    .toLowerCase();
}

function tokenize(normText) {
  if (!normText) return [];
  return normText
    .split(' ')
    .map((t) => t.trim())
    .filter((t) => t.length >= 2);
}

function levenshtein(a, b) {
  if (!a) return b.length;
  if (!b) return a.length;
  const m = a.length;
  const n = b.length;
  const dp = Array(n + 1).fill(0);
  for (let j = 0; j <= n; j += 1) dp[j] = j;
  for (let i = 1; i <= m; i += 1) {
    let prev = i;
    for (let j = 1; j <= n; j += 1) {
      const cur = dp[j];
      const cost = a[i - 1] === b[j - 1] ? 0 : 1;
      dp[j] = Math.min(dp[j] + 1, dp[j - 1] + 1, prev + cost);
      prev = cur;
    }
  }
  return dp[n];
}

function buildIndexes() {
  index = { 'fr-md': new Map(), 'md-fr': new Map() };
  entries = { 'fr-md': [], 'md-fr': [] };
  wordIndex = { 'fr-md': new Map(), 'md-fr': new Map() };
  for (const item of dictionary) {
    if (!item || !item.input || !item.output || !item.from) continue;
    const dir = item.from === 'fr' ? 'fr-md' : 'md-fr';
    const input = item.input.toString().trim();
    const output = item.output.toString().trim();
    const norm = normalize(input);
    if (!norm) continue;
    if (!index[dir].has(norm)) index[dir].set(norm, []);
    index[dir].get(norm).push({ input, output });
    entries[dir].push({ input, output, norm });

    // Build word-level index for single-token entries.
    const tokens = input.split(/\s+/).filter(Boolean);
    if (tokens.length === 1) {
      if (!wordIndex[dir].has(norm)) wordIndex[dir].set(norm, []);
      wordIndex[dir].get(norm).push(output);
    }
  }
}

function loadDictionary() {
  try {
    const raw = fs.readFileSync(DICT_PATH, 'utf8');
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) throw new Error('dictionary.json is not an array');
    dictionary = parsed;
    buildIndexes();
    console.log(`Loaded ${dictionary.length} dictionary entries from ${DICT_PATH}`);
  } catch (err) {
    console.warn(`Failed to load dictionary from ${DICT_PATH}: ${err.message}`);
    dictionary = [];
    buildIndexes();
  }
}

function loadTrainPairs() {
  trainPairs = { 'fr-md': [], 'md-fr': [] };
  pairIndex = { 'fr-md': new Map(), 'md-fr': new Map() };
  try {
    if (!fs.existsSync(PAIR_PATH)) {
      console.warn(`Pairs file not found at ${PAIR_PATH}`);
      return;
    }
    const raw = fs.readFileSync(PAIR_PATH, 'utf8');
    const lines = raw.split(/\r?\n/);
    const seen = new Set();
    for (const line of lines) {
      if (!line.trim()) continue;
      let obj;
      try {
        obj = JSON.parse(line);
      } catch {
        continue;
      }
      const task = obj.task;
      let dir = null;
      if (task === 'fr_to_medumba') dir = 'fr-md';
      if (task === 'medumba_to_fr') dir = 'md-fr';
      if (!dir) continue;
      const input = (obj.input || '').toString().trim();
      const output = (obj.output || '').toString().trim();
      if (!input || !output) continue;
      const norm = normalize(input);
      if (!norm) continue;
      const key = `${dir}::${norm}`;
      if (seen.has(key)) continue;
      seen.add(key);
      const entry = { input, output, norm };
      trainPairs[dir].push(entry);
      if (!pairIndex[dir].has(norm)) pairIndex[dir].set(norm, []);
      pairIndex[dir].get(norm).push(entry);
    }
    console.log(`Loaded ${trainPairs['fr-md'].length + trainPairs['md-fr'].length} training pairs from ${PAIR_PATH}`);
  } catch (err) {
    console.warn(`Failed to load pairs from ${PAIR_PATH}: ${err.message}`);
  }
}

function retrieveSuggestions(input, direction, limit = MAX_SUGGESTIONS) {
  const dir = direction === 'md-fr' ? 'md-fr' : 'fr-md';
  const qNorm = normalize(input);
  if (!qNorm) return [];

  const results = [];
  const seen = new Set();
  const tokens = tokenize(qNorm);

  function push(source, target, score, match) {
    const key = `${source}|||${target}`;
    if (seen.has(key)) return;
    seen.add(key);
    results.push({ source, target, score, match });
  }

  const exact = index[dir].get(qNorm);
  if (exact) {
    for (const hit of exact) {
      push(hit.input, hit.output, 0, 'exact');
    }
  }

  for (const token of tokens) {
    const hits = index[dir].get(token);
    if (!hits) continue;
    for (const hit of hits) {
      push(hit.input, hit.output, 6, 'token');
    }
  }

  for (const entry of entries[dir]) {
    if (!entry.norm) continue;
    let score = null;
    if (entry.norm === qNorm) score = 0;
    else if (entry.norm.startsWith(qNorm)) score = 8;
    else if (entry.norm.includes(qNorm)) score = 14;
    else if (tokens.length) {
      let overlap = 0;
      for (const t of tokens) {
        if (entry.norm.includes(t)) overlap += 1;
      }
      if (overlap > 0) {
        const ratio = overlap / tokens.length;
        score = 20 - Math.round(ratio * 8);
      }
    }

    if (score === null) {
      const maxLen = Math.max(qNorm.length, entry.norm.length);
      if (maxLen <= 40) {
        const dist = levenshtein(qNorm, entry.norm);
        const threshold = Math.max(1, Math.floor(maxLen * 0.34));
        if (dist <= threshold) score = 38 + dist;
      }
    }

    if (score !== null) push(entry.input, entry.output, score, 'fuzzy');
  }

  return results
    .sort((a, b) => a.score - b.score)
    .slice(0, limit);
}

function retrieveExamples(input, direction, limit = LLM_EXAMPLE_LIMIT) {
  const dir = direction === 'md-fr' ? 'md-fr' : 'fr-md';
  const qNorm = normalize(input);
  if (!qNorm) return [];

  const results = [];
  const seen = new Set();
  const tokens = tokenize(qNorm);

  function push(item, score, match) {
    const key = `${item.input}|||${item.output}`;
    if (seen.has(key)) return;
    seen.add(key);
    results.push({ source: item.input, target: item.output, score, match });
  }

  const exact = pairIndex[dir].get(qNorm);
  if (exact) {
    for (const hit of exact) push(hit, 0, 'exact');
  }

  for (const entry of trainPairs[dir]) {
    if (!entry.norm) continue;
    let score = null;
    if (entry.norm === qNorm) score = 0;
    else if (entry.norm.startsWith(qNorm)) score = 8;
    else if (entry.norm.includes(qNorm)) score = 14;
    else if (tokens.length) {
      let overlap = 0;
      for (const t of tokens) {
        if (entry.norm.includes(t)) overlap += 1;
      }
      if (overlap > 0) {
        const ratio = overlap / tokens.length;
        score = 22 - Math.round(ratio * 8);
      }
    }

    if (score === null) {
      const maxLen = Math.max(qNorm.length, entry.norm.length);
      if (maxLen <= 60) {
        const dist = levenshtein(qNorm, entry.norm);
        const threshold = Math.max(1, Math.floor(maxLen * 0.3));
        if (dist <= threshold) score = 40 + dist;
      }
    }

    if (score !== null) push(entry, score, 'fuzzy');
  }

  return results.sort((a, b) => a.score - b.score).slice(0, limit);
}

function exactTranslation(input, direction) {
  const dir = direction === 'md-fr' ? 'md-fr' : 'fr-md';
  const qNorm = normalize(input);
  const exact = index[dir].get(qNorm);
  if (!exact || exact.length === 0) return null;
  return exact[0].output;
}

function exactPairTranslation(input, direction) {
  const dir = direction === 'md-fr' ? 'md-fr' : 'fr-md';
  const qNorm = normalize(input);
  const exact = pairIndex[dir].get(qNorm);
  if (!exact || exact.length === 0) return null;
  return exact[0].output;
}

function applyCase(source, target) {
  if (!target) return target;
  if (!source) return target;
  if (source.toUpperCase() === source && source.toLowerCase() !== source) {
    return target.toUpperCase();
  }
  const first = source[0];
  const rest = source.slice(1);
  if (first && first.toUpperCase() === first && rest.toLowerCase() === rest) {
    return target[0] ? target[0].toUpperCase() + target.slice(1) : target;
  }
  return target;
}

function wordByWordTranslate(input, direction) {
  const dir = direction === 'md-fr' ? 'md-fr' : 'fr-md';
  if (!input) return '';
  const parts = input.split(/(\s+|[^\p{L}\p{N}']+)/u);
  const out = parts.map((part) => {
    if (!part) return part;
    if (/^\s+$/.test(part)) return part;
    if (/^[^\p{L}\p{N}']+$/u.test(part)) return part;
    const norm = normalize(part);
    if (!norm) return part;
    const candidates = wordIndex[dir].get(norm);
    if (!candidates || candidates.length === 0) return part;
    const chosen = candidates[0];
    return applyCase(part, chosen);
  });
  return out.join('');
}

function buildUserContent(input, direction, suggestions, examples) {
  const sourceLang = direction === 'fr-md' ? 'French' : 'Medumba';
  const targetLang = direction === 'fr-md' ? 'Medumba' : 'French';
  let prompt = '';
  prompt += `Translate from ${sourceLang} to ${targetLang}.\n`;
  prompt += 'Use dictionary suggestions as authoritative when they apply.\n';
  prompt += 'Return one best translation.\n';
  if (suggestions.length) {
    prompt += 'Dictionary suggestions:\n';
    for (const s of suggestions) {
      prompt += `- ${s.source} -> ${s.target}\n`;
    }
  } else {
    prompt += 'Dictionary suggestions: none\n';
  }
  if (examples.length) {
    prompt += 'Example translations:\n';
    for (const e of examples) {
      prompt += `- ${e.source} -> ${e.target}\n`;
    }
  }
  prompt += `User input: ${input}\n`;
  return prompt;
}

function extractTranslation(text) {
  if (!text) return null;
  let cleaned = text.trim();
  cleaned = cleaned.replace(/^```(?:json)?/i, '').replace(/```$/i, '').trim();
  if (!cleaned) return null;

  const jsonStart = cleaned.indexOf('{');
  const jsonEnd = cleaned.lastIndexOf('}');
  if (jsonStart !== -1 && jsonEnd > jsonStart) {
    const candidate = cleaned.slice(jsonStart, jsonEnd + 1);
    try {
      const parsed = JSON.parse(candidate);
      if (parsed && typeof parsed.translation === 'string') {
        return parsed.translation.trim();
      }
    } catch (err) {
      // fall through to regex/text handling
    }
  }

  const match = cleaned.match(/"translation"\s*:\s*"([^"]+)"/i);
  if (match) return match[1].trim();

  const firstLine = cleaned.split('\n').map((line) => line.trim()).find(Boolean) || cleaned;
  return firstLine.replace(/^"+|"+$/g, '').trim();
}

async function translateWithLLM(input, direction, suggestions, examples) {
  if (!ai) {
    return { translation: null, llmUsed: false, error: 'GEMINI_API_KEY not configured' };
  }
  const prompt = buildUserContent(input, direction, suggestions, examples);
  const contents = [{ role: 'user', parts: [{ text: prompt }] }];
  const primaryConfig = { ...GENERATION_CONFIG, systemInstruction: SYSTEM_INSTRUCTION };
  const fallbackConfig = {
    temperature: 0.2,
    topP: 0.9,
    maxOutputTokens: 120,
    systemInstruction: SYSTEM_INSTRUCTION,
  };
  try {
    const response = await ai.models.generateContent({
      model: MODEL,
      contents,
      config: primaryConfig,
    });
    const text = response && response.text ? response.text : '';
    const extracted = extractTranslation(text);
    return { translation: extracted || null, llmUsed: true };
  } catch (err) {
    const response = await ai.models.generateContent({
      model: MODEL,
      contents,
      config: fallbackConfig,
    });
    const text = response && response.text ? response.text : '';
    const extracted = extractTranslation(text);
    return { translation: extracted || null, llmUsed: true };
  }
}

app.get('/api/health', (req, res) => {
  res.json({
    ok: true,
    model: MODEL,
    llmAvailable: Boolean(ai),
    dictionaryEntries: dictionary.length,
  });
});

app.post('/api/translate', async (req, res) => {
  const body = req.body || {};
  const input = body.input ? body.input.toString() : '';
  const direction = body.direction === 'md-fr' ? 'md-fr' : 'fr-md';
  const useLlm = body.useLlm !== false;
  const maxSuggestions = Number.isFinite(body.maxSuggestions)
    ? Math.max(1, Math.min(30, body.maxSuggestions))
    : MAX_SUGGESTIONS;

  if (!input.trim()) {
    return res.status(400).json({ error: 'input is required' });
  }

  const sourceLang = direction === 'fr-md' ? 'French' : 'Medumba';
  const targetLang = direction === 'fr-md' ? 'Medumba' : 'French';
  const suggestions = retrieveSuggestions(input, direction, maxSuggestions);
  const llmSuggestions = suggestions.slice(0, LLM_SUGGESTION_LIMIT);
  const examples = retrieveExamples(input, direction, LLM_EXAMPLE_LIMIT);
  const wordByWord = wordByWordTranslate(input, direction);

  let translation = exactPairTranslation(input, direction);
  let fallback = translation ? 'pair_exact' : 'none';
  if (!translation) {
    translation = exactTranslation(input, direction);
    if (translation) fallback = 'dictionary_exact';
  }
  let llmUsed = false;
  let error = null;
  let warnings = [];

  if (useLlm && ai && !translation) {
    try {
      const llmResult = await translateWithLLM(input, direction, llmSuggestions, examples);
      if (llmResult.translation) {
        translation = llmResult.translation;
        fallback = 'llm';
        llmUsed = true;
      }
    } catch (err) {
      error = err.message || String(err);
    }
  } else if (useLlm && !ai) {
    warnings.push('LLM disabled: GEMINI_API_KEY not configured');
  }

  if (!translation && wordByWord) {
    translation = wordByWord;
    fallback = 'word_by_word';
  }

  return res.json({
    input,
    direction,
    sourceLang,
    targetLang,
    model: MODEL,
    llmUsed,
    translation,
    fallback,
    wordByWord,
    suggestions,
    error,
    warnings,
  });
});

loadDictionary();
loadTrainPairs();

app.listen(PORT, () => {
  console.log(`Translator server running on http://localhost:${PORT}`);
});
