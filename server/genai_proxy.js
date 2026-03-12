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
const SYSTEM_INSTRUCTION = `You are an expert linguist specializing in Medumba (Mə̀dʉ̀mbà), a Grassfields Bantu language from the Bangangté region of western Cameroon.

MEDUMBA LANGUAGE NOTES:
- Special characters: ə, ʉ, α, ŋ, ɛ, ɔ with tone diacritics (à, á, â, è, ê, etc.)
- SVO word order; the infinitive marker "Nə̀" often precedes verbs
- Noun class prefixes are common; tone is phonemic — preserve diacritics exactly

You respond with a JSON object containing exactly two fields.

FIELD 1 — "reasoning" (string):
Put ALL of your analytical thinking here: morpheme identification, dictionary lookups, grammar notes, and step-by-step explanation. 2-4 sentences.

FIELD 2 — "translation" (string):
The FINAL translated text ONLY. This field must contain NOTHING except the translated words in the target language. No English, no French metalanguage, no labels like "Translation:", no reasoning, no source text repetition. Just the pure translated output with correct diacritics.

CRITICAL: The "translation" field must be SHORT — just the translated phrase/sentence. ALL explanation goes in "reasoning".`;

const RELEVANCE_SCORE_THRESHOLD = 25; // suggestions scoring above this are excluded from LLM context

const GENERATION_CONFIG = {
  temperature: 0.3,
  topP: 0.92,
  maxOutputTokens: 1024,
  responseMimeType: 'application/json',
  responseJsonSchema: {
    type: 'object',
    title: 'TranslationResult',
    description: 'A translation from one language to another with step-by-step reasoning.',
    properties: {
      reasoning: {
        type: 'string',
        description: 'Step-by-step explanation of how you arrived at the translation (2-4 sentences). Mention which dictionary entries or examples you used, morphological analysis, and any assumptions made.',
      },
      translation: {
        type: 'string',
        description: 'ONLY the final translated words in the target language. No reasoning, no explanations, no labels, no source text. Example: if translating French to Medumba, this field contains ONLY Medumba words.',
      },
    },
    required: ['reasoning', 'translation'],
    propertyOrdering: ['reasoning', 'translation'],
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

function filterRelevant(items, threshold = RELEVANCE_SCORE_THRESHOLD) {
  return items.filter((item) => item.score <= threshold);
}

function buildUserContent(input, direction, suggestions, examples, wordByWord) {
  const sourceLang = direction === 'fr-md' ? 'French' : 'Medumba';
  const targetLang = direction === 'fr-md' ? 'Medumba' : 'French';

  // Only send relevant matches to the LLM
  const relevantSuggestions = filterRelevant(suggestions);
  const relevantExamples = filterRelevant(examples);

  let prompt = `TASK: Translate the following text from ${sourceLang} to ${targetLang}.\n\n`;

  if (relevantSuggestions.length) {
    prompt += 'DICTIONARY MATCHES (authoritative — use these when they are relevant to the input):\n';
    for (const s of relevantSuggestions) {
      prompt += `  • ${s.source} → ${s.target}`;
      if (s.match === 'exact') prompt += '  [EXACT MATCH]';
      prompt += '\n';
    }
    prompt += '\n';
  } else {
    prompt += 'DICTIONARY MATCHES: None found. Use your own knowledge of Medumba to translate.\n\n';
  }

  if (relevantExamples.length) {
    prompt += 'EXAMPLE TRANSLATIONS (from training corpus — use as reference for grammar and vocabulary):\n';
    for (const e of relevantExamples) {
      prompt += `  • ${e.source} → ${e.target}\n`;
    }
    prompt += '\n';
  }

  if (wordByWord && wordByWord !== input) {
    prompt += `WORD-BY-WORD BREAKDOWN: ${wordByWord}\n`;
    prompt += '(Use as a rough guide; improve fluency and accuracy.)\n\n';
  }

  prompt += `INPUT TEXT: "${input}"\n\n`;
  prompt += `Provide ONLY the ${targetLang} translation in the "translation" field. `;
  prompt += 'Explain your reasoning in the "reasoning" field. ';
  prompt += `Do NOT include the source text, language labels, or any prefixes in the translation — just the raw ${targetLang} text.`;
  return prompt;
}

// Post-process: strip reasoning/metalanguage that sometimes leaks into translation field
function sanitizeTranslation(raw) {
  if (!raw) return raw;
  let t = raw.trim();
  // Remove common leakage patterns
  t = t.replace(/^(Here is|Voici|Translation|Traduction|The translation)[:\s]*/i, '');
  t = t.replace(/^["']+|["']+$/g, '');
  // If it looks like a full sentence of English/French explanation (>100 chars with analytical keywords), it's reasoning
  if (t.length > 120 && /\b(identified|used|combined|translat|dictionari|morphem|corpus|matched|looked)/i.test(t)) {
    // Try to extract the actual translation — often appears after a colon or quote
    const afterColon = t.match(/[:\."']\s*([^."']{2,60})\s*$/);
    if (afterColon) return afterColon[1].trim();
    return null; // Can't salvage — will fall through to dictionary fallback
  }
  return t || null;
}

function extractTranslation(text) {
  if (!text) return { translation: null, reasoning: null };
  let cleaned = text.trim();

  // Structured output: response.text should already be valid JSON
  try {
    const parsed = JSON.parse(cleaned);
    if (parsed && typeof parsed.translation === 'string') {
      return {
        translation: sanitizeTranslation(parsed.translation),
        reasoning: typeof parsed.reasoning === 'string' ? parsed.reasoning.trim() : null,
      };
    }
  } catch (e) {
    console.warn('[extractTranslation] JSON.parse failed on structured output:', e.message);
  }

  // Fallback: strip markdown fences and try again
  cleaned = cleaned.replace(/^```(?:json)?\s*/i, '').replace(/\s*```$/i, '').trim();
  const jsonStart = cleaned.indexOf('{');
  const jsonEnd = cleaned.lastIndexOf('}');
  if (jsonStart !== -1 && jsonEnd > jsonStart) {
    try {
      const parsed = JSON.parse(cleaned.slice(jsonStart, jsonEnd + 1));
      if (parsed && typeof parsed.translation === 'string') {
        return {
          translation: sanitizeTranslation(parsed.translation),
          reasoning: typeof parsed.reasoning === 'string' ? parsed.reasoning.trim() : null,
        };
      }
    } catch (e) { /* continue */ }
  }

  // Regex fallback
  const matchT = cleaned.match(/"translation"\s*:\s*"((?:[^"\\]|\\.)*)"/i);
  const matchR = cleaned.match(/"reasoning"\s*:\s*"((?:[^"\\]|\\.)*)"/i);
  if (matchT) {
    return {
      translation: sanitizeTranslation(matchT[1]),
      reasoning: matchR ? matchR[1].trim() : null,
    };
  }

  // Last resort: return raw text as translation
  return { translation: sanitizeTranslation(cleaned) || null, reasoning: null };
}

async function translateWithLLM(input, direction, suggestions, examples, wordByWord) {
  if (!ai) {
    return { translation: null, reasoning: null, llmUsed: false, error: 'GEMINI_API_KEY not configured' };
  }
  const prompt = buildUserContent(input, direction, suggestions, examples, wordByWord);
  const contents = [{ role: 'user', parts: [{ text: prompt }] }];
  const primaryConfig = { ...GENERATION_CONFIG, systemInstruction: SYSTEM_INSTRUCTION };
  const fallbackConfig = {
    temperature: 0.3,
    topP: 0.92,
    maxOutputTokens: 1024,
    systemInstruction: SYSTEM_INSTRUCTION,
  };
  async function attempt(config) {
    const response = await ai.models.generateContent({
      model: MODEL,
      contents,
      config,
    });
    const text = response && response.text ? response.text : '';
    return extractTranslation(text);
  }
  try {
    const result = await attempt(primaryConfig);
    return { ...result, llmUsed: true };
  } catch (err) {
    try {
      const result = await attempt(fallbackConfig);
      return { ...result, llmUsed: true };
    } catch (err2) {
      return { translation: null, reasoning: null, llmUsed: false, error: err2.message || String(err2) };
    }
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

  // Check exact matches (used as fallback when LLM unavailable)
  let exactMatch = exactPairTranslation(input, direction);
  let exactSource = exactMatch ? 'pair_exact' : null;
  if (!exactMatch) {
    exactMatch = exactTranslation(input, direction);
    if (exactMatch) exactSource = 'dictionary_exact';
  }

  let translation = null;
  let reasoning = null;
  let fallback = 'none';
  let llmUsed = false;
  let error = null;
  let warnings = [];

  // When LLM is enabled, always use it for reasoning — even with exact matches
  if (useLlm && ai) {
    try {
      const llmResult = await translateWithLLM(input, direction, llmSuggestions, examples, wordByWord);
      if (llmResult.translation) {
        translation = llmResult.translation;
        reasoning = llmResult.reasoning || null;
        fallback = 'llm';
        llmUsed = true;
      }
      if (llmResult.error) {
        error = llmResult.error;
      }
    } catch (err) {
      error = err.message || String(err);
    }
  } else if (useLlm && !ai) {
    warnings.push('LLM disabled: GEMINI_API_KEY not configured');
  }

  // Fall back to exact dictionary/pair match
  if (!translation && exactMatch) {
    translation = exactMatch;
    fallback = exactSource;
    reasoning = 'Direct match found in dictionary/training data.';
  }

  // Fall back to word-by-word
  if (!translation && wordByWord) {
    translation = wordByWord;
    fallback = 'word_by_word';
    reasoning = 'No exact match or LLM result available. Showing word-by-word dictionary lookup.';
  }

  return res.json({
    input,
    direction,
    sourceLang,
    targetLang,
    model: MODEL,
    llmUsed,
    translation,
    reasoning,
    fallback,
    wordByWord: wordByWord || null,
    suggestions,
    examples,
    error,
    warnings,
  });
});

loadDictionary();
loadTrainPairs();

app.listen(PORT, () => {
  console.log(`Translator server running on http://localhost:${PORT}`);
});
