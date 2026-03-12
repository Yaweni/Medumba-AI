(() => {
  /* ── Element references ── */
  const API_BASE = window.location.protocol === 'file:' ? 'http://localhost:5000' : '';
  const $ = (sel) => document.querySelector(sel);
  const inputEl      = $('#input');
  const outputEl     = $('#output');
  const statusEl     = $('#status');
  const translateBtn = $('#translate-btn');
  const btnText      = translateBtn.querySelector('.btn-text');
  const btnLoader    = translateBtn.querySelector('.btn-loader');
  const useLlmEl     = $('#use-llm');
  const swapBtn      = $('#swap');
  const clearBtn     = $('#clear-btn');
  const copyBtn      = $('#copy-btn');
  const charCount    = $('#char-count');
  const sourceLang   = $('#source-lang');
  const targetLang   = $('#target-lang');
  const methodBadge  = $('#method-badge');
  const reasoningSec = $('#reasoning-section');
  const reasoningTxt = $('#reasoning-text');
  const wbwSec       = $('#wbw-section');
  const wbwTxt       = $('#wbw-text');
  const matchesSec   = $('#matches-section');
  const matchesList  = $('#matches-list');
  const matchCount   = $('#match-count');
  const errorToast   = $('#error-toast');

  let direction = 'fr-md'; // default

  /* ── Helpers ── */
  function show(el) { el.classList.remove('hidden'); }
  function hide(el) { el.classList.add('hidden'); }

  function showError(msg) {
    errorToast.textContent = msg;
    show(errorToast);
    setTimeout(() => hide(errorToast), 6000);
  }

  function updateCharCount() {
    const len = inputEl.value.length;
    charCount.textContent = `${len} character${len !== 1 ? 's' : ''}`;
  }

  function setLoading(on) {
    translateBtn.disabled = on;
    if (on) {
      btnText.textContent = 'Translating…';
      show(btnLoader);
    } else {
      btnText.textContent = 'Translate';
      hide(btnLoader);
    }
  }

  function resetResults() {
    outputEl.innerHTML = '<span class="output-placeholder">Translation will appear here</span>';
    methodBadge.textContent = '';
    methodBadge.className = 'method-badge';
    hide(reasoningSec);
    hide(wbwSec);
    hide(matchesSec);
    hide(errorToast);
    matchesList.innerHTML = '';
  }

  /* ── Health check ── */
  async function checkHealth() {
    try {
      const resp = await fetch(`${API_BASE}/api/health`);
      const data = await resp.json();
      if (data && data.ok) {
        const llm = data.llmAvailable ? 'AI ready' : 'AI off';
        statusEl.textContent = `Online · ${llm} · ${data.dictionaryEntries} entries`;
        statusEl.className = 'status-pill online';
      } else {
        statusEl.textContent = 'Unknown status';
        statusEl.className = 'status-pill';
      }
    } catch {
      statusEl.textContent = 'Offline';
      statusEl.className = 'status-pill offline';
    }
  }

  /* ── Swap direction ── */
  swapBtn.addEventListener('click', () => {
    direction = direction === 'fr-md' ? 'md-fr' : 'fr-md';
    sourceLang.textContent = direction === 'fr-md' ? 'French' : 'Medumba';
    targetLang.textContent = direction === 'fr-md' ? 'Medumba' : 'French';
    resetResults();
  });

  /* ── Clear ── */
  clearBtn.addEventListener('click', () => {
    inputEl.value = '';
    updateCharCount();
    resetResults();
    inputEl.focus();
  });

  /* ── Copy ── */
  copyBtn.addEventListener('click', () => {
    const text = outputEl.textContent;
    if (!text || outputEl.querySelector('.output-placeholder')) return;
    navigator.clipboard.writeText(text).then(() => {
      const prev = copyBtn.innerHTML;
      copyBtn.textContent = '✓';
      setTimeout(() => { copyBtn.innerHTML = prev; }, 1200);
    });
  });

  /* ── Char count ── */
  inputEl.addEventListener('input', updateCharCount);

  /* ── Render matches list ── */
  function renderMatches(suggestions, examples) {
    matchesList.innerHTML = '';
    const all = [];
    if (Array.isArray(suggestions)) {
      for (const s of suggestions) all.push({ src: s.source, tgt: s.target, type: s.match || 'fuzzy', origin: 'dict' });
    }
    if (Array.isArray(examples)) {
      for (const e of examples) all.push({ src: e.source, tgt: e.target, type: e.match || 'example', origin: 'example' });
    }
    if (!all.length) { hide(matchesSec); return; }

    matchCount.textContent = all.length;
    show(matchesSec);

    for (const item of all) {
      const row = document.createElement('div');
      row.className = 'match-row';

      const src = document.createElement('span');
      src.className = 'match-src';
      src.textContent = item.src;

      const tgt = document.createElement('span');
      tgt.className = 'match-tgt';
      tgt.textContent = item.tgt;

      const tag = document.createElement('span');
      tag.className = 'match-tag ' + (item.type === 'exact' ? 'exact' : item.type === 'token' ? 'token' : 'fuzzy');
      tag.textContent = item.origin === 'example' ? 'example' : item.type;

      row.append(src, tgt, tag);
      matchesList.appendChild(row);
    }
  }

  /* ── Translate ── */
  async function translate() {
    const input = inputEl.value.trim();
    if (!input) {
      showError('Please enter text to translate.');
      return;
    }

    setLoading(true);
    resetResults();
    outputEl.innerHTML = '<span class="output-placeholder">Thinking…</span>';

    try {
      const resp = await fetch(`${API_BASE}/api/translate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          input,
          direction,
          useLlm: Boolean(useLlmEl && useLlmEl.checked),
          maxSuggestions: 12,
        }),
      });
      const data = await resp.json();

      // -- Translation output
      if (data.translation) {
        outputEl.textContent = data.translation;
      } else {
        outputEl.innerHTML = '<span class="output-placeholder">No translation available</span>';
      }

      // -- Method badge
      if (data.llmUsed) {
        methodBadge.textContent = 'AI';
        methodBadge.className = 'method-badge llm';
      } else if (data.fallback === 'dictionary_exact' || data.fallback === 'pair_exact') {
        methodBadge.textContent = 'Dictionary';
        methodBadge.className = 'method-badge dict';
      } else if (data.fallback === 'word_by_word') {
        methodBadge.textContent = 'Word-by-word';
        methodBadge.className = 'method-badge wbw';
      }

      // -- Reasoning
      if (data.reasoning) {
        reasoningTxt.textContent = data.reasoning;
        show(reasoningSec);
      }

      // -- Word-by-word
      if (data.wordByWord && data.wordByWord !== data.translation) {
        wbwTxt.textContent = data.wordByWord;
        show(wbwSec);
      }

      // -- Matches
      renderMatches(data.suggestions, data.examples);

      // -- Warnings / errors
      if (data.error) showError(data.error);
      if (Array.isArray(data.warnings) && data.warnings.length) {
        showError(data.warnings.join(' | '));
      }
    } catch (err) {
      showError(err.message || 'Network error');
      outputEl.innerHTML = '<span class="output-placeholder">Translation failed</span>';
    } finally {
      setLoading(false);
    }
  }

  translateBtn.addEventListener('click', translate);
  inputEl.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      translate();
    }
  });

  /* ── Init ── */
  updateCharCount();
  checkHealth();
})();
