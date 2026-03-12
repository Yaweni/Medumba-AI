(() => {
  const API_BASE = window.location.protocol === 'file:' ? 'http://localhost:5000' : '';
  const qEl = document.getElementById('q');
  const dirEl = document.getElementById('direction');
  const btn = document.getElementById('search');
  const out = document.getElementById('results');
  const statusEl = document.getElementById('status');
  const useLlmEl = document.getElementById('use-llm');

  function setStatus(text) {
    if (statusEl) statusEl.textContent = text;
  }

  async function checkHealth() {
    if (!statusEl) return;
    try {
      const resp = await fetch(`${API_BASE}/api/health`);
      const data = await resp.json();
      if (data && data.ok) {
        const llm = data.llmAvailable ? 'LLM ready' : 'LLM disabled';
        setStatus(`Server: online | ${llm} | entries: ${data.dictionaryEntries}`);
      } else {
        setStatus('Server: online (health unknown)');
      }
    } catch (err) {
      setStatus('Server: offline (start server on http://localhost:5000)');
    }
  }

  function clearResults() {
    out.innerHTML = '';
  }

  function addCard({ label, source, target, meta }) {
    const card = document.createElement('div');
    card.className = 'card';
    const left = document.createElement('div');
    left.className = 'left';
    if (label) {
      const labelEl = document.createElement('div');
      labelEl.className = 'label';
      labelEl.textContent = label;
      left.appendChild(labelEl);
    }
    if (source) {
      const sourceEl = document.createElement('div');
      sourceEl.className = 'source';
      sourceEl.textContent = source;
      left.appendChild(sourceEl);
    }
    if (target) {
      const targetEl = document.createElement('div');
      targetEl.className = 'target';
      targetEl.textContent = target;
      left.appendChild(targetEl);
    }
    if (meta) {
      const metaEl = document.createElement('div');
      metaEl.className = 'meta';
      metaEl.textContent = meta;
      left.appendChild(metaEl);
    }
    card.appendChild(left);
    out.appendChild(card);
  }

  function addSuggestions(suggestions, direction) {
    const details = document.createElement('details');
    details.className = 'card details';
    const summary = document.createElement('summary');
    summary.textContent = `Dictionary matches (${suggestions.length})`;
    details.appendChild(summary);

    const list = document.createElement('div');
    list.className = 'suggestions';
    for (const s of suggestions) {
      const row = document.createElement('div');
      row.className = 'suggestion-row';
      const src = document.createElement('div');
      src.className = 'suggestion-source';
      src.textContent = s.source;
      const tgt = document.createElement('div');
      tgt.className = 'suggestion-target';
      tgt.textContent = s.target;
      const meta = document.createElement('div');
      meta.className = 'meta';
      meta.textContent = s.match ? `match: ${s.match} | score: ${s.score}` : `score: ${s.score}`;
      row.appendChild(src);
      row.appendChild(tgt);
      row.appendChild(meta);
      list.appendChild(row);
    }
    details.appendChild(list);
    out.appendChild(details);
  }

  function renderResult(data) {
    clearResults();
    if (!data) {
      addCard({
        label: 'Error',
        target: 'No response from server.',
      });
      return;
    }

    if (data.error) {
      addCard({
        label: 'Error',
        target: data.error,
      });
    }

    if (data.translation) {
      const label = data.llmUsed ? 'LLM Translation' : 'Dictionary Translation';
      let meta = data.llmUsed ? `Model: ${data.model}` : 'Exact dictionary match';
      if (data.fallback === 'word_by_word') meta = 'Word-by-word fallback';
      addCard({
        label,
        source: data.input,
        target: data.translation,
        meta,
      });
    } else {
      addCard({
        label: 'No Exact Translation',
        target: 'Use the dictionary suggestions below or enable the LLM.',
      });
    }

    if (data.wordByWord && data.wordByWord !== data.translation) {
      addCard({
        label: 'Word-by-word',
        source: data.input,
        target: data.wordByWord,
        meta: 'Dictionary-based word mapping',
      });
    }

    if (Array.isArray(data.warnings) && data.warnings.length) {
      addCard({
        label: 'Warning',
        target: data.warnings.join(' | '),
      });
    }

    const suggestions = Array.isArray(data.suggestions) ? data.suggestions : [];
    if (suggestions.length) {
      addSuggestions(suggestions, data.direction);
    }
  }

  async function translate() {
    const input = qEl.value.trim();
    const direction = dirEl.value;
    if (!input) {
      clearResults();
      addCard({
        label: 'Input Required',
        target: 'Type a word or phrase to translate.',
      });
      return;
    }

    btn.disabled = true;
    btn.textContent = 'Translating...';
    clearResults();
    addCard({ label: 'Working', target: 'Fetching translation and matches...' });

    try {
      const resp = await fetch(`${API_BASE}/api/translate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          input,
          direction,
          useLlm: Boolean(useLlmEl && useLlmEl.checked),
          maxSuggestions: 8,
        }),
      });
      const data = await resp.json();
      renderResult(data);
    } catch (err) {
      renderResult({ error: err.message || String(err) });
    } finally {
      btn.disabled = false;
      btn.textContent = 'Translate';
    }
  }

  btn.addEventListener('click', translate);
  qEl.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') translate();
  });

  checkHealth();
})();
