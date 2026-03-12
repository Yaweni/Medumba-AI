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
  const swapBtn      = $('#swap');
  const clearBtn     = $('#clear-btn');
  const copyBtn      = $('#copy-btn');
  const charCount    = $('#char-count');
  const sourceLang   = $('#source-lang');
  const targetLang   = $('#target-lang');
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
      btnText.textContent = 'Translating\u2026';
      show(btnLoader);
    } else {
      btnText.textContent = 'Translate';
      hide(btnLoader);
    }
  }

  function resetOutput() {
    outputEl.innerHTML = '<span class="output-placeholder">Translation will appear here</span>';
    hide(errorToast);
  }

  /* ── Health check ── */
  async function checkHealth() {
    try {
      const resp = await fetch(`${API_BASE}/api/health`);
      const data = await resp.json();
      if (data && data.ok) {
        statusEl.textContent = 'Online';
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
    resetOutput();
  });

  /* ── Clear ── */
  clearBtn.addEventListener('click', () => {
    inputEl.value = '';
    updateCharCount();
    resetOutput();
    inputEl.focus();
  });

  /* ── Copy ── */
  copyBtn.addEventListener('click', () => {
    const text = outputEl.textContent;
    if (!text || outputEl.querySelector('.output-placeholder')) return;
    navigator.clipboard.writeText(text).then(() => {
      const prev = copyBtn.innerHTML;
      copyBtn.textContent = '\u2713';
      setTimeout(() => { copyBtn.innerHTML = prev; }, 1200);
    });
  });

  /* ── Char count ── */
  inputEl.addEventListener('input', updateCharCount);

  /* ── Translate ── */
  async function translate() {
    const input = inputEl.value.trim();
    if (!input) {
      showError('Please enter text to translate.');
      return;
    }

    setLoading(true);
    resetOutput();
    outputEl.innerHTML = '<span class="output-placeholder">Thinking\u2026</span>';

    try {
      const resp = await fetch(`${API_BASE}/api/translate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          input,
          direction,
          useLlm: true,
          maxSuggestions: 12,
        }),
      });
      const data = await resp.json();

      if (data.translation) {
        outputEl.textContent = data.translation;
      } else {
        outputEl.innerHTML = '<span class="output-placeholder">No translation available</span>';
      }

      // Suppress quota/rate-limit errors — transparent to user
      if (data.error && !/429|quota|rate.?limit|RESOURCE_EXHAUSTED/i.test(data.error)) {
        showError(data.error);
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
