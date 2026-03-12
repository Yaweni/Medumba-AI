---
marp: true
theme: default
paginate: true
size: 16:9
style: |
  section {
    font-size: 28px;
  }
  h1 {
    font-size: 40px;
  }
  h2 {
    font-size: 26px;
  }
  table {
    font-size: 21px;
  }
  code {
    font-size: 18px;
  }
---

# Slide 1 — Why Medumba is a different AI problem
## Tonal languages break standard natural language processing assumptions

| What Works for English | Why It Breaks for Medumba |
|------------------------|---------------------------|
| Whisper automatic speech recognition (pitch-invariant) | Pitch is the phoneme: tone changes meaning |
| Byte pair encoding tokenization (`bá` → `b` + `á`) | Splits tone from base and weakens semantics |
| Text-only large language model training | 40%+ of Medumba words distinguished only by tone |

**Core message:** In Medumba, prosody is lexical information.

---

# Slide 2 — Architectural inversion vs classic natural language processing
## Our AI pipeline is tone-first, not text-first

```text
Audio input
  → Forced alignment (Montreal Forced Aligner)
  → Fundamental frequency (pitch) extraction
  → Tone classifier (high/mid/low/rising/falling, speaker-normalized)
  → learner feedback + adaptive drills
```

**Key differences**
- Prosody modeling as primary data point
- Alignment first, then tone scoring (boundary-aware supervision)
- Tone-aware representations (`ba1`) instead of tone-splitting tokens

---

# Slide 3 — What we are building
## Technical roadmap for low-resource tonal artificial intelligence

| Phase | What We Ship | Technical Reality |
|-------|--------------|-------------------|
| **Phase 1** | Dictionary + rule-based tonal scoring | Build validated corpus and speaker coverage |
| **Phase 2** | Neural tone classifier | Cross-lingual transfer + synthetic augmentation |
| **Months 9-15** | Full artificial intelligence integration | Learning app bridge + dictionary-grounded retrieval-augmented generation assistant |

**Bottom line:** We are defining a tonal-language AI pipeline where pitch, alignment, and culture-aware data are first-class signals.
