/* ===== PPR — the Stage 1 MLP, running in the browser =====
   A line-by-line port of src/detect.m: tokenize -> strip marks -> vocab lookup -> pad with <UNK>
   -> 7-word windows -> embedding lookup -> W1 + ReLU -> W2 + softmax -> argmax -> post-process.
   Weights come from web/model/weights.bin, a flat little-endian float32 blob written by
   src/utils/export_web.m; meta.json carries the byte offsets, so re-training only means re-exporting.

   Two deliberate differences from the Octave original, both fixes rather than ports:
   - post_process.m only uppercases ASCII, so it leaves "ósmy" lowercase; here "Ó" comes out right.
   - labelize.m turns a lone "," into an empty-string word; here such tokens are dropped. */
const PPR = (() => {
  const BASE = "model/";
  const CHUNK = 200; // words per slice before yielding the main thread

  const build = document.querySelector('meta[name="ppr-build"]')?.content || "";
  const bust = build ? "?v=" + build : "";

  let loading = null; // memoized load promise
  let m = null;       // { meta, vocab, E, W1, b1, W2, b2, V, d, h, k, win, unk }

  /* ===== loading ===== */

  // Reads the body incrementally so the UI can show real download progress on the 1.2 MB blob.
  async function fetchBuffer(url, expectedBytes, onProgress) {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
    if (!res.body) return res.arrayBuffer(); // no streams: fall back to a plain read
    const total = Number(res.headers.get("content-length")) || expectedBytes || 0;
    const reader = res.body.getReader();
    const chunks = [];
    let got = 0;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      got += value.length;
      if (onProgress && total) onProgress(Math.min(1, got / total));
    }
    const buf = new Uint8Array(got);
    let at = 0;
    for (const c of chunks) {
      buf.set(c, at);
      at += c.length;
    }
    return buf.buffer;
  }

  function view(buffer, t) {
    return new Float32Array(buffer, t.offset, t.rows * t.cols);
  }

  function load(onProgress) {
    if (loading) return loading;
    loading = (async () => {
      const metaRes = await fetch(BASE + "meta.json" + bust);
      if (!metaRes.ok) throw new Error(`meta.json: HTTP ${metaRes.status}`);
      const meta = await metaRes.json();

      const [vocabText, buffer] = await Promise.all([
        fetch(BASE + "vocab.txt" + bust).then((r) => {
          if (!r.ok) throw new Error(`vocab.txt: HTTP ${r.status}`);
          return r.text();
        }),
        fetchBuffer(BASE + "weights.bin" + bust, meta.bytes, onProgress),
      ]);

      if (buffer.byteLength !== meta.bytes) {
        throw new Error(`weights.bin is ${buffer.byteLength} bytes, meta.json says ${meta.bytes}`);
      }

      // 1-based Octave indices become 0-based here; <UNK> is the last row of E.
      // One vocab slot holds the empty string — labelize.m emits an empty word for a standalone
      // comma, and it was frequent enough in the corpus to make the top 5000. Nothing maps to it
      // here (tokenize drops empty words), but its row still has to keep every later index in place.
      const lines = vocabText.split("\n");
      while (lines.length && lines[lines.length - 1] === "") lines.pop(); // trailing newline
      if (lines.length !== meta.V) {
        throw new Error(`vocab.txt has ${lines.length} lines, meta.json says ${meta.V}`);
      }
      const vocab = new Map();
      for (let i = 0; i < lines.length; i++) {
        if (lines[i]) vocab.set(lines[i], i);
      }

      const t = meta.tensors;
      m = {
        meta,
        vocab,
        E: view(buffer, t.E),
        W1: view(buffer, t.W1),
        b1: view(buffer, t.b1),
        W2: view(buffer, t.W2),
        b2: view(buffer, t.b2),
        V: meta.V,
        d: meta.d,
        h: meta.h,
        k: meta.k,
        win: 2 * meta.k + 1,
        unk: meta.V, // 0-based <UNK> row
      };
      if (onProgress) onProgress(1);
      return m;
    })();
    loading.catch(() => {
      loading = null; // a failed load must not poison the next attempt
    });
    return loading;
  }

  /* ===== text in ===== */

  // tokenize.m + labelize.m: fold case, keep Polish letters and the two marks, split, strip one
  // trailing mark per token. Whatever punctuation the user typed is thrown away — the model
  // re-predicts it from scratch, exactly as detect.m does.
  function tokenize(text) {
    const cleaned = text.toLowerCase().replace(/[^a-ząćęłńóśźż\s,.]/g, "");
    const words = [];
    for (const token of cleaned.split(/\s+/)) {
      if (!token) continue;
      const last = token[token.length - 1];
      const word = last === "," || last === "." ? token.slice(0, -1) : token;
      if (word) words.push(word);
    }
    return words;
  }

  /* ===== forward pass ===== */

  // One window: gather 7 embeddings into x[350], then W1 -> ReLU -> W2 -> softmax.
  function classify(idx, at, x, a1, probs) {
    const { E, W1, b1, W2, b2, d, h, win } = m;
    for (let s = 0; s < win; s++) {
      const src = idx[at + s] * d;
      x.set(E.subarray(src, src + d), s * d);
    }

    const n = win * d;
    for (let j = 0; j < h; j++) {
      let acc = b1[j];
      const row = j * n;
      for (let i = 0; i < n; i++) acc += W1[row + i] * x[i];
      a1[j] = acc > 0 ? acc : 0; // ReLU
    }

    let max = -Infinity;
    for (let c = 0; c < 3; c++) {
      let acc = b2[c];
      const row = c * h;
      for (let j = 0; j < h; j++) acc += W2[row + j] * a1[j];
      probs[c] = acc;
      if (acc > max) max = acc;
    }

    let sum = 0;
    for (let c = 0; c < 3; c++) {
      probs[c] = Math.exp(probs[c] - max); // subtract the row max, as mlp_forward.m does
      sum += probs[c];
    }
    let best = 0;
    for (let c = 0; c < 3; c++) {
      probs[c] /= sum;
      if (probs[c] > probs[best]) best = c; // strict >, so ties fall to the lowest class, like Octave's max
    }
    return best;
  }

  // Pads with k <UNK> at both ends (detect.m:50-51) so every word gets a window — including the
  // last one, which is exactly where the closing period belongs.
  async function predict(words, onProgress) {
    if (!m) throw new Error("model not loaded");
    const { k, win, unk } = m;
    const idx = new Int32Array(words.length + 2 * k).fill(unk);
    for (let i = 0; i < words.length; i++) {
      const found = m.vocab.get(words[i]);
      idx[k + i] = found === undefined ? unk : found;
    }

    const x = new Float32Array(win * m.d);
    const a1 = new Float32Array(m.h);
    const probs = new Float32Array(3);
    const out = [];
    for (let i = 0; i < words.length; i++) {
      out.push({ label: classify(idx, i, x, a1, probs), probs: Array.from(probs) });
      if ((i + 1) % CHUNK === 0) {
        if (onProgress) onProgress((i + 1) / words.length);
        await new Promise((r) => setTimeout(r)); // keep the page responsive on long inputs
      }
    }
    if (onProgress) onProgress(1);
    return out;
  }

  /* ===== text out ===== */

  // post_process.m, applied to tokens instead of the joined string: close the sentence if it does
  // not end on a mark, then capitalize the first word and every word that follows a period.
  function finalize(tokens) {
    if (!tokens.length) return tokens;
    const last = tokens[tokens.length - 1];
    if (!last.mark) last.mark = ".";
    let capitalize = true;
    for (const t of tokens) {
      if (capitalize && t.word) {
        t.word = t.word[0].toUpperCase() + t.word.slice(1);
        capitalize = false;
      }
      if (t.mark === ".") capitalize = true;
    }
    return tokens;
  }

  function join(tokens) {
    return tokens.map((t) => t.word + t.mark).join(" ");
  }

  /* ===== the whole pipeline ===== */

  async function restore(text, onProgress) {
    const words = tokenize(text);
    if (!words.length) return { tokens: [], text: "", words: 0 };
    const marks = m.meta.classes;
    const preds = await predict(words, onProgress);
    const tokens = finalize(words.map((word, i) => ({ word, mark: marks[preds[i].label] })));
    return { tokens, text: join(tokens), words: words.length };
  }

  // Guards the port: meta.json carries a sentence run through the real Octave pipeline at export
  // time, so any drift between detect.m and this file shows up in the console on first load.
  async function selfTest() {
    const { input, output } = m.meta.selfTest;
    const got = (await restore(input)).text;
    if (got !== output) {
      console.warn("PPR self-test mismatch\n  Octave: %s\n  JS:     %s", output, got);
      return false;
    }
    return true;
  }

  return {
    load,
    tokenize,
    predict,
    finalize,
    restore,
    selfTest,
    get meta() {
      return m && m.meta;
    },
    get ready() {
      return m !== null;
    },
  };
})();
