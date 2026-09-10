/* ===== i18n ===== */
const LANG_KEY = "ppr-lang";

function currentLang() {
  const stored = localStorage.getItem(LANG_KEY);
  return stored === "en" || stored === "pl" ? stored : "pl";
}

function applyLang(lang) {
  const dict = STRINGS[lang];
  document.documentElement.lang = lang;
  document.querySelectorAll("[data-i18n]").forEach((el) => {
    const key = el.dataset.i18n;
    if (dict[key] !== undefined) el.textContent = dict[key];
  });
  document.querySelectorAll("[data-i18n-html]").forEach((el) => {
    const key = el.dataset.i18nHtml;
    if (dict[key] !== undefined) el.innerHTML = dict[key];
  });
  const other = lang === "pl" ? "EN" : "PL";
  document.getElementById("langToggle").textContent = dict["ui.langToggle"] || other;
}

/* ===== Demo =====
   One box, two lives. It starts as an attract loop — the scripted animation that shows what the
   project does — and turns into a real, editable text field the moment the visitor touches it.
   From there it runs the actual model through PPR (model.js), and the result is revealed with the
   very same mark-by-mark animation, so the promise and the product look like one thing. */
const panel = document.getElementById("demoPanel");
const box = document.getElementById("demoBox");
const btn = document.getElementById("demoBtn");
const statusEl = document.getElementById("demoStatus");
const countEl = document.getElementById("demoCount");

const MAX_CHARS = 8000;
const IDLE_BACK_MS = 20000; // empty and untouched for this long -> the animation comes back

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const reduceMotion = () => window.matchMedia("(prefers-reduced-motion: reduce)").matches;

let state = "attract"; // attract | live | working | done | error
let idleTimer = null;

function t(key, vars) {
  let str = STRINGS[currentLang()][key] || "";
  if (vars) for (const [k, v] of Object.entries(vars)) str = str.replace("{" + k + "}", v);
  return str;
}

// "Wiosna przyszła nagle," -> [{word: "Wiosna", mark: ""}, ..., {word: "nagle", mark: ","}]
function tokenize(sample) {
  return sample.split(/\s+/).map((chunk) => {
    const m = chunk.match(/^(.+?)([,.])?$/);
    return { word: m[1], mark: m[2] || "" };
  });
}

// Box content while "typing": plain text + blinking caret.
function renderTyping(tokens, chars) {
  const stripped = tokens.map((t) => t.word).join(" ");
  box.textContent = stripped.slice(0, chars);
  const caret = document.createElement("span");
  caret.className = "caret";
  box.appendChild(caret);
  return stripped.length;
}

// Box content for the reveal phase: words + hidden mark spans.
function renderTokens(tokens, marksVisible) {
  box.textContent = "";
  tokens.forEach((t, i) => {
    box.appendChild(document.createTextNode(t.word));
    if (t.mark) {
      const span = document.createElement("span");
      span.className = "mark" + (marksVisible ? " on" : "");
      span.textContent = t.mark;
      box.appendChild(span);
    }
    if (i < tokens.length - 1) box.appendChild(document.createTextNode(" "));
  });
}

async function revealMarks() {
  const marks = box.querySelectorAll(".mark");
  if (reduceMotion()) {
    marks.forEach((mark) => mark.classList.add("on"));
    return;
  }
  for (const mark of marks) {
    mark.classList.add("on");
    await sleep(240);
  }
}

/* ----- attract mode: the scripted animation ----- */

let generation = 0;

async function runDemoLoop(lang) {
  const gen = ++generation;
  const alive = () => gen === generation && state === "attract";
  const dict = STRINGS[lang];
  const samples = DEMO_SAMPLES; // always Polish — the model only handles Polish text
  let idx = 0;

  if (reduceMotion()) {
    renderTokens(tokenize(samples[0]), true);
    statusEl.textContent = dict["demo.done"];
    btn.classList.add("armed");
    return;
  }

  while (alive()) {
    const tokens = tokenize(samples[idx % samples.length]);
    idx++;

    // 1. Type the stripped text.
    btn.classList.remove("armed", "pressed");
    statusEl.textContent = "";
    statusEl.classList.remove("working");
    const total = renderTyping(tokens, 0);
    for (let c = 1; c <= total; c++) {
      if (!alive()) return;
      renderTyping(tokens, c);
      await sleep(28);
    }
    await sleep(500);
    if (!alive()) return;

    // 2. Button "presses itself".
    btn.classList.add("armed");
    await sleep(900);
    if (!alive()) return;
    btn.classList.add("pressed");
    await sleep(180);
    btn.classList.remove("pressed");
    statusEl.textContent = dict["demo.working"];
    statusEl.classList.add("working");
    renderTokens(tokens, false);
    await sleep(1100);
    if (!alive()) return;

    // 3. Marks pop in one by one.
    statusEl.classList.remove("working");
    statusEl.textContent = "";
    const marks = box.querySelectorAll(".mark");
    for (const mark of marks) {
      if (!alive()) return;
      mark.classList.add("on");
      await sleep(420);
    }
    statusEl.textContent = dict["demo.done"];
    await sleep(3500);
  }
}

function enterAttract() {
  clearTimeout(idleTimer);
  state = "attract";
  box.removeAttribute("contenteditable");
  box.removeAttribute("role");
  box.removeAttribute("aria-multiline");
  box.removeAttribute("tabindex");
  box.textContent = "";
  panel.classList.remove("is-live");
  btn.tabIndex = -1;
  btn.disabled = false;
  statusEl.textContent = "";
  statusEl.className = "demo-status";
  countEl.textContent = "";
  runDemoLoop(currentLang());
}

/* ----- live mode: the real thing ----- */

function enterLive() {
  generation++; // tears down the attract loop
  clearTimeout(idleTimer);
  state = "live";
  box.textContent = "";
  // plaintext-only keeps pasted markup out; Firefox falls back to true plus the paste handler below
  box.setAttribute("contenteditable", "plaintext-only");
  if (box.contentEditable !== "plaintext-only") box.setAttribute("contenteditable", "true");
  box.setAttribute("role", "textbox");
  box.setAttribute("aria-multiline", "true");
  box.setAttribute("aria-label", t("demo.placeholder"));
  panel.classList.add("is-live");
  btn.tabIndex = 0;
  btn.classList.add("armed");
  btn.classList.remove("pressed");
  statusEl.textContent = "";
  statusEl.className = "demo-status";
  updateCount();
  box.focus();
  warmModel();
}

function updateCount() {
  const n = box.textContent.trim().length;
  box.classList.toggle("is-empty", n === 0);
  countEl.textContent = n ? t("demo.count", { n, max: MAX_CHARS }) : "";
  countEl.classList.toggle("over", n > MAX_CHARS);
}

function scheduleIdleReturn() {
  clearTimeout(idleTimer);
  if (state !== "live" || box.textContent.trim()) return;
  idleTimer = setTimeout(() => {
    if (state === "live" && !box.textContent.trim()) enterAttract();
  }, IDLE_BACK_MS);
}

// Starts the 1.2 MB download as soon as the visitor engages, so it overlaps with them typing.
function warmModel() {
  PPR.load(onDownload).then(
    () => {
      if (statusEl.classList.contains("loading")) {
        statusEl.textContent = "";
        statusEl.className = "demo-status";
      }
      PPR.selfTest();
    },
    (err) => fail(err)
  );
}

function onDownload(fraction) {
  if (state === "done" || PPR.ready) return;
  statusEl.className = "demo-status loading";
  statusEl.textContent = t("demo.loadingPct", { pct: Math.round(fraction * 100) });
}

function fail(err) {
  console.error(err);
  state = "error";
  statusEl.className = "demo-status error";
  statusEl.textContent = t("demo.error");
  btn.disabled = true;
}

async function run() {
  if (state === "working" || state === "error") return;
  const text = box.textContent.trim();
  if (!text) {
    statusEl.className = "demo-status";
    statusEl.textContent = t("demo.empty");
    box.focus();
    return;
  }
  if (text.length > MAX_CHARS) {
    statusEl.className = "demo-status error";
    statusEl.textContent = t("demo.tooLong", { n: text.length, max: MAX_CHARS });
    return;
  }

  state = "working";
  clearTimeout(idleTimer);
  btn.classList.add("pressed");
  setTimeout(() => btn.classList.remove("pressed"), 180);

  try {
    if (!PPR.ready) {
      statusEl.className = "demo-status loading";
      statusEl.textContent = t("demo.loading");
      await PPR.load(onDownload);
      PPR.selfTest();
    }
    statusEl.className = "demo-status working";
    statusEl.textContent = t("demo.working");

    const started = performance.now();
    const result = await PPR.restore(text);
    // Give the spinner a beat on short inputs — the model is far too fast to be believed otherwise.
    const elapsed = performance.now() - started;
    if (!reduceMotion() && elapsed < 450) await sleep(450 - elapsed);

    if (!result.tokens.length) {
      state = "live";
      statusEl.className = "demo-status";
      statusEl.textContent = t("demo.empty");
      return;
    }

    renderTokens(result.tokens, false);
    statusEl.className = "demo-status";
    statusEl.textContent = "";
    await revealMarks();
    statusEl.textContent = t("demo.done");
    state = "done";
    updateCount();
  } catch (err) {
    fail(err);
  }
}

/* ----- wiring ----- */

panel.addEventListener("pointerdown", (e) => {
  if (state === "attract") {
    e.preventDefault(); // the click belongs to the box, not to whatever the animation painted
    enterLive();
  }
});

box.addEventListener("keydown", (e) => {
  if (state === "attract") {
    enterLive();
    return;
  }
  if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
    e.preventDefault();
    run();
  }
});

box.addEventListener("input", () => {
  if (state === "done") state = "live";
  updateCount();
  scheduleIdleReturn();
});

box.addEventListener("blur", scheduleIdleReturn);

// contenteditable would otherwise happily swallow styled HTML.
box.addEventListener("paste", (e) => {
  if (state === "attract") return;
  e.preventDefault();
  const text = (e.clipboardData || window.clipboardData).getData("text/plain");
  document.execCommand("insertText", false, text);
});

btn.addEventListener("click", () => {
  if (state === "attract") enterLive();
  else run();
});

/* ===== Site-wide punctuation motif =====
   Wraps every , and . in page text in <span class="punct"> (accent color;
   bold in headings via CSS). Skips the demo box (own .mark system), already
   wrapped nodes, and decimal separators ("1,2", "0,608"). Re-run after every
   applyLang(): i18n swaps reset textContent and destroy the wraps. */
function accentPunctuation() {
  // A page can narrow the motif to chosen blocks by marking them [data-punct];
  // with none marked it falls back to main and footer, as on index.html.
  const marked = document.querySelectorAll("[data-punct]");
  const roots = marked.length ? marked : document.querySelectorAll("main, footer");
  roots.forEach((root) => {
    const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, {
      acceptNode(node) {
        const parent = node.parentElement;
        if (!parent || parent.closest("#demoBox, #demoStatus, #demoCount, .punct, .accent")) {
          return NodeFilter.FILTER_REJECT;
        }
        return /[,.]/.test(node.nodeValue)
          ? NodeFilter.FILTER_ACCEPT
          : NodeFilter.FILTER_SKIP;
      },
    });
    const nodes = [];
    while (walker.nextNode()) nodes.push(walker.currentNode);

    for (const node of nodes) {
      const text = node.nodeValue;
      const frag = document.createDocumentFragment();
      let last = 0;
      for (const m of text.matchAll(/[,.]/g)) {
        const isDecimal =
          /\d/.test(text[m.index - 1] || "") && /\d/.test(text[m.index + 1] || "");
        if (isDecimal) continue;
        frag.appendChild(document.createTextNode(text.slice(last, m.index)));
        const span = document.createElement("span");
        span.className = "punct";
        span.textContent = m[0];
        frag.appendChild(span);
        last = m.index + 1;
      }
      if (last === 0) continue;
      frag.appendChild(document.createTextNode(text.slice(last)));
      node.replaceWith(frag);
    }
  });
}

/* ===== Ambient hero background: thin-line MLP schematic ===== */
function drawHeroNet() {
  const host = document.getElementById("heroNet");
  if (!host) return; // the document-style page has no ambient background
  const layers = [
    { x: 80, ys: [140, 260, 380, 500, 620] },
    { x: 420, ys: [80, 180, 280, 380, 480, 580, 680] },
    { x: 760, ys: [230, 380, 530] },
  ];
  const svgNS = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(svgNS, "svg");
  svg.setAttribute("viewBox", "0 0 840 760");
  svg.setAttribute("preserveAspectRatio", "xMidYMid slice");
  for (let l = 0; l < layers.length - 1; l++) {
    for (const y1 of layers[l].ys) {
      for (const y2 of layers[l + 1].ys) {
        const line = document.createElementNS(svgNS, "line");
        line.setAttribute("x1", layers[l].x);
        line.setAttribute("y1", y1);
        line.setAttribute("x2", layers[l + 1].x);
        line.setAttribute("y2", y2);
        svg.appendChild(line);
      }
    }
  }
  for (const layer of layers) {
    for (const y of layer.ys) {
      const c = document.createElementNS(svgNS, "circle");
      c.setAttribute("cx", layer.x);
      c.setAttribute("cy", y);
      c.setAttribute("r", 7);
      svg.appendChild(c);
    }
  }
  host.appendChild(svg);
}

/* ===== Wiring ===== */
function setLang(lang) {
  localStorage.setItem(LANG_KEY, lang);
  applyLang(lang);
  accentPunctuation();
  box.dataset.placeholder = STRINGS[lang]["demo.placeholder"];
  if (state === "attract") runDemoLoop(lang);
  else refreshLiveStrings();
}

// A language switch must not throw away what the visitor typed, so only the chrome is re-rendered.
function refreshLiveStrings() {
  box.setAttribute("aria-label", t("demo.placeholder"));
  updateCount();
  if (state === "done") statusEl.textContent = t("demo.done");
  else if (state === "error") statusEl.textContent = t("demo.error");
}

document.getElementById("langToggle").addEventListener("click", () => {
  setLang(currentLang() === "pl" ? "en" : "pl");
});

drawHeroNet();
setLang(currentLang());
