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
  document.getElementById("langToggle").textContent = lang === "pl" ? "EN" : "PL";
}

/* ===== Demo animation ===== */
const box = document.getElementById("demoBox");
const btn = document.getElementById("demoBtn");
const statusEl = document.getElementById("demoStatus");

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

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

let generation = 0;

async function runDemoLoop(lang) {
  const gen = ++generation;
  const alive = () => gen === generation;
  const dict = STRINGS[lang];
  const samples = DEMO_SAMPLES; // always Polish — the model only handles Polish text
  let idx = 0;

  if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
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

/* ===== Ambient hero background: thin-line MLP schematic ===== */
function drawHeroNet() {
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
  document.getElementById("heroNet").appendChild(svg);
}

/* ===== Wiring ===== */
function setLang(lang) {
  localStorage.setItem(LANG_KEY, lang);
  applyLang(lang);
  runDemoLoop(lang);
}

document.getElementById("langToggle").addEventListener("click", () => {
  setLang(currentLang() === "pl" ? "en" : "pl");
});

drawHeroNet();
setLang(currentLang());
