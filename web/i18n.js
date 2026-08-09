// PL/EN string dictionary. Keys map to elements via data-i18n (textContent)
// or data-i18n-html (innerHTML, for strings containing markup).
const STRINGS = {
  pl: {
    "hero.title": 'Wklej tekst<span class="accent">.</span><br>Odzyskaj przecinki<span class="accent">.</span>',
    "hero.sub": "Ręcznie napisana od zera sieć neuronowa, która czyta Twój tekst i stawia brakujące przecinki oraz kropki — wytrenowana na klasyce szkolnych lektur.",
    "hero.cta": "Zobacz, jak działa ↓",
    "hero.b1": "100% hand-crafted",
    "hero.b2": "vintage · architektura 1986",
    "hero.b3": "dojrzewała na klasykach",
    "hero.b4": "bez sztucznych dodatków",

    "demo.title": "Tekst bez interpunkcji? Chwileczkę.",
    "demo.button": "Uzupełnij znaki",
    "demo.note": "Animacja poglądowa — prawdziwe pole tekstowe podłączone do modelu pojawi się tu wraz z wersją MVP.",
    "demo.working": "model analizuje…",
    "demo.done": "gotowe ✓",

    "how.title": "Jak to działa?",
    "how.s1.title": "Czyta kontekst 7 słów",
    "how.s1.text": "Model patrzy na każde słowo w kontekście — trzy słowa przed nim i trzy po nim. Tak jak człowiek, który nie stawia przecinka w ciemno.",
    "how.s2.title": "Sieć zgaduje znak",
    "how.s2.text": "Sieć neuronowa ocenia, co powinno stać po danym słowie: nic, przecinek czy kropka. Nauczyła się tego z ponad miliona słów prozy.",
    "how.s3.title": "I tak słowo po słowie",
    "how.s3.text": "Procedura powtarza się dla całego tekstu. Na końcu dostajesz ten sam tekst — tylko z interpunkcją na swoim miejscu.",

    "books.title": "Wytrenowany na klasyce",
    "books.lead": "Model uczył się interpunkcji z ~1,07 miliona słów literackiej polszczyzny — lektur, które każdy zna ze szkoły.",
    "books.trainLabel": "zbiór treningowy · 7 książek",
    "books.valLabel": "zbiór walidacyjny · 2 książki",
    "books.testLabel": "zbiór testowy · 2 książki",
    "books.credit": 'Teksty pochodzą z biblioteki <a href="https://wolnelektury.pl" target="_blank" rel="noopener">Wolne Lektury</a>.',

    "tech.title": "Dla ciekawskich: co pod maską",
    "tech.lead": "Wszystko zbudowane od zera w GNU Octave — bez bibliotek ML, z gradientami wyprowadzonymi ręcznie na kartce. Architektura w dobrym, starym stylu: MLP, rocznik 1986. Zero transformerów — na razie.",
    "tech.f1": "parametrów",
    "tech.f2": "Macro-F1 (test)",
    "tech.f3": "ponad bazę bigramową (0,511)",
    "tech.f4": "użytych bibliotek ML",
    "tech.f5": "linii kodu Octave",
    "tech.arch.in": "okno 7 słów",
    "tech.arch.hidden": "warstwa ukryta",
    "tech.archCaption": "Perceptron wielowarstwowy (MLP): dla każdego słowa przewiduje jedną z trzech klas — brak znaku, przecinek, kropka.",
    "tech.roadmapTitle": "Mapa drogowa projektu",
    "tech.r0": "Baza bigramowa",
    "tech.r1": "MLP z ręcznym backpropem",
    "tech.r2": "Bi-LSTM / mini-Transformer",
    "tech.r3": "Więcej znaków: ? !",
    "tech.r4": "+ wielkie litery",
    "tech.r5": "API i wersja MVP",

    "links.title": "Publikacja i kod",
    "links.paperKicker": "artykuł naukowy (draft)",
    "links.repoKicker": "kod źródłowy · MIT",

    "footer.line1": "Projekt badawczo-edukacyjny · etap 1 z 5 · wyniki będą się poprawiać z każdym etapem.",
    "footer.ai": "Rola AI: cała matematyka i kod modelu napisane ręcznie przez autora. AI pełniło rolę mentora — pytania i wskazówki, bez gotowego kodu. Z pomocą AI powstała natomiast ta strona.",
  },

  en: {
    "hero.title": 'Paste your text<span class="accent">.</span><br>Get your commas back<span class="accent">.</span>',
    "hero.sub": "A neural network written by hand, from scratch, that reads your text and restores missing commas and periods — trained on the classics of the Polish school reading list.",
    "hero.cta": "See how it works ↓",
    "hero.b1": "100% hand-crafted",
    "hero.b2": "vintage · 1986 architecture",
    "hero.b3": "aged on the classics",
    "hero.b4": "no artificial additives",

    "demo.title": "Text with no punctuation? One moment.",
    "demo.button": "Restore punctuation",
    "demo.note": "A scripted preview — the model works on Polish text, so the sample stays Polish. A real text box connected to the model will appear here with the MVP release.",
    "demo.working": "model is thinking…",
    "demo.done": "done ✓",

    "how.title": "How does it work?",
    "how.s1.title": "Reads a 7-word window",
    "how.s1.text": "The model looks at every word in context — three words before it and three after. Like a human, it never places a comma blindly.",
    "how.s2.title": "The network makes a call",
    "how.s2.text": "A neural network decides what should follow each word: nothing, a comma, or a period. It learned this from over a million words of prose.",
    "how.s3.title": "Word after word",
    "how.s3.text": "The procedure repeats across the whole text. You get the same text back — with the punctuation where it belongs.",

    "books.title": "Trained on the classics",
    "books.lead": "The model learned punctuation from ~1.07 million words of literary Polish — the books every Polish school kid knows.",
    "books.trainLabel": "training set · 7 books",
    "books.valLabel": "validation set · 2 books",
    "books.testLabel": "test set · 2 books",
    "books.credit": 'Texts come from the <a href="https://wolnelektury.pl" target="_blank" rel="noopener">Wolne Lektury</a> free library.',

    "tech.title": "For the curious: under the hood",
    "tech.lead": "Everything built from scratch in GNU Octave — no ML libraries, gradients derived by hand on paper. Architecture in the good old style: an MLP, vintage 1986. Zero transformers — for now.",
    "tech.f1": "parameters",
    "tech.f2": "Macro-F1 (test)",
    "tech.f3": "over the bigram baseline (0.511)",
    "tech.f4": "ML libraries used",
    "tech.f5": "lines of Octave code",
    "tech.arch.in": "7-word window",
    "tech.arch.hidden": "hidden layer",
    "tech.archCaption": "A multilayer perceptron (MLP): for every word it predicts one of three classes — no mark, comma, or period.",
    "tech.roadmapTitle": "Project roadmap",
    "tech.r0": "Bigram baseline",
    "tech.r1": "MLP with hand-written backprop",
    "tech.r2": "Bi-LSTM / mini-Transformer",
    "tech.r3": "More marks: ? !",
    "tech.r4": "+ capitalization",
    "tech.r5": "API and MVP release",

    "links.title": "Paper & code",
    "links.paperKicker": "research paper (draft)",
    "links.repoKicker": "source code · MIT",

    "footer.line1": "A research & educational project · stage 1 of 5 · results improve with every stage.",
    "footer.ai": "AI's role: all the math and model code were written by hand by the author. AI served as a mentor — questions and hints, no ready-made code. This website, however, was built with AI assistance.",
  },
};

// Demo sample sentences (with punctuation; the animation strips and restores it).
// Always Polish — the model only handles Polish text, regardless of UI language.
const DEMO_SAMPLES = [
  "Wiosna przyszła nagle, śnieg stopniał w ciągu jednej nocy, a rzeka wylała na łąki.",
  "Napisz, kiedy dojedziesz na miejsce, bo zaczynam się martwić.",
  "Zrobiło się późno, więc zamknęliśmy laptopy i zamówiliśmy pizzę.",
];
