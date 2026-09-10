/* Copy for the feature-article page (alt.html).
   Loaded after i18n.js: extends the shared STRINGS with mag.* keys and overrides the
   language-toggle label. All demo.* strings stay shared with index.html.
   Numbers live in mag.n.* (integers) and mag.num.* (decimals) so the separators follow
   the language. */

Object.assign(STRINGS.pl, {
  "ui.langToggle": "English",

  "mag.h1": "Sieć, która stawia przecinki.<br>Napisana od zera w Octave.",
  "mag.deck":
    "Perceptron wielowarstwowy z ręcznie wyprowadzonym backpropem: 295 365 parametrów, zero " +
    "bibliotek ML, czysty GNU Octave. Etap 1 z 5. Na zbiorze testowym Macro-F1 0,608, o 9,7 pp " +
    "więcej niż baza bigramowa.",
  "mag.byline": "Tomasz Ceszke, 2026",

  "mag.browser.title": "W przeglądarce",
  "mag.browser.text":
    "Bez serwera. Wagi to 1 181 460 bajtów float32 pobierane raz; model.js zakłada na bufor " +
    "widoki Float32Array pod stałymi offsetami i odtwarza potok detect.m w JavaScripcie. Około 45 tys. " +
    "mnożeń na słowo, 2100 słów w ~104 ms w headless Chrome.",

  "mag.task.title": "Zadanie",
  "mag.task.p1":
    "Klasyfikacja tokenów: dla każdego słowa jedna z trzech etykiet, opisująca znak, który po nim " +
    "następuje. Wejściem jest okno 2K+1 słów przy K = 3, czyli siedem indeksów słownika. Brzegi " +
    "tekstu dopełnia indeks <UNK>, więc ostatnie słowo też dostaje pełne okno.",
  "mag.task.p2":
    "Tokenizer składa wielkość liter i wyrzuca wszystko poza polskimi literami. Wielkich liter model " +
    "nie przewiduje: post_process.m stawia je na początku tekstu i po każdej kropce, a kropkę domyka " +
    "tekst, jeśli sieć jej nie postawiła.",
  "mag.task.asideTitle": "Etykiety",
  "mag.task.thLabel": "Etykieta",
  "mag.task.thValue": "Wartość",
  "mag.task.thMeaning": "Znaczenie",
  "mag.task.rNone": "brak znaku po słowie",
  "mag.task.rComma": "przecinek po słowie",
  "mag.task.rPeriod": "kropka po słowie",

  "mag.data.title": "Dane",
  "mag.data.p1":
    "Jedenaście książek z Wolnych Lektur, 1 195 940 par (słowo, etykieta). Podział jest " +
    "dokumentowy: książka trafia w całości do jednego zbioru, więc żadne zdanie treningowe nie " +
    "występuje w walidacji ani w teście. Słownik to 5000 najczęstszych słów zbioru treningowego, " +
    "reszta mapuje się na <UNK>.",
  "mag.data.p2":
    "Rozkład etykiet jest silnie niezbalansowany: około czterech na pięć słów nie ma po sobie znaku. " +
    "Sieć trenowana nieważoną entropią krzyżową uczy się odpowiadać NONE i zatrzymuje na 80,6% " +
    "trafności przy Macro-F1 około 0,33. Stąd ważenie klas, opisane w wynikach.",
  "mag.data.asideTitle": "Podział",
  "mag.data.thSplit": "Zbiór",
  "mag.data.thTokens": "Tokeny",
  "mag.data.thBooks": "Książki",
  "mag.data.rTrain": "treningowy",
  "mag.data.rVal": "walidacyjny",
  "mag.data.rTest": "testowy",
  "mag.data.rTotal": "razem",
  "mag.data.figTitle": "Udział etykiet w każdym zbiorze",
  "mag.data.legComma": "przecinek",
  "mag.data.legPeriod": "kropka",
  "mag.data.legNone": "reszta paska: brak znaku",
  "mag.data.figCaption":
    "Pasek to 100% tokenów zbioru. Wypełnienie to słowa, po których stoi przecinek albo kropka; " +
    "pozostałe cztery piąte to NONE. Dokładne liczby w karcie korpusu na Hugging Face.",
  "mag.data.credit":
    'Teksty pochodzą z biblioteki <a href="https://wolnelektury.pl" target="_blank" rel="noopener">Wolne Lektury</a>.',

  "mag.model.title": "Model",
  "mag.arch.in": "7 indeksów słownika",
  "mag.arch.lookup": "wiersze E",
  "mag.arch.concat": "konkatenacja, 350",
  "mag.arch.hidden": "warstwa ukryta, 128",
  "mag.arch.out": "3 klasy",
  "mag.model.figCaption":
    "Przebieg w przód dla jednego słowa. W wersji wsadowej te same operacje na macierzach " +
    "N × 7, N × 350, N × 128 i N × 3.",
  "mag.model.p1":
    "Siedem indeksów wybiera wiersze macierzy zanurzeń E (5001 × 50, ostatni wiersz to <UNK>), " +
    "sklejone w wektor 350 liczb. Dalej jedna warstwa ukryta: W1 (128 × 350), b1 i ReLU, potem " +
    "W2 (3 × 128), b2 i softmax. Zanurzenia uczą się razem z resztą sieci, od losowej inicjalizacji.",
  "mag.model.p2":
    "Inicjalizacja: E to randn · 0,01, W1 i W2 po He (randn · √(2/fan_in)), biasy zerowe. Strata " +
    "to ważona entropia krzyżowa, L = Σ wᵢ · (−log pᵢ,yᵢ) / N, gdzie wᵢ jest wagą klasy prawdziwej " +
    "etykiety.",
  "mag.model.p3":
    "Backprop wyprowadzony na kartce i zaimplementowany na macierzach w mlp_backward.m. " +
    "δ₂ = (p − y) ⊙ w / N to pełny gradient softmaksu z entropią; δ₁ = (δ₂ W₂) ⊙ 𝟙[s₁ > 0]; " +
    "gradient zanurzeń dx = δ₁ W₁ jest rozrzucany z dodawaniem do dE osobno dla każdej z siedmiu " +
    "pozycji okna. Optymalizacja: mini-batch SGD, batch 64, lr 0,005, do 30 epok z wczesnym stopem " +
    "na Macro-F1 walidacji i cierpliwością 5.",
  "mag.model.asideTitle": "Parametry",
  "mag.model.thTensor": "Tensor",
  "mag.model.thShape": "Kształt",
  "mag.model.thCount": "Liczba",
  "mag.model.thInit": "Init",
  "mag.model.initE": "randn · 0,01",
  "mag.model.initHe": "He",
  "mag.model.initZero": "zera",
  "mag.model.rTotal": "razem",

  "mag.results.title": "Wyniki",
  "mag.results.p1":
    "Metryka to Macro-F1 na zbiorze testowym: nieważona średnia F1 trzech klas, więc rzadkie klasy " +
    "liczą się tyle samo co NONE. Baza z etapu 0 to model bigramowy, który dla każdego słowa " +
    "zapamiętuje najczęstszy znak po nim w zbiorze treningowym i walidacyjnym.",
  "mag.results.thClass": "Klasa",
  "mag.results.thPrec": "Precyzja",
  "mag.results.thRec": "Czułość",
  "mag.results.thModel": "Model",
  "mag.results.thScore": "Macro-F1 (test)",
  "mag.results.rBaseline": "Bigramy, etap 0",
  "mag.results.rUntuned": "MLP przed strojeniem",
  "mag.results.rTuned": "MLP po strojeniu",
  "mag.results.rTarget": "Cel etapu",
  "mag.results.quote": "Pojemność modelu nie była wąskim gardłem. Była nim precyzja rzadkich klas.",
  "mag.results.ladderTitle": "Co dała każda zmiana, w kolejności wprowadzania",
  "mag.results.thStep": "Krok",
  "mag.results.thChange": "Zmiana",
  "mag.results.thF1": "Macro-F1",
  "mag.results.l0": "punkt wyjścia",
  "mag.results.l0c": "α = 1, K = 2, lr 0,01, 10 epok",
  "mag.results.l0v": "0,535",
  "mag.results.l1": "optymalizator",
  "mag.results.l1c": "lr z 0,01 na 0,005, 30 epok, cierpliwość 5",
  "mag.results.l1v": "≈ 0,555 (wal.)",
  "mag.results.l2": "kontekst",
  "mag.results.l2c": "K z 2 na 3",
  "mag.results.l2v": "≈ 0,559 (wal.)",
  "mag.results.l3": "wagi klas",
  "mag.results.l3c": "α z 1 na 0,5",
  "mag.results.l3v": "0,606",
  "mag.results.l4": "retrening",
  "mag.results.l4c": "α = 0,5, czysty przebieg",
  "mag.results.l4v": "0,608",
  "mag.results.p2":
    "Pełna odwrotna częstość (α = 1) waży rzadkie klasy około 11:1 względem NONE i model sypie " +
    "przecinkami: wysoka czułość, niska precyzja. Pierwiastek z odwrotnej częstości (α = 0,5) ściska " +
    "stosunek do około 3:1. Precyzja przecinka wzrosła z 0,33 do 0,59, a NONE też zyskało, bo ubyło " +
    "fałszywych przecinków. Przegląd α ∈ {0,4, 0,5, 0,6} ma czyste maksimum w 0,5. Kontekst i " +
    "optymalizator dały po kilka tysięcznych; d i h zostały, bo nie tam był problem.",
  "mag.results.asideTitle": "Wagi klas",
  "mag.results.asideText":
    "α = 0 daje wagi jednostkowe, α = 1 pełną odwrotną częstość. Stała C_ALPHA w " +
    "config/settings.m, wzór w train.m.",

  "mag.next.title": "Dalej",
  "mag.next.s0": "Preprocessing i baza bigramowa",
  "mag.next.s1": "MLP, ręczny backprop",
  "mag.next.s2": "Bi-LSTM albo mini-Transformer",
  "mag.next.s3": "Rozszerzona interpunkcja: pytajnik i wykrzyknik",
  "mag.next.s4": "Multi-task: wielkie litery",
  "mag.next.s5": "REST API i wersja MVP",
  "mag.next.done": "zrobione",
  "mag.next.now": "w toku",

  "mag.links.title": "Kod, wagi, korpus, artykuł",
  "mag.links.paper": "Artykuł",
  "mag.links.paperNote": "Cel, metoda, wyniki, wnioski. Wersja robocza w paper/paper.md.",
  "mag.links.repo": "Repozytorium",
  "mag.links.repoNote": "Cały potok w Octave: preprocess, baseline, mlp_*, train, check, detect. Licencja MIT.",
  "mag.links.model": "Model na Hugging Face",
  "mag.links.modelNote": "model.mat, karta modelu, eksport czytelny przez scipy.io.",
  "mag.links.data": "Korpus na Hugging Face",
  "mag.links.dataNote": "1,2 mln par (słowo, etykieta) w parquet, podział dokumentowy.",

  "mag.footer.ai":
    "Matematyka i kod modelu: ręcznie. AI pracowało jako mentor, pytaniami i wskazówkami, bez " +
    "gotowego kodu. Ta strona natomiast powstała z pomocą AI.",
  "mag.footer.meta": '<a href="https://tomek.ceszke.com/">Tomasz Ceszke</a>, 2026, licencja MIT.',
  "mag.repoBtn": "Kod na GitHubie",

  "mag.n.train": "828 125",
  "mag.n.val": "239 580",
  "mag.n.test": "128 235",
  "mag.n.total": "1 195 940",
  "mag.n.pE": "250 050",
  "mag.n.pW1": "44 800",
  "mag.n.pAll": "295 365",
  "mag.pct.tr.c": "11,9%", "mag.pct.tr.p": "7,3%",
  "mag.pct.va.c": "12,9%", "mag.pct.va.p": "7,2%",
  "mag.pct.te.c": "10,2%", "mag.pct.te.p": "6,4%",
  "mag.num.922": "0,922", "mag.num.588": "0,588", "mag.num.467": "0,467", "mag.num.520": "0,520",
  "mag.num.333": "0,333", "mag.num.442": "0,442", "mag.num.381": "0,381", "mag.num.608": "0,608",
  "mag.num.511": "0,511", "mag.num.535": "0,535", "mag.num.611": "0,611",
});

Object.assign(STRINGS.en, {
  "ui.langToggle": "Polski",

  "mag.h1": "A network that places commas.<br>Built from scratch in Octave.",
  "mag.deck":
    "A multilayer perceptron with backprop derived by hand: 295,365 parameters, zero ML libraries, " +
    "plain GNU Octave. Stage 1 of 5. Test-set Macro-F1 0.608, 9.7 pp above the bigram baseline.",
  "mag.byline": "Tomasz Ceszke, 2026",

  "mag.browser.title": "In the browser",
  "mag.browser.text":
    "No server. The weights are 1,181,460 bytes of float32, downloaded once; model.js lays " +
    "Float32Array views over the buffer at fixed offsets and replays the detect.m pipeline in " +
    "JavaScript. About 45K multiply-adds per word; 2,100 words in ~104 ms in headless Chrome.",

  "mag.task.title": "Task",
  "mag.task.p1":
    "Token classification: one of three labels per word, naming the mark that follows it. The input " +
    "is a window of 2K+1 words with K = 3, i.e. seven vocabulary indices. Text edges are padded with " +
    "the <UNK> index, so the last word gets a full window too.",
  "mag.task.p2":
    "The tokenizer folds case and drops everything but Polish letters. Capitalisation is not " +
    "predicted: post_process.m capitalises the first word and every word after a period, and closes " +
    "the text with a period if the network left it open.",
  "mag.task.asideTitle": "Labels",
  "mag.task.thLabel": "Label",
  "mag.task.thValue": "Value",
  "mag.task.thMeaning": "Meaning",
  "mag.task.rNone": "no mark after the word",
  "mag.task.rComma": "comma after the word",
  "mag.task.rPeriod": "period after the word",

  "mag.data.title": "Data",
  "mag.data.p1":
    "Eleven books from Wolne Lektury, 1,195,940 (word, label) pairs. The split is by document: a book " +
    "goes whole into one set, so no training sentence appears in validation or test. The vocabulary is " +
    "the 5,000 most frequent training words; everything else maps to <UNK>.",
  "mag.data.p2":
    "The label distribution is heavily imbalanced: roughly four words in five carry no mark. A network " +
    "trained with unweighted cross-entropy learns to answer NONE and stalls at 80.6% accuracy with a " +
    "Macro-F1 around 0.33. Hence the class weighting described under results.",
  "mag.data.asideTitle": "Split",
  "mag.data.thSplit": "Set",
  "mag.data.thTokens": "Tokens",
  "mag.data.thBooks": "Books",
  "mag.data.rTrain": "train",
  "mag.data.rVal": "validation",
  "mag.data.rTest": "test",
  "mag.data.rTotal": "total",
  "mag.data.figTitle": "Label share in each set",
  "mag.data.legComma": "comma",
  "mag.data.legPeriod": "period",
  "mag.data.legNone": "rest of the bar: no mark",
  "mag.data.figCaption":
    "Each bar is 100% of the set's tokens. The fill is the words followed by a comma or a period; " +
    "the remaining four fifths are NONE. Exact counts are on the corpus card on Hugging Face.",
  "mag.data.credit":
    'The texts come from the <a href="https://wolnelektury.pl" target="_blank" rel="noopener">Wolne Lektury</a> library.',

  "mag.model.title": "Model",
  "mag.arch.in": "7 vocabulary indices",
  "mag.arch.lookup": "rows of E",
  "mag.arch.concat": "concatenate, 350",
  "mag.arch.hidden": "hidden layer, 128",
  "mag.arch.out": "3 classes",
  "mag.model.figCaption":
    "Forward pass for one word. The batched version runs the same operations on N × 7, N × 350, " +
    "N × 128 and N × 3 matrices.",
  "mag.model.p1":
    "Seven indices select rows of the embedding matrix E (5001 × 50, last row is <UNK>), concatenated " +
    "into a 350-vector. Then a single hidden layer: W1 (128 × 350), b1 and ReLU, followed by " +
    "W2 (3 × 128), b2 and softmax. The embeddings are learned with the rest of the network from " +
    "random init.",
  "mag.model.p2":
    "Init: E is randn · 0.01, W1 and W2 use He (randn · √(2/fan_in)), biases are zero. The loss is " +
    "weighted cross-entropy, L = Σ wᵢ · (−log pᵢ,yᵢ) / N, with wᵢ the weight of the true class.",
  "mag.model.p3":
    "Backprop was derived on paper and implemented on matrices in mlp_backward.m. " +
    "δ₂ = (p − y) ⊙ w / N is the full softmax-plus-cross-entropy gradient; δ₁ = (δ₂ W₂) ⊙ 𝟙[s₁ > 0]; " +
    "the embedding gradient dx = δ₁ W₁ is scatter-added into dE separately for each of the seven " +
    "window positions. Optimisation: mini-batch SGD, batch 64, lr 0.005, up to 30 epochs with early " +
    "stopping on validation Macro-F1 and patience 5.",
  "mag.model.asideTitle": "Parameters",
  "mag.model.thTensor": "Tensor",
  "mag.model.thShape": "Shape",
  "mag.model.thCount": "Count",
  "mag.model.thInit": "Init",
  "mag.model.initE": "randn · 0.01",
  "mag.model.initHe": "He",
  "mag.model.initZero": "zeros",
  "mag.model.rTotal": "total",

  "mag.results.title": "Results",
  "mag.results.p1":
    "The metric is test-set Macro-F1: the unweighted mean of the three per-class F1 scores, so the rare " +
    "classes count as much as NONE. The stage 0 baseline is a bigram model that remembers, for each " +
    "word, the most frequent mark after it in the train and validation sets.",
  "mag.results.thClass": "Class",
  "mag.results.thPrec": "Precision",
  "mag.results.thRec": "Recall",
  "mag.results.thModel": "Model",
  "mag.results.thScore": "Macro-F1 (test)",
  "mag.results.rBaseline": "Bigrams, stage 0",
  "mag.results.rUntuned": "MLP before tuning",
  "mag.results.rTuned": "MLP after tuning",
  "mag.results.rTarget": "Stage target",
  "mag.results.quote": "Model capacity was not the bottleneck. Precision on the rare classes was.",
  "mag.results.ladderTitle": "What each change was worth, in the order applied",
  "mag.results.thStep": "Step",
  "mag.results.thChange": "Change",
  "mag.results.thF1": "Macro-F1",
  "mag.results.l0": "starting point",
  "mag.results.l0c": "α = 1, K = 2, lr 0.01, 10 epochs",
  "mag.results.l0v": "0.535",
  "mag.results.l1": "optimiser",
  "mag.results.l1c": "lr from 0.01 to 0.005, 30 epochs, patience 5",
  "mag.results.l1v": "≈ 0.555 (val)",
  "mag.results.l2": "context",
  "mag.results.l2c": "K from 2 to 3",
  "mag.results.l2v": "≈ 0.559 (val)",
  "mag.results.l3": "class weights",
  "mag.results.l3c": "α from 1 to 0.5",
  "mag.results.l3v": "0.606",
  "mag.results.l4": "retrain",
  "mag.results.l4c": "α = 0.5, clean run",
  "mag.results.l4v": "0.608",
  "mag.results.p2":
    "Full inverse frequency (α = 1) weights the rare classes about 11:1 against NONE and the model " +
    "scatters commas: high recall, low precision. The square root of inverse frequency (α = 0.5) " +
    "compresses the ratio to about 3:1. Comma precision rose from 0.33 to 0.59, and NONE gained too, " +
    "with fewer false commas. A sweep over α ∈ {0.4, 0.5, 0.6} peaks cleanly at 0.5. Context and the " +
    "optimiser were worth a few thousandths each; d and h stayed put, because the problem was not there.",
  "mag.results.asideTitle": "Class weights",
  "mag.results.asideText":
    "α = 0 gives unit weights, α = 1 full inverse frequency. The constant C_ALPHA lives in " +
    "config/settings.m, the formula in train.m.",

  "mag.next.title": "Next",
  "mag.next.s0": "Preprocessing and the bigram baseline",
  "mag.next.s1": "MLP, hand-written backprop",
  "mag.next.s2": "Bi-LSTM or a mini-Transformer",
  "mag.next.s3": "Extended punctuation: question and exclamation marks",
  "mag.next.s4": "Multi-task: capitalisation",
  "mag.next.s5": "REST API and an MVP release",
  "mag.next.done": "done",
  "mag.next.now": "in progress",

  "mag.links.title": "Code, weights, corpus, paper",
  "mag.links.paper": "Paper",
  "mag.links.paperNote": "Goal, method, results, conclusions. Draft in paper/paper.md.",
  "mag.links.repo": "Repository",
  "mag.links.repoNote": "The whole Octave pipeline: preprocess, baseline, mlp_*, train, check, detect. MIT licence.",
  "mag.links.model": "Model on Hugging Face",
  "mag.links.modelNote": "model.mat, model card, an export scipy.io can read.",
  "mag.links.data": "Corpus on Hugging Face",
  "mag.links.dataNote": "1.2M (word, label) pairs in parquet, split by document.",

  "mag.footer.ai":
    "Mathematics and model code: by hand. AI worked as a mentor, through questions and hints, with no " +
    "ready-made code. This page, on the other hand, was built with AI help.",
  "mag.footer.meta": '<a href="https://tomek.ceszke.com/">Tomasz Ceszke</a>, 2026, MIT licence.',
  "mag.repoBtn": "Code on GitHub",

  "mag.n.train": "828,125",
  "mag.n.val": "239,580",
  "mag.n.test": "128,235",
  "mag.n.total": "1,195,940",
  "mag.n.pE": "250,050",
  "mag.n.pW1": "44,800",
  "mag.n.pAll": "295,365",
  "mag.pct.tr.c": "11.9%", "mag.pct.tr.p": "7.3%",
  "mag.pct.va.c": "12.9%", "mag.pct.va.p": "7.2%",
  "mag.pct.te.c": "10.2%", "mag.pct.te.p": "6.4%",
  "mag.num.922": "0.922", "mag.num.588": "0.588", "mag.num.467": "0.467", "mag.num.520": "0.520",
  "mag.num.333": "0.333", "mag.num.442": "0.442", "mag.num.381": "0.381", "mag.num.608": "0.608",
  "mag.num.511": "0.511", "mag.num.535": "0.535", "mag.num.611": "0.611",
});
