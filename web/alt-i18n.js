/* Copy for the document-style page (alt.html).
   Loaded after i18n.js: extends the shared STRINGS with doc.* keys and overrides
   the language-toggle label. All demo.* strings stay shared with index.html. */

Object.assign(STRINGS.pl, {
  "ui.langToggle": "English",

  "doc.abstract":
    'Sieć neuronowa, która wstawia przecinki i kropki do polskiego tekstu pozbawionego interpunkcji. ' +
    'Napisałem ją od zera w GNU Octave, wyprowadzając wszystkie gradienty ręcznie, bez żadnej biblioteki ' +
    'uczenia maszynowego. Model liczy się poniżej, w Twojej przeglądarce.',
  "doc.byline": "Tomasz Ceszke, projekt badawczo-edukacyjny, etap 1 z 5",

  "doc.how.title": "Jak to działa",
  "doc.how.p1":
    'Model ogląda tekst słowo po słowie. Dla każdego słowa bierze okno siedmiu wyrazów: trzy przed nim, ' +
    'samo słowo i trzy po nim. Na tej podstawie rozstrzyga, co powinno stać zaraz za środkowym wyrazem: ' +
    'nic, przecinek albo kropka. Potem okno przesuwa się o jedną pozycję i wszystko zaczyna się od nowa. ' +
    'Na brzegach tekstu brakujące miejsca wypełnia słowo puste.',
  "doc.how.figCaption":
    'Okno wokół wyrazu „nocy”. Spośród trzech możliwości model wybiera przecinek.',
  "doc.how.choiceNone": "bez znaku",
  "doc.how.p2":
    'Każde z siedmiu słów zamienia się w wektor pięćdziesięciu liczb, wyuczony razem z resztą sieci. ' +
    'Sklejone w jeden ciąg trafiają do warstwy ukrytej ze 128 neuronami ReLU, a stamtąd do softmaksu ' +
    'nad trzema klasami. Razem 295 400 parametrów. GPT-2 small ma ich mniej więcej czterysta razy więcej.',

  "doc.corpus.title": "Na czym się uczył",
  "doc.corpus.p1":
    'Korpus to jedenaście książek z Wolnych Lektur, razem 1 195 940 par (słowo, znak). ' +
    'Podział jest dokumentowy, nie losowy: żadna książka nie trafia do dwóch zbiorów naraz, ' +
    'więc zdanie ze zbioru treningowego nie ma jak wyciec do oceny.',
  "doc.corpus.trainLabel": "Zbiór treningowy, 7 książek, 828 125 słów.",
  "doc.corpus.valLabel": "Zbiór walidacyjny, 2 książki, 239 580 słów.",
  "doc.corpus.testLabel": "Zbiór testowy, 2 książki, 128 235 słów.",
  "doc.corpus.p2":
    'Cztery słowa na pięć nie mają po sobie żadnego znaku. Przecinek pada po 11,9% wyrazów, kropka po 7,3%. ' +
    'Sieć trenowana wprost na takim rozkładzie uczy się odpowiadać „nic” zawsze i osiąga 80% trafności, ' +
    'nie nauczywszy się przy tym niczego. To jest właściwy problem tego zadania i o nim jest sekcja poniżej.',
  "doc.corpus.credit":
    'Teksty pochodzą z biblioteki <a href="https://wolnelektury.pl" target="_blank" rel="noopener">Wolne Lektury</a>.',

  "doc.results.title": "Wyniki",
  "doc.results.p1":
    'Miarą jest Macro-F1: średnia F1 z trzech klas liczona bez wag, więc rzadka kropka znaczy tyle samo ' +
    'co częsty brak znaku. Punktem odniesienia jest model bigramowy z etapu 0, który dla każdego słowa ' +
    'zapamiętuje, jaki znak najczęściej po nim stawał.',
  "doc.results.thClass": "Klasa",
  "doc.results.thPrec": "Precyzja",
  "doc.results.thRec": "Czułość",
  "doc.results.rowNone": "bez znaku",
  "doc.results.rowComma": "przecinek",
  "doc.results.rowPeriod": "kropka",
  "doc.results.rowMacro": "Macro-F1",
  "doc.results.thModel": "Model",
  "doc.results.thScore": "Macro-F1 na teście",
  "doc.results.rowBaseline": "Bigramy, etap 0",
  "doc.results.rowUntuned": "MLP przed strojeniem",
  "doc.results.rowTuned": "MLP po strojeniu",
  "doc.num.922": "0,922",
  "doc.num.588": "0,588",
  "doc.num.467": "0,467",
  "doc.num.520": "0,520",
  "doc.num.333": "0,333",
  "doc.num.442": "0,442",
  "doc.num.381": "0,381",
  "doc.num.608": "0,608",
  "doc.num.511": "0,511",
  "doc.num.535": "0,535",
  "doc.results.p2":
    'Największą różnicę zrobiło nie powiększanie sieci, tylko temperowanie wag klas. Pełna odwrotna ' +
    'częstość faworyzuje rzadkie klasy w proporcji 11:1 i model zaczyna sypać przecinkami. Pierwiastek ' +
    'z odwrotnej częstości ściska tę proporcję do 3:1: precyzja przecinka podniosła się z 0,33 do 0,59, ' +
    'a Macro-F1 z 0,535 do 0,608. Pojemność modelu nigdy nie była wąskim gardłem.',

  "doc.next.title": "Co dalej",
  "doc.next.s0": "Wstępne przetwarzanie korpusu i baza bigramowa.",
  "doc.next.s1": "MLP z ręcznie wyprowadzonym backpropem.",
  "doc.next.s2": "Bi-LSTM albo mini-Transformer.",
  "doc.next.s3": "Więcej znaków: pytajnik i wykrzyknik.",
  "doc.next.s4": "Wielkie litery jako drugie zadanie.",
  "doc.next.s5": "API i wersja MVP.",
  "doc.next.done": "zrobione",
  "doc.next.now": "w toku",

  "doc.links.title": "Kod, wagi, artykuł",
  "doc.links.paper": "Artykuł",
  "doc.links.paperNote": "Cel, metoda, wyniki i wnioski w formie akademickiej. Wersja robocza.",
  "doc.links.repo": "Repozytorium na GitHubie",
  "doc.links.repoNote": "Kod w Octave, notatki z każdego etapu, licencja MIT.",
  "doc.links.model": "Model na Hugging Face",
  "doc.links.modelNote": "Wagi, karta modelu i eksport czytelny dla scipy.",
  "doc.links.data": "Korpus na Hugging Face",
  "doc.links.dataNote": "Wszystkie pary (słowo, znak) z podziałem dokumentowym.",

  "doc.footer.ai":
    'Całą matematykę i kod modelu napisałem ręcznie. AI było mentorem: zadawało pytania i naprowadzało, ' +
    'nie podawało gotowego kodu. Ta strona powstała inaczej, z pomocą AI.',
  "doc.footer.meta": "Tomasz Ceszke, 2026, licencja MIT.",
});

Object.assign(STRINGS.en, {
  "ui.langToggle": "Polski",

  "doc.abstract":
    'A neural network that puts commas and periods back into Polish text that has lost them. ' +
    'I wrote it from scratch in GNU Octave, deriving every gradient by hand, with no machine ' +
    'learning library. The model runs below, inside your browser.',
  "doc.byline": "Tomasz Ceszke, a research and teaching project, stage 1 of 5",

  "doc.how.title": "How it works",
  "doc.how.p1":
    'The model reads the text one word at a time. For each word it takes a window of seven: three ' +
    'before it, the word itself, and three after. From that it decides what belongs right after the ' +
    'middle word: nothing, a comma, or a period. Then the window shifts by one position and the whole ' +
    'thing starts again. At the edges of the text the missing slots are filled with an empty word.',
  "doc.how.figCaption":
    'The window around the word “nocy”. Of the three options the model picks the comma.',
  "doc.how.choiceNone": "no mark",
  "doc.how.p2":
    'Each of the seven words becomes a vector of fifty numbers, learned together with the rest of the ' +
    'network. Strung into one long vector they feed a hidden layer of 128 ReLU units, and from there ' +
    'a softmax over three classes. 295,400 parameters in total. GPT-2 small has roughly four hundred times more.',

  "doc.corpus.title": "What it learned from",
  "doc.corpus.p1":
    'The corpus is eleven books from Wolne Lektury, 1,195,940 (word, mark) pairs in all. The split is ' +
    'by document rather than at random: no book appears in two sets, so a sentence from training has ' +
    'no way of leaking into the evaluation.',
  "doc.corpus.trainLabel": "Training set, 7 books, 828,125 words.",
  "doc.corpus.valLabel": "Validation set, 2 books, 239,580 words.",
  "doc.corpus.testLabel": "Test set, 2 books, 128,235 words.",
  "doc.corpus.p2":
    'Four words in five are followed by no mark at all. A comma follows 11.9% of them, a period 7.3%. ' +
    'A network trained straight on that distribution learns to answer “nothing” every time and scores ' +
    '80% accuracy having learned nothing at all. That is the real problem in this task, and the section ' +
    'below is about it.',
  "doc.corpus.credit":
    'The texts come from the <a href="https://wolnelektury.pl" target="_blank" rel="noopener">Wolne Lektury</a> library.',

  "doc.results.title": "Results",
  "doc.results.p1":
    'The measure is Macro-F1: the unweighted mean of the three per-class F1 scores, so the rare period ' +
    'counts as much as the common empty slot. The reference point is the bigram model from stage 0, ' +
    'which remembers, for every word, the mark that most often followed it.',
  "doc.results.thClass": "Class",
  "doc.results.thPrec": "Precision",
  "doc.results.thRec": "Recall",
  "doc.results.rowNone": "no mark",
  "doc.results.rowComma": "comma",
  "doc.results.rowPeriod": "period",
  "doc.results.rowMacro": "Macro-F1",
  "doc.results.thModel": "Model",
  "doc.results.thScore": "Macro-F1 on test",
  "doc.results.rowBaseline": "Bigrams, stage 0",
  "doc.results.rowUntuned": "MLP before tuning",
  "doc.results.rowTuned": "MLP after tuning",
  "doc.num.922": "0.922",
  "doc.num.588": "0.588",
  "doc.num.467": "0.467",
  "doc.num.520": "0.520",
  "doc.num.333": "0.333",
  "doc.num.442": "0.442",
  "doc.num.381": "0.381",
  "doc.num.608": "0.608",
  "doc.num.511": "0.511",
  "doc.num.535": "0.535",
  "doc.results.p2":
    'What made the difference was not a bigger network but tempering the class weights. Full inverse ' +
    'frequency favours the rare classes by 11:1 and the model starts scattering commas everywhere. The ' +
    'square root of inverse frequency compresses that to 3:1: comma precision rose from 0.33 to 0.59, ' +
    'and Macro-F1 from 0.535 to 0.608. Model capacity was never the bottleneck.',

  "doc.next.title": "What comes next",
  "doc.next.s0": "Corpus preprocessing and the bigram baseline.",
  "doc.next.s1": "An MLP with backprop derived by hand.",
  "doc.next.s2": "Bi-LSTM or a mini-Transformer.",
  "doc.next.s3": "More marks: question mark and exclamation mark.",
  "doc.next.s4": "Capitalisation as a second task.",
  "doc.next.s5": "An API and an MVP release.",
  "doc.next.done": "done",
  "doc.next.now": "in progress",

  "doc.links.title": "Code, weights, paper",
  "doc.links.paper": "Paper",
  "doc.links.paperNote": "Goal, method, results and conclusions, written up academically. Draft.",
  "doc.links.repo": "Repository on GitHub",
  "doc.links.repoNote": "The Octave code, notes from every stage, MIT licence.",
  "doc.links.model": "Model on Hugging Face",
  "doc.links.modelNote": "Weights, model card, and an export scipy can read.",
  "doc.links.data": "Corpus on Hugging Face",
  "doc.links.dataNote": "Every (word, mark) pair, split by document.",

  "doc.footer.ai":
    'All the mathematics and the model code I wrote by hand. AI acted as a mentor: it asked questions ' +
    'and pointed the way, it did not hand over code. This page came about differently, with AI help.',
  "doc.footer.meta": "Tomasz Ceszke, 2026, MIT licence.",
});
