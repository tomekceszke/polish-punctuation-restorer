# Plan nauki: Polish Punctuation Restorer

Projekt edukacyjny. Cel: zrozumieć, jak zbudować sekwencyjny klasyfikator od zera, wyprowadzając matematykę i implementując backprop własnoręcznie na macierzach.

## Punkty odniesienia z Twojego doświadczenia

W całym planie celowo odwołuję się do rzeczy, które już zrobiłeś — żeby nowy projekt był ciągiem dalszym ścieżki, a nie startem od zera. Mapa referencji:

| Źródło | Co przenosisz tutaj |
|--------|---------------------|
| **`ml-applications`** (twoje repo-atlas) | Większość notatek teoretycznych już masz spisaną: cost function, gradient descent, feature scaling, λ-regularizacja, bias/variance, sigmoid, one-vs-all, backprop. W `notes/` zamiast przepisywać — linkuj i dopisuj tylko delty dla softmax+CE i embeddingów. |
| **`vehicles-counter`** | **Bezpośredni szablon strukturalny.** Octave, sieć z 1 warstwą ukrytą, backprop, zapis wag jako `Theta1.mat`/`Theta2.mat`, split `learn.m`/`check.m`/`detect.m`, hierarchia `bin/conf/doc/lib/datasource`. "Intentionally written in pure Matlab language, using only elementary arithmetic operations" — identyczna filozofia, którą tu kontynuujemy. |
| **`traffic-light-detection`** | Ten sam szkielet co vehicles-counter, ale z logistic regression + gradient descent. Referencja, jak wyglądało u ciebie przejście od prostszego modelu (log-reg) do bardziej złożonego (NN). Tutaj robisz analogiczny skok: n-gram baseline → MLP. |
| **`car-price-prediction`** | Feature engineering na danych z realnego źródła (scraping otomoto). Przy preprocessingu Wolnych Lektur przyda się twoje doświadczenie z "brudnymi" danymi. |
| **`ml-login`** | Filozofia "no external ML libraries, all algorithms written using only basic math formulas" — ta sama, którą trzymamy tutaj. Dodatkowo: pokazuje, że potrafisz zrobić ML jako osobną usługę z czystym interfejsem (train/predict split) — wzór architektoniczny, który warto zachować nawet w Octave. |
| **Andrew Ng ML (Coursera)** | Szczególnie **ex3** (forward pass, multi-class klasyfikacja) i **ex4** (`nnCostFunction.m`, `sigmoidGradient.m`, `randInitializeWeights.m`, `checkNNGradients.m`). 80% backpropu z ex4 jest reużywalne — różnice wymieniam przy każdym kroku poniżej. |

**Kluczowa różnica koncepcyjna** vs ex4/vehicles-counter/traffic-light-detection: tam **wejście to piksele** (ciągłe wartości w `[0,1]`), tu **wejście to indeksy słów** (dyskretne wartości w `[1, V]`). To wymusza jedną dodatkową warstwę — **embedding lookup** — której w żadnym z twoich dotychczasowych projektów nie było. To jest jedyny nowy koncept implementacyjny do opanowania. Reszta (cost, backprop, gradient check, optimizer) to kontynuacja tego, co robiłeś.

**Druga różnica** vs ex4: tam output layer używa sigmoid + K jednostek wyjściowych (one-hot przez one-vs-all). Tutaj używamy **softmax** + cross-entropy — jedyna zmiana matematyczna wymagająca własnego wyprowadzenia (reszta gradientów jest identyczna).

## Zakres i decyzje

| Parametr | Wartość | Uzasadnienie |
|----------|---------|--------------|
| Zadanie | Token classification | Dla każdego słowa przewidujemy etykietę znaku po nim |
| Etykiety | `{NONE=0, COMMA=1, PERIOD=2}` | Minimalny sensowny zestaw |
| Korpus | Wolne Lektury (PL) | Czysty tekst literacki, legalny do użytku, starczy na prototyp |
| Stack | GNU Octave | `brew install octave`, składnia MATLAB-like |
| Reprezentacja | Embeddingi na top-N słowach + `<UNK>` | Lookup = indeksowanie wiersza macierzy |
| Kontekst | Okno ±k słów | Zaczynamy od k=2, tuningujemy |
| Tokenizacja | Whitespace + lowercase | Diakrytyki zostawiamy |
| Podział | 80/10/10 po dokumentach | Unikamy wycieku fraz |

## Struktura repo

Wzoruję się na twojej konwencji z `vehicles-counter` i `traffic-light-detection` (`learn.m`/`check.m`/`detect.m` + `bin/conf/doc/lib/datasource`) — z rozszerzeniem pod NLP:

```
polish-punctuation-restorer/
├── data/
│   ├── raw/                 # pobrane teksty z Wolnych Lektur
│   └── processed/           # tokeny + etykiety (CSV/MAT)
├── src/
│   ├── preprocess.m         # surowy tekst → (tokeny, etykiety)
│   ├── vocab.m              # budowa słownika top-N
│   ├── baseline_ngram.m     # Etap 0
│   ├── mlp_forward.m        # Etap 1: forward pass
│   ├── mlp_backward.m       # Etap 1: gradient ręcznie
│   ├── learn.m              # pętla uczenia (odpowiednik vehicles-counter/learn.m)
│   ├── check.m              # ewaluacja na zbiorze testowym (odpowiednik check.m)
│   ├── detect.m             # inference na dowolnym tekście (odpowiednik detect.m)
│   └── evaluate.m           # precision/recall/F1 per klasa
├── Theta1.mat, Theta2.mat   # zapisane wagi — konwencja 1:1 z vehicles-counter
├── E.mat                    # macierz embeddingów (nowe vs vehicles-counter)
├── vocab.mat                # słownik słowo↔indeks (nowe)
├── notes/                   # wyprowadzenia matematyki
└── README.md
```

## Etap 0 — Baseline statystyczny (1–2 wieczory)

**Cel:** mieć *floor*, który każdy kolejny etap musi pobić. Zero ML. *Analogicznie do tego, jak w `ml-applications` najpierw opisujesz linear regression jako punkt wyjścia przed logistic regression i NN.*

1. Pobierz 5–10 powieści z Wolnych Lektur (np. Prus, Sienkiewicz, Żeromski). *Prostsze źródło niż scraping otomoto z `car-price-prediction` — tam walczyłeś z HTML-em; tutaj `.txt` z czystym tekstem.*
2. Napisz `preprocess.m`:
   - wczytaj plik, lowercase, usuń wszystko poza `[a-ząćęłńóśźż\s,.]`
   - iteruj po tokenach, buduj listę `(word, label)` gdzie label = znak występujący bezpośrednio po słowie (lub NONE)
   - zapisz jako `.mat` (macierze łatwiejsze w Octave niż struktury)
3. `baseline_ngram.m`: dla każdej pary `(w_i, w_{i+1})` licz `count[label | w_i, w_{i+1}]`. Predykcja = argmax z wygładzaniem Laplace'a. *To odpowiednik "zgadnij medianę ceny w otomoto" z `car-price-prediction` — baseline bez modelu, do porównania.*
4. Oceń na walidacji: **raportuj F1 per klasa**, nie accuracy.

**Czego się nauczysz:** pracy z korpusem, pułapek preprocessingu PL, świadomości dysproporcji klas (spodziewaj się ~85% NONE), różnicy accuracy vs F1.

**Oczekiwany wynik:** F1 dla PERIOD ~0.4–0.55, dla COMMA ~0.15–0.3. Słaby, ale to punkt odniesienia.

## Etap 1 — MLP z własnym backpropem (2–4 tygodnie)

**Cel:** zbudować sieć neuronową od macierzy, wyprowadzić każdy gradient ręcznie.

### Architektura

```
input:  [w_{i-2}, w_{i-1}, w_i, w_{i+1}, w_{i+2}]   (indeksy słów, 5 liczb)
  ↓ embedding lookup (E ∈ R^{V×d})
embed:  5 wektorów po d wymiarów
  ↓ concat
x:      R^{5d}
  ↓ W1 ∈ R^{h×5d}, b1 ∈ R^h, ReLU
h:      R^h
  ↓ W2 ∈ R^{3×h}, b2 ∈ R^3, softmax
y_hat:  R^3  (rozkład nad {NONE, COMMA, PERIOD})
```

Startowe hiperparametry: `V=5000, d=50, h=128, k=2, batch=64, lr=0.01`.

### Kroki implementacji

1. **Słownik** (`vocab.m`): top-V słów + `<UNK>` + `<PAD>` (dla początku/końca dokumentu). *Nowość vs vehicles-counter — tam nie było słownika, bo wejściem były piksele.*
2. **Forward** (`mlp_forward.m`): zwraca `y_hat` oraz cache aktywacji do backpropu. *Analogicznie do `predict.m` z ex3 Andrew Ng — ten sam wzorzec `a1 → z2 → a2 → z3 → a3`, tylko z embedding-lookup przed `a1` i softmax zamiast sigmoid na końcu.*
3. **Loss**: weighted cross-entropy. Wagi klas = odwrotność częstości (inaczej model nauczy się zawsze przewidywać NONE). *Porównaj z `nnCostFunction.m` z ex4 — tam był sum-of-log-losses dla sigmoid outputs; tutaj jedno sumowanie po klasach z softmax.*
4. **Backward** (`mlp_backward.m`): wyprowadź i zaimplementuj gradient po każdym parametrze. Kluczowe: gradient softmax+CE wychodzi jako `(y_hat - y_true)`, to trzeba samemu wyprowadzić — nie przepisywać. *W ex4/`vehicles-counter` masz dokładnie ten sam wzór jako `δ3 = a3 - y` (tam dla sigmoid+CE; że wychodzi identycznie dla softmax+CE, to jest właśnie to, co warto samemu zobaczyć).* Dalej `δ2 = (Θ2' * δ3) .* ReLU'(z2)` — ten sam szkielet co w ex4, tylko z ReLU zamiast sigmoidGradient.
5. **Gradient check**: porównaj analityczny gradient z numerycznym `(L(θ+ε) - L(θ-ε)) / 2ε`. Bez tego nie ruszaj dalej. Różnica względna < 1e-6 na próbce parametrów. *Zaadaptuj `checkNNGradients.m` z ex4 — struktura jest 1:1, zmieniasz tylko wywoływaną funkcję kosztu i dorzucasz sprawdzenie gradientu po embeddingach.*
6. **Inicjalizacja wag**: Xavier/He zamiast `randInitializeWeights.m` z ex4 (`ε_init` uniform). Przy ReLU to ma znaczenie — uzasadnienie w `notes/`.
7. **Optimizer**: najpierw SGD z momentum, potem Adam. Zobacz empirycznie różnicę w krzywych strat. *W `vehicles-counter` używałeś `fmincg` (advanced optimizer z ex4) — tu świadomie robimy krok wstecz do SGD, żeby zobaczyć mechanikę uczenia, potem krok w przód do Adam.*
8. **Pętla uczenia** (`learn.m`): mini-batche, logowanie train/val loss co epokę, early stopping. Zapis wag do `Theta1.mat`, `Theta2.mat`, `E.mat` — konwencja z vehicles-counter.
9. **Ewaluacja** (`check.m`, `evaluate.m`): ta sama metoda co w Etapie 0. Porównaj bezpośrednio macro-F1.

### Matematyka do wyprowadzenia samemu (w `notes/`)

W `ml-applications` masz już spisane: hypothesis, cost function, gradient descent, feature scaling, regularizację, bias/variance, sigmoid, decision boundary, one-vs-all, backprop na wysokim poziomie. Nie powielaj — linkuj. Dopisz tylko delty specyficzne dla tego projektu:

- Gradient softmax w izolacji: ∂softmax(z)_i / ∂z_j (macierz Jacobiego, różna dla i=j i i≠j)
- Gradient cross-entropy + softmax razem — piękne uproszczenie do `p - y` (to ta sama forma co `δ3 = a3 - y` dla sigmoid+CE z ex4 — warto zobaczyć, że to nie przypadek)
- Gradient ReLU: prosty, ale warto zapisać przypadki brzegowe (różnica vs sigmoidGradient z ex4)
- **Gradient po embeddingu**: dlaczego to jest `scatter-add`, nie pełne mnożenie — to jedyny gradient, którego nie znajdziesz ani w ex4, ani w ml-applications. Własne wyprowadzenie jest obowiązkowe.
- Reguła łańcuchowa przez cały graf — wypisz krok po kroku, analogicznie do diagramu z ex4

## Metryki i ewaluacja

Jedna funkcja `evaluate.m`, ten sam zbiór testowy dla wszystkich etapów.

Raportuj zawsze:
- **Precision, Recall, F1 per klasa** (NONE, COMMA, PERIOD)
- **Macro-F1** (średnia F1 po klasach — nie waż klas)
- **Confusion matrix** 3×3
- **Accuracy** tylko jako kontekst, nie jako główna metryka

## Typowe pułapki

- **Dominacja NONE**: bez wag klas model osiąga ~85% accuracy przewidując zawsze NONE. Wagi klas w CE są obowiązkowe.
- **Leak przez zdania**: jeśli split robisz po zdaniach zamiast po dokumentach, frazy z jednego rozdziału trafią do train i test. Split po plikach.
- **Indexing off-by-one**: etykieta po ostatnim słowie dokumentu — musisz jakoś obsłużyć (PAD albo wyrzucić).
- **Octave SparseMatrix**: one-hot wejście trzymaj jako indeksy, nie jako macierz rzadką — lookup przez indeksowanie wiersza jest szybszy i czytelniejszy.
- **Inicjalizacja wag**: Xavier/He, nie losowe z N(0,1). Inaczej ReLU umiera.
- **Gradient explosion w długich dokumentach**: w MLP nie problem, ale clip i tak warto dodać dla higieny.

## Dalsze etapy (outlook, do decyzji po Etapie 1)

Po zamknięciu Etapu 1 masz własny punktuator oparty na MLP — mapa rozszerzeń:

- **Etap 2a** — bi-LSTM: wprowadza BPTT, gradient przez czas, gates. Duży skok konceptualny. *To jest twój pierwszy kontakt z pamięcią sekwencyjną — w żadnym dotychczasowym projekcie takiej nie miałeś.*
- **Etap 2b** — mini-Transformer encoder: self-attention od podstaw, positional encoding. Matematycznie piękniejszy, implementacyjnie prostszy od LSTM.
- **Etap 3** — wyjście poza NONE/COMMA/PERIOD: pytajniki, wykrzykniki, średniki. Reużywalna infrastruktura.
- **Etap 4** — multi-task: jednocześnie interpunkcja + wielkie litery (truecasing).
- **Etap 5 (opcjonalny)** — deploy jako usługa REST, analogicznie do architektury z `ml-login` (`collector-service`/`learning-service`/`validator-service`). Skoro pracujesz na co dzień w Spring Boot, to naturalne rozszerzenie — Octave trenuje, Java serwuje (wagi exportowane z `.mat` do JSON lub binary).

Decyzja po Etapie 1, na podstawie tego, co cię wciągnie.

## Źródła danych

- **Wolne Lektury**: https://wolnelektury.pl/katalog/ — pobierasz pliki .txt, licencja wolna
- **API**: https://wolnelektury.pl/api/ — można pobrać programatycznie

## Definicja ukończenia każdego etapu

Etap 0: działający `baseline_ngram.m` + raport F1 na zbiorze testowym w `notes/etap-0-wyniki.md`.

Etap 1: gradient check zalicza, sieć trenuje stabilnie, macro-F1 > macro-F1 baseline'u o co najmniej 10 punktów procentowych, wyprowadzenia w `notes/` kompletne.
