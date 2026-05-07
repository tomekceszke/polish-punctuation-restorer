%% Polish Punctuation Restorer
%% Author: Tomasz Ceszke 2026

C_LABELS = struct('NONE', 1, 'COMMA', 2, 'PERIOD', 3);

C_TRAIN_BOOKS = {
  '../data/raw/chlopi.txt',
  '../data/raw/lalka.txt',
  '../data/raw/ziemia-obiecana.txt',
  '../data/raw/nad-niemnem.txt',
  '../data/raw/kafka-proces.txt',
  '../data/raw/przedwiosnie.txt',
  '../data/raw/moralnosc-pani-dulskiej.txt',
  '../data/raw/saint-exupery-maly-ksiaze.txt',
  '../data/raw/orwell-rok-1984.txt',
};

C_TEST_BOOKS = {
  '../data/raw/syzyfowe-prace.txt',
  '../data/raw/tajemniczy-ogrod.txt',
};

C_V= 5000; % vocab size
