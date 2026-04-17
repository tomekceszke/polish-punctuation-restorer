%% Author: Tomasz Ceszke 2026

clear;
close all;
more off;

addpath('lib');

books = {'../data/raw/lalka.txt', '../data/raw/chlopi.txt'};

tokens = {};

for i = 1:length(books)
    printf('Loading: %s\n', books{i});
    prev = length(tokens);
    tokens = [tokens, tokenize(books{i})];
    printf('  -> %d tokens\n', length(tokens) - prev);
end

printf('\nTotal tokens: %d\n', length(tokens));

printf('Labelizing...\n');
[words, labels] = labelize(tokens);
printf('Done. %d words labeled.\n\n', length(words));

save ../data/processed/data.mat words labels