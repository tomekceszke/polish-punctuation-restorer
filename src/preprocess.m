%% Polish Punctuation Restorer
%% Author: Tomasz Ceszke 2026

clear;
close all;
more off;

source('config/settings.m');
addpath('lib');

books = C_TRAINING_BOOKS;

tokens = {};
doc_ids = [];

for i = 1:length(books)
    printf('Loading: %s\n', books{i});
    prev_length = length(tokens);
    curr = tokenize(books{i});
    doc_ids(prev_length+1: prev_length + length(curr)) = i;
    tokens = [tokens, curr];
    printf('  -> %d tokens\n', length(tokens) - prev_length);
end

printf('\nTotal tokens: %d\n', length(tokens));

printf('Labelizing...\n');
[words, labels] = labelize(tokens);
printf('Done. %d words labeled.\n\n', length(words));

save ../data/processed/data.mat words labels doc_ids