%% Polish Punctuation Restorer
%% Author: Tomasz Ceszke 2026

clear;
close all;
more off;

source('config/settings.m');
addpath('lib');

printf('Loading...\n');
load '../data/processed/data.mat'
printf('Loaded %d tokens and %d labels\n', length(words), length(labels));

word_indices = build_vocab(words, C_CUT_OFF_WORDS);
printf('Loaded %d word indices\n', length(word_indices));

% bigram counts: counter(idx1, idx2, label) = number of occurrences in corpus
counter = zeros(C_CUT_OFF_WORDS + 1, C_CUT_OFF_WORDS + 1, numel(fieldnames(C_LABELS)));
% last position left as 0 (no next word), excluded from evaluation
y_pred = zeros(length(word_indices), 1);
y_true = labels;

printf('Counting occurrences...\n');
% pass 1: accumulate bigram-label counts
for i = 1:length(word_indices)-1
    idx1 = word_indices(i);
    idx2 = word_indices(i + 1);
    l = labels(i);
    counter(idx1, idx2, l) += 1;
end

printf('Predicting...\n');
% pass 2: predict most frequent label for each bigram (trained and tested on same data)
for i = 1:length(word_indices)-1
    idx1 = word_indices(i);
    idx2 = word_indices(i + 1);
    [~, idx] = max(counter(idx1, idx2, :));
    y_pred(i) = idx;
end

printf('Done.\n');
