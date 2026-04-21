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

[unique_words, ~, unique_words_positions] = unique(words);
word_counts = accumarray(unique_words_positions, 1);
[~, unique_words_indices] = sort(word_counts, 'descend');

top_unique_words_indices = unique_words_indices(1:C_CUT_OFF_WORDS);

top_unique_words = unique_words(top_unique_words_indices);
printf('Vocabulary: %d words (unknown words -> index %d)\n', C_CUT_OFF_WORDS, C_CUT_OFF_WORDS + 1);

[~, word_indices] = ismember(words, top_unique_words);

word_indices(word_indices == 0) = C_CUT_OFF_WORDS + 1;

oov_count = sum(word_indices == C_CUT_OFF_WORDS + 1);
printf('Mapped %d tokens (%d OOV, %.1f%%)\n', length(word_indices), oov_count, 100 * oov_count / length(word_indices));

save ../data/processed/vocab.mat word_indices
printf('Saved vocab.mat\n');
