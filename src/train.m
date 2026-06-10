%% Polish Punctuation Restorer
%% Author: Tomasz Ceszke 2026

clear;
close all;
more off;

source('config/settings.m');
addpath('lib');

%   1. load train.mat
printf('Loading data...\n');
load '../data/processed/train.mat'
printf('Loaded training %d tokens and %d labels\n', length(train_words), length(train_labels));

%   2. vocab
printf('Loading vocabulary...\n');
if ~exist('../data/processed/vocab.mat', 'file')
    vocab = build_vocab(train_words, C_V);
    save '../data/processed/vocab.mat' vocab;
else
    load '../data/processed/vocab.mat';
end
printf('Loaded %d top unique words\n', length(vocab));

%   3. word indices
printf('Getting training word indices...\n');
train_word_indices = get_word_indices(train_words, vocab);

%   4. build_windows → X_idx [N×5]
printf('Building windows...\n');
[X_idx, y] = build_windows(train_word_indices, train_labels, C_K);



printf('All done\n');