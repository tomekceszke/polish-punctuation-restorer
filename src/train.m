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

%   2. load/re-create vocab
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
N = rows(X_idx);

%   5. initialization
[E, W1, b1, W2, b2] = mlp_init(rows(vocab)+1, C_D, C_H, C_K, numfields(C_LABELS));

%   6. main training loop
for epoch = 1:C_EPOCHS
    % each epoch - random order
    ind = randperm(N);
    X_idx_r = X_idx(ind, :);
    y_r = y(ind, :);

    for start = 1:C_BATCH:N
        batch_end = min(start + C_BATCH - 1, N);
        X_batch = X_idx_r(start:batch_end, :);
        y_batch = y_r(start:batch_end, :);
        [probs, cache] = mlp_forward(X_batch, E, W1, b1, W2, b2, C_K);
        [loss] = mlp_loss(probs, y_batch);
    end

end

printf('All done\n');
