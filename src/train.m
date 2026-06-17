%% Polish Punctuation Restorer
%% Author: Tomasz Ceszke 2026

clear;
close all;
more off;

source('config/settings.m');
addpath('lib');

%   load train.mat
printf('Loading data...\n');
load '../data/processed/train.mat'
printf('Loaded training %d tokens and %d labels\n', length(train_words), length(train_labels));

%   load/re-create vocab
printf('Loading vocabulary...\n');

if ~exist('../data/processed/vocab.mat', 'file')
    vocab = build_vocab(train_words, C_V);
    save '../data/processed/vocab.mat' vocab;
else
    load '../data/processed/vocab.mat';
end

printf('Loaded %d top unique words\n', length(vocab));

%   word indices
printf('Getting training word indices...\n');
train_word_indices = get_word_indices(train_words, vocab);

%   build_windows → X_idx [N×5]
printf('Building windows...\n');
[X_idx, y] = build_windows(train_word_indices, train_labels, C_K);
N = rows(X_idx);

%   initialization
[E, W1, b1, W2, b2] = mlp_init(length(vocab)+1, C_D, C_H, C_K, numfields(C_LABELS));

%   weights
class_counts = accumarray(y, 1); % [3x1]
class_w = N ./ (numfields(C_LABELS) * class_counts); % [3x1]
w_all = class_w(y); % [Nx1] per-sample weight: each class index in y mapped to its class weight

%   main training loop
for epoch = 1:C_EPOCHS
    % each epoch - random order
    ind = randperm(N);
    X_idx_r = X_idx(ind, :);
    y_r = y(ind, :);
    w_r = w_all(ind); % weights also need to be shuffled in the same way
    epoch_loss = 0;
    epoch_tic = tic;

    for start = 1:C_BATCH:N
        batch_end = min(start + C_BATCH - 1, N);
        X_batch = X_idx_r(start:batch_end, :);
        y_batch = y_r(start:batch_end, :);
        w_batch = w_r(start:batch_end, :);
        % forward
        [probs, cache] = mlp_forward(X_batch, E, W1, b1, W2, b2, C_K);
        % loss
        [loss] = mlp_loss(probs, y_batch, w_batch);
        epoch_loss = epoch_loss + loss;
        % backward
        [dW2, db2, dW1, db1, dE] = mlp_backward(probs, cache, E, X_batch, y_batch, w_batch, C_K, W1, W2);
        % weights update (SGD)
        W1 = W1 - (C_LR * dW1); % [128 x 250]
        b1 = b1 - (C_LR * db1); % [128 x 1]
        W2 = W2 - (C_LR * dW2); % [3 x 128]
        b2 = b2 - (C_LR * db2); % [3 x 1]
        E = E - (C_LR * dE); % [V x 50]
    end

    epoch_time = toc(epoch_tic);
    n_batches = ceil(N / C_BATCH);
    printf('Epoch %2d/%d  train loss = %.4f  (%.1fs, %.2f ms/batch)\n', ...
           epoch, C_EPOCHS, epoch_loss / n_batches, ...
           epoch_time, 1000 * epoch_time / n_batches);
end

printf('All done\n');
