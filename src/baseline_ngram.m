%% Polish Punctuation Restorer
%% Author: Tomasz Ceszke 2026

clear;
close all;
more off;

source('config/settings.m');
addpath('lib');

printf('Loading data...\n');
load '../data/processed/train.mat'
load '../data/processed/val.mat'
% baseline has no hyperparameters / early stopping, so the val split is irrelevant
% to it; recombine train+val to train on the full corpus train set (see ef64e8f)
train_words = [train_words(:); val_words(:)];
train_labels = [train_labels(:); val_labels(:)];
printf('Loaded training %d tokens and %d labels (train+val)\n', length(train_words), length(train_labels));
load '../data/processed/test.mat'
printf('Loaded testing %d tokens and %d labels\n', length(test_words), length(test_labels));

% vocab: built from the full train set in memory (top C_V), not the committed
% vocab.mat (which the MLP regenerated from the reduced train split)
printf('Building vocabulary from full train...\n');
vocab = build_vocab(train_words, C_V);
printf('Built %d top unique words\n', length(vocab));

% indices
printf('Getting training/testing word indices...\n');
train_word_indices = get_word_indices(train_words, vocab);
printf('Got %d training word indices\n', length(train_word_indices));
test_word_indices = get_word_indices(test_words, vocab);
printf('Got %d testing word indices\n', length(test_word_indices));

% learning (only on trained data)

% bigram counts: counter(idx1, idx2, label) = number of occurrences in corpus
counter = zeros(C_V + 1, C_V + 1, numel(fieldnames(C_LABELS)));

% accumulate bigram-label counts
printf('Counting occurrences...\n');

for i = 1:length(train_word_indices) - 1
    idx1 = train_word_indices(i);
    idx2 = train_word_indices(i + 1);
    l = train_labels(i);
    counter(idx1, idx2, l) += 1;
end

% predict most frequent label for each bigram on trained data
printf('Predicting on trained data...\n');
y_pred_trained = zeros(length(train_word_indices), 1);

for i = 1:length(train_word_indices) - 1
    idx1 = train_word_indices(i);
    idx2 = train_word_indices(i + 1);
    [~, idx] = max(counter(idx1, idx2, :));
    y_pred_trained(i) = idx;
end
% last token has no successor → no bigram prediction; trim both to the predicted range
[~, ~, ~, f1_train] = metrics(train_labels(1:end-1), y_pred_trained(1:end-1));

% predict most frequent label for each bigram on tested data
y_pred_tested = zeros(length(test_word_indices), 1);
printf('Predicting on tested data...\n');

for i = 1:length(test_word_indices) - 1
    idx1 = test_word_indices(i);
    idx2 = test_word_indices(i + 1);
    [~, idx] = max(counter(idx1, idx2, :));
    y_pred_tested(i) = idx;
end

[confusion_matrix_test, ~, ~, f1_test] = metrics(test_labels(1:end-1), y_pred_tested(1:end-1));

labels = fieldnames(C_LABELS);

printf('\nConfusion matrix (test):\n');
disp(confusion_matrix_test);

printf('%-8s %10s %10s\n', 'Class', 'Train F1', 'Test F1');
for i = 1:length(labels)
    printf('%-8s %10.4f %10.4f\n', labels{i}, f1_train(i), f1_test(i));
end
printf('%-8s %10.4f %10.4f\n', 'Macro', mean(f1_train), mean(f1_test));

printf('\nDone.\n');
