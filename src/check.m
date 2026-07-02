%% Polish Punctuation Restorer
%% Author: Tomasz Ceszke 2026

clear;
close all;
more off;

source('config/settings.m');
addpath('lib');

printf('Loading data...\n');
%   load test.mat
load '../data/processed/test.mat'
printf('Loaded test %d tokens and %d labels\n', length(test_words), length(test_labels));

%   load vocab
printf('Loading vocabulary...\n');
load '../data/processed/vocab.mat';
printf('Loaded %d top unique words\n', length(vocab));

%   load model
printf('Loading model...\n');
load '../data/processed/model.mat';
printf('Model loaded\n');

%   word indices
printf('Getting test word indices...\n');
test_word_indices = get_word_indices(test_words, vocab);

%   build_windows
printf('Building test windows...\n');
[X_idx_test, y_test] = build_windows(test_word_indices, test_labels, C_K);
N_test = rows(X_idx_test);

%   forward
probs_test = mlp_forward(X_idx_test, best_E, best_W1, best_b1, best_W2, best_b2, C_K);
%   metrics
[~, y_pred] = max(probs_test, [], 2);
[cm, precision, recall, f1] = metrics(y_test, y_pred);
macro_f1_test = mean(f1);

printf('\nTest set: %d windows\n\n', N_test);
printf('Confusion matrix (rows = actual, cols = predicted):\n');
disp(cm);
printf('\n%-8s %10s %10s %10s\n', 'Class', 'Precision', 'Recall', 'F1');
class_names = fieldnames(C_LABELS);
for c = 1:numel(class_names)
    printf('%-8s %10.4f %10.4f %10.4f\n', class_names{c}, precision(c), recall(c), f1(c));
end
printf('\nMacro-F1 = %.4f\n', macro_f1_test);
