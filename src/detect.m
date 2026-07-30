%% Polish Punctuation Restorer
%% Author: Tomasz Ceszke 2026

clear;
close all;
more off;

source('config/settings.m');
addpath('lib');

model_path = '../data/processed/model.mat';
vocab_path = '../data/processed/vocab.mat';

if exist(model_path, 'file') != 2
    error('model not found at %s — run train.m first', model_path);
end
if exist(vocab_path, 'file') != 2
    error('vocabulary not found at %s — run train.m first', vocab_path);
end

load(model_path);
load(vocab_path);

printf('Polish Punctuation Restorer — Stage 1 MLP\n');
printf('vocab %d words · context k=%d · restores , and .\n', length(vocab), C_K);
printf('empty line or "exit" quits\n\n');

while true
    try
        input_text = input('Text: ', 's');
    catch
        printf('\n');
        break;
    end
    if isempty(strtrim(input_text)) || any(strcmp(strtrim(lower(input_text)), {'exit', 'quit'}))
        break;
    end

    %   word indices
    tokens = tokenize(input_text);
    %   labelize strips any punctuation the user already typed — the model predicts it from scratch
    [words, ~] = labelize(tokens);
    if isempty(words)
        printf('\n  nothing to punctuate — Polish letters only\n\n');
        continue;
    end
    word_indices = get_word_indices(words, vocab);
    %   pad both ends with k UNK so every real word gets a window (n+2k indices -> n windows),
    %   including the last one, which is exactly where the final period belongs
    unk_index = length(vocab) + 1;
    word_indices = [repmat(unk_index, 1, C_K), word_indices, repmat(unk_index, 1, C_K)];

    %   build_windows
    [X_idx, ~] = build_windows(word_indices, ones(length(word_indices), 1), C_K);

    %   prediction
    probs = mlp_forward(X_idx, best_E, best_W1, best_b1, best_W2, best_b2, C_K);
    [~, y_pred] = max(probs, [], 2);

    %   restoration
    %   position = class index, so the order must match C_LABELS (NONE, COMMA, PERIOD)
    label_marks = {'', ',', '.'};
    out_words = {};
    for i = 1 : length(words)
        out_words{end + 1} = [words{i} label_marks{y_pred(i)}];
    end
    out_text = strjoin(out_words, ' ');

    printf('\n  %s\n\n', out_text);
    printf('%d word(s) · %d comma(s) · %d period(s)\n\n', ...
           length(words), ...
           sum(y_pred == C_LABELS.COMMA), ...
           sum(y_pred == C_LABELS.PERIOD));
end
