%% Polish Punctuation Restorer
%% Author: Tomasz Ceszke 2026

function [X_idx, y] = build_windows(word_indices, labels, k)
    X_idx = zeros(length(word_indices) - 2 * k, 2 * k + 1);
    y = zeros(length(word_indices) - 2*k, 1);

    for i = k + 1:length(word_indices) - k
        row = i - k;
        X_idx(row, :) = word_indices(i - k : i + k);
        y(row) = labels(i);
    end

end
