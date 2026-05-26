%% Polish Punctuation Restorer
%% Author: Tomasz Ceszke 2026

addpath('..');
[E, W1, b1, W2, b2] = mlp_init(5000, 50, 128, 2, 3);
assert(isequal(size(E),  [5000, 50]));
assert(isequal(size(W1), [128, 250]));
assert(isequal(size(b1), [128, 1]));
assert(isequal(size(W2), [3, 128]));
assert(isequal(size(b2), [3, 1]));
