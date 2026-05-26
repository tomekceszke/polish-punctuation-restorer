%% Polish Punctuation Restorer
%% Author: Tomasz Ceszke 2026

addpath('..');

X_idx = ones(5000, 5);
E = rand(5000, 50);
W1 = rand(128, 250);
b1 = rand(128, 1);
W2 = rand(3, 128);
b2 = rand(3, 1);
k = 2;

[probs, cache] = mlp_forward(X_idx, E, W1, b1, W2, b2, k);

assert(size(probs),[5000,3]);
% softmax smoke assertions:
assert(all(probs(:) > 0));
assert(all(abs(sum(probs, 2) - 1) < 1e-9));
