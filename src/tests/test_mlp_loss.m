%% Polish Punctuation Restorer
%% Author: Tomasz Ceszke 2026

addpath('..');
[loss] = mlp_loss([1, 0, 0], [1], 1);

assert(loss, 0);

[loss] = mlp_loss([0, 1, 0], [1], 1);

assert(loss, inf);