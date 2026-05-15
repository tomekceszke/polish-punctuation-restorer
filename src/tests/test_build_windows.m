%% Polish Punctuation Restorer
%% Author: Tomasz Ceszke 2026

addpath('../lib');
[X, y] = build_windows([1; 2; 3; 4; 5], [11; 22; 33; 44; 55], 1);

assert(size(X), [3, 3])
assert(X(1,:), [1 2 3])
assert(y(1), 22)