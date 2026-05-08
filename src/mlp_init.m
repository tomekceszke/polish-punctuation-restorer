%% Polish Punctuation Restorer
%% Author: Tomasz Ceszke 2026

% Initialise all MLP weight matrices and bias vectors.
% V - vocab size
% d - embedding dim
% h - hidden units
% k - context radius (window = 2k+1 words)
% c - number of output classes
function [E, W1, b1, W2, b2] = mlp_init(V, d, h, k, c)

E  = randn(V, d) * 0.01;                           % embedding matrix — maps word index to d-dim vector; small random, no He needed (no ReLU on embeddings)
W1 = randn(h, (2*k+1) * d) * sqrt(2/((2*k+1)*d));  % first linear layer weights — He init (ReLU zeroes ~half neurons, scale up to compensate)
b1 = zeros(h, 1);                                  % first layer bias
W2 = randn(c, h) * sqrt(2/h);                      % second linear layer weights — He init, same reason as W1
b2 = zeros(c, 1);                                  % second layer bias

end