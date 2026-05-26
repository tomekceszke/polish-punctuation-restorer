%% Polish Punctuation Restorer
%% Author: Tomasz Ceszke 2026

% X_idx - words indices matrix, [N*2k+1] (N*5)
% E - embedding matrix, maps word index to d-dim vector, [V*d] (5000*50)
% W1 - first linear layer weights, [h*(2*k+1)*d] (128*250)
% b1 - bias 1 [h] (128)
% W2 - second linear layer weights, [c*h] (3*128)
% b2 - bias 2 [c] (3)
% k - context radius (window = 2k+1 words) (2)
function [probs, cache] = mlp_forward(X_idx, E, W1, b1, W2, b2, k)
    d = columns(E); % 50
    x_embed = zeros(rows(X_idx), (2 * k + 1) * d); % N*250

    for i = 1:2 * k + 1
        x_embed(:, (i - 1) * d + 1:i * d) = E(X_idx(:, i), :); % embedding lookup
    end

    s1 = x_embed * W1' + b1'; % linear 1, before ReLU, [N*128]
    a1 = max(0, s1); % ReLU [N*128]
    s2 = a1 * W2' + b2'; % linear 2, [N*3]

    z = s2 - max(s2, [], 2); % subtract max per row for numerical stability (prevents exp overflow) [N*3]
    e = exp(z); % [N*3]
    probs = e ./ sum(e, 2); % softmax, each row divided by its sum (N*3./N*1)=(N*3)

    cache = struct("x_embed", x_embed, "s1", s1, "a1", a1);


end
