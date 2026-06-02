%% Polish Punctuation Restorer
%% Author: Tomasz Ceszke 2026

function [dW2, db2, dW1, db1, dE] = mlp_backward(probs, cache, E, X_idx, y, w, k, W1, W2)
    N = rows(probs);
    Y = zeros(N, 3);
    Y(sub2ind(size(Y), (1:N)', y)) = 1;

    d2 = (probs - Y) .* w / N;
    dW2 = d2' * cache.a1;
    db2 = sum(d2, 1)';

    d1 = ((W2' * d2') .* (cache.s1 > 0)')';
    dW1 = d1' * cache.x_embed;
    db1 = sum(d1, 1)';

    V = rows(E);
    d = columns(E);
    dE = zeros(V, d);
    dx_embed = d1 * W1;

    for j = 1:2 * k + 1
        % scatter add
        for i = 1:N
            dE(X_idx(i, j), :) += dx_embed(i, (j - 1) * d + 1:j * d);
        end
    end

end
