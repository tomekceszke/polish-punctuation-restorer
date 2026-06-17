%% Polish Punctuation Restorer
%% Author: Tomasz Ceszke 2026


% probs - probability of occurence each of c-classes (N x c) [N x 3]
% y     - true class index per sample (N x 1)
% w     - per-sample class weight (N x 1)
% loss  - weighted average cross-entropy loss over N samples
function [loss] = mlp_loss(probs, y, w)
    
    % f(x) = -log(x) - Cross-Entropy (CE) loss
    N = size(probs, 1);
    ce = -log(probs(sub2ind(size(probs), (1:N)', y))); % (N x 1)
    wce = w .* ce;
    loss = sum(wce) / N;
end