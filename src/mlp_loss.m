%% Polish Punctuation Restorer
%% Author: Tomasz Ceszke 2026


% probs - probability of occurence each of c-classes (N x c) [N x 3]
% loss - average cross-entropy loss over N samples
function [loss] = mlp_loss(probs, y)
    
    % f(x) = -log(x) - cross-entropy loss
    N = size(probs, 1);
    loss = mean(-log(probs(sub2ind(size(probs), (1:N)', y'))));

end