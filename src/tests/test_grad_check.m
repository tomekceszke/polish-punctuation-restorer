%% Polish Punctuation Restorer
%% Author: Tomasz Ceszke 2026

addpath('..');

V = 20;
N = 5;
c = 3;
h = 4;
d = 3;
k = 2;

X_idx = randi(V, N, 2 * k + 1);
y = randi(c, N, 1);

[E, W1, b1, W2, b2] = mlp_init(V, d, h, k, c);
w = ones(N, 1);

[probs, cache] = mlp_forward(X_idx, E, W1, b1, W2, b2, k);

[dW2, db2, dW1, db1, dE] = mlp_backward(probs, cache, E, X_idx, y, w, k, W1, W2);

eps = 1e-5;


% W2
orig = W2(1, 1);
W2(1, 1) = orig + eps;
probs_plus = mlp_forward(X_idx, E, W1, b1, W2, b2, k);
L_plus = mlp_loss(probs_plus, y, w);

W2(1, 1) = orig - eps;
probs_minus = mlp_forward(X_idx, E, W1, b1, W2, b2, k);
L_minus = mlp_loss(probs_minus, y, w);

W2(1, 1) = orig;
grad_num = (L_plus - L_minus) / (2 * eps);

% relative error = |analytic - numerical| / (|analytic| + |numerical|)
rel_err_W2 = abs(dW2(1, 1) - grad_num)/(abs(dW2(1, 1)) + abs(grad_num));
assert(rel_err_W2 < eps);

% W1
orig = W1(1, 1);
W1(1, 1) = orig + eps;
probs_plus = mlp_forward(X_idx, E, W1, b1, W2, b2, k);
L_plus = mlp_loss(probs_plus, y, w);

W1(1, 1) = orig - eps;
probs_minus = mlp_forward(X_idx, E, W1, b1, W2, b2, k);
L_minus = mlp_loss(probs_minus, y, w);

W1(1, 1) = orig;
grad_num = (L_plus - L_minus) / (2 * eps);

% relative error = |analytic - numerical| / (|analytic| + |numerical|)
rel_err_W1 = abs(dW1(1, 1) - grad_num)/(abs(dW1(1, 1)) + abs(grad_num));
assert(rel_err_W1 < eps);

% b1
orig = b1(1);
b1(1) = orig + eps;
probs_plus = mlp_forward(X_idx, E, W1, b1, W2, b2, k);
L_plus = mlp_loss(probs_plus, y, w);

b1(1) = orig - eps;
probs_minus = mlp_forward(X_idx, E, W1, b1, W2, b2, k);
L_minus = mlp_loss(probs_minus, y, w);

b1(1) = orig;
grad_num = (L_plus - L_minus) / (2 * eps);

% relative error = |analytic - numerical| / (|analytic| + |numerical|)
rel_err_b1 = abs(db1(1) - grad_num)/(abs(db1(1)) + abs(grad_num));
assert(rel_err_b1 < eps);

% b2
orig = b2(1);
b2(1) = orig + eps;
probs_plus = mlp_forward(X_idx, E, W1, b1, W2, b2, k);
L_plus = mlp_loss(probs_plus, y, w);

b2(1) = orig - eps;
probs_minus = mlp_forward(X_idx, E, W1, b1, W2, b2, k);
L_minus = mlp_loss(probs_minus, y, w);

b2(1) = orig;
grad_num = (L_plus - L_minus) / (2 * eps);

% relative error = |analytic - numerical| / (|analytic| + |numerical|)
rel_err_b2 = abs(db2(1) - grad_num)/(abs(db2(1)) + abs(grad_num));
assert(rel_err_b2 < eps);

% E
orig = E(X_idx(1,1), 1);
E(X_idx(1,1), 1) = orig + eps;
probs_plus = mlp_forward(X_idx, E, W1, b1, W2, b2, k);
L_plus = mlp_loss(probs_plus, y, w);

E(X_idx(1,1), 1) = orig - eps;
probs_minus = mlp_forward(X_idx, E, W1, b1, W2, b2, k);
L_minus = mlp_loss(probs_minus, y, w);

E(X_idx(1,1), 1) = orig;
grad_num = (L_plus - L_minus) / (2 * eps);

% relative error = |analytic - numerical| / (|analytic| + |numerical|)
rel_err_E = abs(dE(X_idx(1,1), 1) - grad_num)/(abs(dE(X_idx(1,1), 1)) + abs(grad_num));
assert(rel_err_E < eps);
