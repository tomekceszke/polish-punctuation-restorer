%% Polish Punctuation Restorer
%% Author: Tomasz Ceszke 2026

addpath('../lib');

% given
y =         [1 1 2 1 3 2];
y_pred =    [2 2 2 1 3 1];

confusion_matrix_valid = zeros(3);
confusion_matrix_valid(1,2) = 2;
confusion_matrix_valid(2,2) = 1;
confusion_matrix_valid(1,1) = 1;
confusion_matrix_valid(3,3) = 1;
confusion_matrix_valid(2,1) = 1;

precision_valid = [0.5 ; 1/3 ; 1];
recall_valid = [1/3 ; 1/2 ; 1];
f1_valid = [0.4 ; 0.4 ; 1];

% when
[confusion_matrix, precision, recall, f1] = metrics(y, y_pred);

% then
assert(confusion_matrix, confusion_matrix_valid);
assert(precision, precision_valid, 1e-9);
assert(recall, recall_valid, 1e-9)
assert(f1, f1_valid, 1e-9);



