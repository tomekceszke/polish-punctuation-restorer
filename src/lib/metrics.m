%% Polish Punctuation Restorer
%% Author: Tomasz Ceszke 2026

function [confusion_matrix, precision, recall, f1] = metrics(y, y_pred)
    
    n_classes = numel(unique(y));

    confusion_matrix = zeros(n_classes);
    for sample_idx = 1:length(y) - 1
        act = y(sample_idx);
        pred = y_pred(sample_idx);
        confusion_matrix(act, pred) += 1;
    end

    tp = zeros(n_classes, 1);
    fp = zeros(n_classes, 1);
    fn = zeros(n_classes, 1);
    tn = zeros(n_classes, 1);
    precision = zeros(n_classes, 1);
    recall = zeros(n_classes, 1);
    f1 = zeros(n_classes, 1);

    for class_idx = 1:n_classes
        tp(class_idx) = confusion_matrix(class_idx, class_idx);
        fp(class_idx) = sum(confusion_matrix(:, class_idx)) - tp(class_idx);
        fn(class_idx) = sum(confusion_matrix(class_idx,:)) - tp(class_idx);
        tn(class_idx) = sum(sum(confusion_matrix)) - tp(class_idx) - fp(class_idx) - fn(class_idx);
        precision(class_idx) = tp(class_idx) / (tp(class_idx) + fp(class_idx));
        recall(class_idx) = tp(class_idx) / (tp(class_idx) + fn(class_idx));
        f1(class_idx) = 2 * precision(class_idx) * recall(class_idx) / (precision(class_idx) + recall(class_idx));
    end

end