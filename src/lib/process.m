%% Polish Punctuation Restorer
%% Author: Tomasz Ceszke 2026

function [words, labels] = process(books)
    tokens = {};
    for i = 1:length(books)
        tokens = [tokens, tokenize(books{i})];
    end
    [words, labels] = labelize(tokens);
end
