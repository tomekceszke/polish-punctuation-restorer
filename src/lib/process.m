%% Polish Punctuation Restorer
%% Author: Tomasz Ceszke 2026

function [words, labels] = process(books)
    tokens = {};
    for i = 1:length(books)
        fid = fopen(books{i}, 'r');
        if fid == -1
            error('Cannot open file: %s', books{i});
        end
        text = fread(fid, '*char')';
        tokens = [tokens, tokenize(text)];
        fclose(fid);
    end
    [words, labels] = labelize(tokens);
end
