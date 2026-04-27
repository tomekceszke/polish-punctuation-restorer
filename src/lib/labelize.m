%% Polish Punctuation Restorer
%% Author: Tomasz Ceszke 2026

function [words, labels] = labelize(tokens)
    source('config/settings.m');

    words = {};
    labels = [];

    for i = 1:length(tokens)
        word = tokens{i};

        if word(end) == ','
            words{end + 1} = word(1:end - 1);
            labels(end + 1) = C_LABELS.COMMA;
        elseif word(end) == '.'
            words{end + 1} = word(1:end - 1);
            labels(end + 1) = C_LABELS.PERIOD;
        else
            words{end + 1} = word;
            labels(end + 1) = C_LABELS.NONE;
        end

    end

end
