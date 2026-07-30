%% Polish Punctuation Restorer
%% Author: Tomasz Ceszke 2026

function tokens = tokenize(text)
    text = lower(text);
    text = regexprep(text, '[^a-ząćęłńóśźż\s,.]', '');
    tokens = strsplit(text);
    tokens = strtrim(tokens);
    tokens = tokens(~cellfun('isempty', tokens));
end
