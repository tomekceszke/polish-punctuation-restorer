%% Polish Punctuation Restorer
%% Author: Tomasz Ceszke 2026

function tokens = tokenize(path)
    fid = fopen(path, 'r');
    if fid == -1
        error('Cannot open file: %s', path);
    end
    text = fread(fid, '*char')';
    text = lower(text);
    text = regexprep(text, '[^a-ząćęłńóśźż\s,.]', '');  
    tokens = strsplit(text);                               
    tokens = strtrim(tokens);
    fclose(fid);
end