function tokens = tokenize(path)
    fid = fopen(path, 'r');
    text = fread(fid, '*char')';
    text = lower(text);
    text = regexprep(text, '[^a-ząćęłńóśźż\s,.]', '');  
    tokens = strsplit(text);                               
    tokens = strtrim(tokens);
    fclose(fid);
end