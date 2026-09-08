%% Polish Punctuation Restorer
%% Author: Tomasz Ceszke 2026

%   Exports the processed (word, label) pairs as CSV, one file per split, for the
%   Hugging Face dataset repo. Words are quoted, so a stray internal separator cannot
%   break a row.
%
%   Run from src/:  octave-cli utils/export_dataset.m

clear;
more off;

out_dir = '../hf-dataset/data';
if exist(out_dir, 'dir') != 7
    mkdir(out_dir);
end

splits = {'train', 'val', 'test'};

for s = 1 : length(splits)
    split = splits{s};
    in_path = ['../data/processed/' split '.mat'];

    if exist(in_path, 'file') != 2
        error('%s not found — run preprocess.m first', in_path);
    end

    data = load(in_path);
    words  = data.([split '_words']);
    labels = data.([split '_labels']);

    out_path = [out_dir '/' split '.csv'];
    fid = fopen(out_path, 'w');
    if fid < 0
        error('cannot write %s', out_path);
    end

    fprintf(fid, 'word,label\n');
    for i = 1 : length(words)
        fprintf(fid, '"%s",%d\n', words{i}, labels(i));
    end
    fclose(fid);

    printf('%-5s %8d rows -> %s\n', split, length(words), out_path);
end
