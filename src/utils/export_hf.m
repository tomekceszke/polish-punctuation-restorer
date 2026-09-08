%% Polish Punctuation Restorer
%% Author: Tomasz Ceszke 2026

%   Exports the trained model into portable formats for the Hugging Face model repo.
%   Octave's default `save` writes its own text format, which scipy.io.loadmat cannot read —
%   so the weights are re-saved as MATLAB v7 and the vocabulary as plain UTF-8 text.
%
%   Run from src/:  octave-cli utils/export_hf.m

clear;
more off;

model_path = '../data/processed/model.mat';
vocab_path = '../data/processed/vocab.mat';
out_dir    = '../hf';

if exist(model_path, 'file') != 2
    error('model not found at %s — run train.m first', model_path);
end
if exist(vocab_path, 'file') != 2
    error('vocabulary not found at %s — run train.m first', vocab_path);
end

load(model_path);
load(vocab_path);

if exist(out_dir, 'dir') != 7
    mkdir(out_dir);
end

%   MATLAB v7: binary, compressed, readable by scipy.io.loadmat
v7_path = [out_dir '/model_v7.mat'];
save('-v7', v7_path, 'best_E', 'best_W1', 'best_b1', 'best_W2', 'best_b2');

%   vocabulary as one word per line; line number = model index, <UNK> is index length(vocab)+1
txt_path = [out_dir '/vocab.txt'];
fid = fopen(txt_path, 'w');
if fid < 0
    error('cannot write %s', txt_path);
end
for i = 1 : length(vocab)
    fprintf(fid, '%s\n', vocab{i});
end
fclose(fid);

printf('exported to %s\n', out_dir);
printf('  model_v7.mat\n');
printf('    best_E  %d x %d\n', rows(best_E), columns(best_E));
printf('    best_W1 %d x %d\n', rows(best_W1), columns(best_W1));
printf('    best_b1 %d x %d\n', rows(best_b1), columns(best_b1));
printf('    best_W2 %d x %d\n', rows(best_W2), columns(best_W2));
printf('    best_b2 %d x %d\n', rows(best_b2), columns(best_b2));
printf('  vocab.txt   %d words (<UNK> = index %d)\n', length(vocab), length(vocab) + 1);
