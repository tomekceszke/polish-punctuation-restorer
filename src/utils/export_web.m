%% Polish Punctuation Restorer
%% Author: Tomasz Ceszke 2026

%   Exports the trained model for the website — the browser runs the same forward pass in JavaScript,
%   so the weights have to leave Octave as a plain little-endian float32 blob it can fetch and view
%   through a Float32Array. GitHub Pages serves static files only; there is no backend to load .mat.
%
%   Writes ../web/model/: weights.bin, vocab.txt, meta.json
%   Run from src/:  octave-cli utils/export_web.m

clear;
more off;

source('config/settings.m');
addpath('lib');

model_path = '../data/processed/model.mat';
vocab_path = '../data/processed/vocab.mat';
out_dir    = '../web/model';

if exist(model_path, 'file') != 2
    error('model not found at %s — run train.m first', model_path);
end
if exist(vocab_path, 'file') != 2
    error('vocabulary not found at %s — run train.m first', vocab_path);
end

load(model_path);
load(vocab_path);

%   shapes must agree with settings.m — a stale model.mat would silently ship wrong offsets
V = length(vocab);
if V != C_V
    error('vocab has %d words, settings.m says C_V = %d', V, C_V);
end
if rows(best_E) != V + 1 || columns(best_E) != C_D
    error('best_E is %dx%d, expected %dx%d', rows(best_E), columns(best_E), V + 1, C_D);
end
if rows(best_W1) != C_H || columns(best_W1) != (2 * C_K + 1) * C_D
    error('best_W1 is %dx%d, expected %dx%d', ...
          rows(best_W1), columns(best_W1), C_H, (2 * C_K + 1) * C_D);
end
if numel(best_b1) != C_H
    error('best_b1 has %d elements, expected %d', numel(best_b1), C_H);
end
if rows(best_W2) != 3 || columns(best_W2) != C_H
    error('best_W2 is %dx%d, expected 3x%d', rows(best_W2), columns(best_W2), C_H);
end
if numel(best_b2) != 3
    error('best_b2 has %d elements, expected 3', numel(best_b2));
end

if exist(out_dir, 'dir') != 7
    mkdir(out_dir);
end

%   weights.bin — five tensors back to back, row-major float32, little-endian.
%   Octave writes column-major, so every matrix is transposed first: fwrite of X.' lays X out row by row.
bin_path = [out_dir '/weights.bin'];
fid = fopen(bin_path, 'w', 'ieee-le');
if fid < 0
    error('cannot write %s', bin_path);
end
fwrite(fid, single(best_E.'),  'single');
fwrite(fid, single(best_W1.'), 'single');
fwrite(fid, single(best_b1.'), 'single');
fwrite(fid, single(best_W2.'), 'single');
fwrite(fid, single(best_b2.'), 'single');
fclose(fid);

n_E  = numel(best_E);
n_W1 = numel(best_W1);
n_b1 = numel(best_b1);
n_W2 = numel(best_W2);
n_b2 = numel(best_b2);
%   byte offset of each tensor inside the blob — the JS loader slices views at exactly these positions
off_E  = 0;
off_W1 = off_E  + 4 * n_E;
off_b1 = off_W1 + 4 * n_W1;
off_W2 = off_b1 + 4 * n_b1;
off_b2 = off_W2 + 4 * n_W2;
total_bytes = off_b2 + 4 * n_b2;

written = dir(bin_path).bytes;
if written != total_bytes
    error('weights.bin is %d bytes, expected %d', written, total_bytes);
end

%   vocabulary as one word per line; line number = model index, <UNK> is index length(vocab)+1
txt_path = [out_dir '/vocab.txt'];
fid = fopen(txt_path, 'w');
if fid < 0
    error('cannot write %s', txt_path);
end
for i = 1 : V
    fprintf(fid, '%s\n', vocab{i});
end
fclose(fid);

%   self-test — run the real pipeline here so the browser can prove its port still matches Octave
self_input = 'wiosna przyszła nagle śnieg stopniał w ciągu jednej nocy a rzeka wylała na łąki';
tokens = tokenize(self_input);
[words, ~] = labelize(tokens);
word_indices = get_word_indices(words, vocab);
unk_index = V + 1;
word_indices = [repmat(unk_index, 1, C_K), word_indices, repmat(unk_index, 1, C_K)];
[X_idx, ~] = build_windows(word_indices, ones(length(word_indices), 1), C_K);
probs = mlp_forward(X_idx, best_E, best_W1, best_b1, best_W2, best_b2, C_K);
[~, y_pred] = max(probs, [], 2);
label_marks = {'', ',', '.'};
out_words = {};
for i = 1 : length(words)
    out_words{end + 1} = [words{i} label_marks{y_pred(i)}];
end
self_output = post_process(strjoin(out_words, ' '));

json_path = [out_dir '/meta.json'];
fid = fopen(json_path, 'w');
if fid < 0
    error('cannot write %s', json_path);
end
fprintf(fid, '{\n');
fprintf(fid, '  "exported": "%s",\n', datestr(now(), 'yyyy-mm-dd'));
fprintf(fid, '  "V": %d,\n', V);
fprintf(fid, '  "d": %d,\n', C_D);
fprintf(fid, '  "h": %d,\n', C_H);
fprintf(fid, '  "k": %d,\n', C_K);
fprintf(fid, '  "unk": %d,\n', unk_index);
fprintf(fid, '  "classes": ["", ",", "."],\n');
fprintf(fid, '  "bytes": %d,\n', total_bytes);
fprintf(fid, '  "tensors": {\n');
fprintf(fid, '    "E":  { "offset": %d, "rows": %d, "cols": %d },\n', off_E,  rows(best_E),  columns(best_E));
fprintf(fid, '    "W1": { "offset": %d, "rows": %d, "cols": %d },\n', off_W1, rows(best_W1), columns(best_W1));
fprintf(fid, '    "b1": { "offset": %d, "rows": %d, "cols": %d },\n', off_b1, 1, n_b1);
fprintf(fid, '    "W2": { "offset": %d, "rows": %d, "cols": %d },\n', off_W2, rows(best_W2), columns(best_W2));
fprintf(fid, '    "b2": { "offset": %d, "rows": %d, "cols": %d }\n',  off_b2, 1, n_b2);
fprintf(fid, '  },\n');
fprintf(fid, '  "selfTest": {\n');
fprintf(fid, '    "input": "%s",\n', self_input);
fprintf(fid, '    "output": "%s"\n', self_output);
fprintf(fid, '  }\n');
fprintf(fid, '}\n');
fclose(fid);

printf('exported to %s\n', out_dir);
printf('  weights.bin  %d bytes (%d floats)\n', total_bytes, n_E + n_W1 + n_b1 + n_W2 + n_b2);
printf('    E   %5d x %3d  @ %8d\n', rows(best_E),  columns(best_E),  off_E);
printf('    W1  %5d x %3d  @ %8d\n', rows(best_W1), columns(best_W1), off_W1);
printf('    b1  %5d x %3d  @ %8d\n', 1, n_b1, off_b1);
printf('    W2  %5d x %3d  @ %8d\n', rows(best_W2), columns(best_W2), off_W2);
printf('    b2  %5d x %3d  @ %8d\n', 1, n_b2, off_b2);
printf('  vocab.txt    %d words (<UNK> = index %d)\n', V, unk_index);
printf('  meta.json    self-test: %s\n', self_output);
