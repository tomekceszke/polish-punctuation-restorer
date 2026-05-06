%% Polish Punctuation Restorer
%% Author: Tomasz Ceszke 2026

function word_indices = get_word_indices(words, vocab)
  [~, word_indices] = ismember(words, vocab);
  word_indices(word_indices == 0) = length(vocab) + 1;
endfunction