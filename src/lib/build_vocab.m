%% Polish Punctuation Restorer
%% Author: Tomasz Ceszke 2026

function top_unique_words = build_vocab(words, cut_off)
  [unique_words, ~, unique_words_positions] = unique(words);
  word_counts = accumarray(unique_words_positions, 1);
  [~, unique_words_indices] = sort(word_counts, 'descend');
  top_unique_words = unique_words(unique_words_indices(1:cut_off));
endfunction
