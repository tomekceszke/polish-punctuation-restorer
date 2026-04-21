%% Polish Punctuation Restorer
%% Author: Tomasz Ceszke 2026

clear;
close all;
more off;

source('lib/constants.m');

% load tokens
printf('Loading...\n');
load '../data/processed/data.mat'
printf('Loaded %d tokens and %d labels\n', length(words), length(labels));

counter = containers.Map();


printf('Main loop...\n');
% for i = 1:length(words) - 1
%     w1 = words{i};
%     w2 = words{i + 1};
%     l = labels(i);
%     composite_key = strcat(w1, '_', w2, '_', num2str(l));

%     if counter.isKey(composite_key)
%         counter(composite_key) = counter(composite_key) + 1;
%     else
%         counter(composite_key) = 1;
%     end

% end
printf('Loop done\n');
