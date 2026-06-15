%% Polish Punctuation Restorer
%% Author: Tomasz Ceszke 2026

clear;
close all;
more off;

source('config/settings.m');
addpath('lib');

printf('Processing training data...\n\n');
[train_words, train_labels] = process(C_TRAIN_BOOKS);
printf('Done. %d words labeled.\n\n', length(train_words));
save '../data/processed/train.mat' train_words train_labels;
printf('Data saved.\n\n');

printf('Processing validating data...\n\n');
[val_words, val_labels] = process(C_VAL_BOOKS);
printf('Done. %d words labeled.\n\n', length(val_words));
save '../data/processed/val.mat' val_words val_labels;
printf('Data saved.\n\n');

printf('Processing testing data...\n\n');
[test_words, test_labels] = process(C_TEST_BOOKS);
printf('Done. %d words labeled.\n\n', length(test_words));
save '../data/processed/test.mat' test_words test_labels;
printf('Data saved.\n\n');

printf('All done.\n\n');
