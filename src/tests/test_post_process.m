%% Polish Punctuation Restorer
%% Author: Tomasz Ceszke 2026

addpath('../lib');

% given / when / then

% capitalizes first letter and appends period
assert(post_process('ala ma kota'), 'Ala ma kota.');

% single character
assert(post_process('a'), 'A.');

% digit at the end is alphanumeric too
assert(post_process('ala ma kota 5'), 'Ala ma kota 5.');

% does not duplicate an existing period
assert(post_process('ala ma kota.'), 'Ala ma kota.');

% trailing comma is not alphanumeric, no period added
assert(post_process('ala ma kota,'), 'Ala ma kota,');

% inner punctuation is preserved
assert(post_process('to jest zdanie, a to drugie'), 'To jest zdanie, a to drugie.');

% every sentence after a period is capitalized too
assert(post_process('ala ma kota. a kot ma ale'), 'Ala ma kota. A kot ma ale.');
assert(post_process('raz. dwa. trzy.'), 'Raz. Dwa. Trzy.');

% a comma does not open a new sentence
assert(post_process('raz, dwa'), 'Raz, dwa.');

% surrounding whitespace is trimmed
assert(post_process('  ala ma kota  '), 'Ala ma kota.');

% empty and whitespace-only input must not crash
assert(post_process(''), '');
assert(post_process('   '), '');

% multi-byte final letter still counts as alphanumeric
assert(post_process('ala ma koń'), 'Ala ma koń.');

% KNOWN LIMITATION: multi-byte first letter is not capitalized yet
assert(post_process('ósmy dzień'), 'ósmy dzień.');
