%% Polish Punctuation Restorer
%% Author: Tomasz Ceszke 2026

function [txt] = post_process(txt)
    txt = strtrim(txt);
    if isempty(txt)
        return;
    end

    if isalnum(txt(end)) || txt(end) >= 128
        txt = [txt '.'];
    end

    capitalize_next = true;
    for i = 1 : length(txt)
        if capitalize_next && (isalnum(txt(i)) || txt(i) >= 128)
            if txt(i) < 128
                txt(i) = upper(txt(i));
            end
            capitalize_next = false;
        elseif txt(i) == '.'
            capitalize_next = true;
        end
    end
end
