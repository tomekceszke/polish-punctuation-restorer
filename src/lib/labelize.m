function [words, labels] = labelize(tokens)
    words = {};
    labels = [];

    for i = 1:length(tokens)
        word = tokens{i};

        if word(end) == ','
            words{end + 1} = word(1:end - 1);
            labels(end + 1) = 1;
        elseif word(end) == '.'
            words{end + 1} = word(1:end - 1);
            labels(end + 1) = 2;
        else
            words{end + 1} = word;
            labels(end + 1) = 0;
        end

    end

end
