function [ reg_labels ] = conlabels( imlabels )
%CONLABELS Connected region labels
%   Detailed explanation goes here
    nhood = 4;
    nb_cluster = max(imlabels(:));
    prev_nb_reg = 0;
    reg_labels = zeros(size(imlabels));
    for n = 1:nb_cluster
        new_labels = bwlabel(imlabels==n, nhood);
        nb_curr_levels = max(new_labels(:));
        new_labels(new_labels ~= 0) = new_labels(new_labels ~= 0) + prev_nb_reg;
        prev_nb_reg = prev_nb_reg + nb_curr_levels;
        reg_labels = reg_labels + new_labels;
    end

end

