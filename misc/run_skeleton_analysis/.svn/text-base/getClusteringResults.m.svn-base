function clusterMask = getClusteringResults( attr, sc_labels, mec_labels )
%GETCLUSTERÝNGRESULTS Returns clustering results ,each cluster as a mask
%   TODO : comment
%   TODO : optimize
%
% orhanf

if attr.sc_flag
%     tmp=load(['outputs\sc_results\' attr.segAlgo '\' attr.name '\meanstd\sc_cluster_labels.mat']);
%     clustering_labels = tmp.cluster_labels_1;
    clustering_labels = sc_labels;
else
%     tmp=load(['outputs\sc_mec_results\' attr.segAlgo '\' attr.name '\meanstd\' attr.name '_meanstd_meanstd_mec_labels.mat']);
%     clustering_labels = tmp.mec_labels;
    clustering_labels = mec_labels;
end

% rearrange labelmap for vegetation filtering
mec_label_map = attr.filtered_labelmap;

for m = 1: size(clustering_labels,1)
    mec_label_map(mec_label_map==m) = clustering_labels(m);
end
clusterMask = mec_label_map;
end

