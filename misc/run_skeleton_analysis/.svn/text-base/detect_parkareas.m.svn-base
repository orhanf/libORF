%====================================================================
%> @brief This function detects the park areas from the given masks.
%>
%> @author orhanf
%>
%> Change                                      Date       Performed By 
%>
%> @param attr Structure containing properties related to the input image
%> @param airfield_mask Mask containing whole airfield region
%> @param runway_mask Mask containing detected runways
%> @param taxiroute_mask Mask containing detected taxiroutes
%> @param taxiroute_surplus_mask Mask containing taxiroute candidates
%> @param sc_labels Spectral clustering label mask of the input image
%> @param mec_labels MEC label mask of the input image
%>
%> @retval parkarea_mask Binary mask of detected park areas
%> @retval parkarea_surplus_mask Binary mask of park area candidates
%> (excluding regions that are provided in parkarea_mask)
%> 
%> @example
%> airfield_mask = attr.airfield_mask;
%> [runway_mask runways_idx runways_prop]= detect_runways(attr,airfield_mask);
%> [taxiroute_mask taxiroute_surplus_mask] = detect_taxiroutes(attr,airfield_mask,runway_mask);
%> [parkarea_mask parkarea_surplus_mask] = detect_parkareas(attr,airfield_mask,...
%>                                              runway_mask,taxiroute_mask,taxiroute_surplus_mask, sc_labels, mec_labels);
%====================================================================
function [parkarea_mask parkarea_surplus_mask] = detect_parkareas(attr,airfield_mask,...
                                                runway_mask,taxiroute_mask,taxiroute_surplus_mask, sc_labels, mec_labels)
%DETECT_PARKAREAS Summary of this function goes here
%   Detailed explanation goes here
%
% orhanf

%%
global DEBUG_FLAG;

    park_cand = (imsubtract(airfield_mask,(or(runway_mask,(or(taxiroute_mask,taxiroute_surplus_mask))))));
    park_cand = park_cand==1;
    park_cand = bwlabel(park_cand);
    park_cand_stats = regionprops(logical(park_cand),'Area');
    
    % do not consider regions lower that 500 px also this region can be
    % used to enhance taxiroutes as a future work
    idx = find([park_cand_stats.Area] > 500);
    park_cand = ismember(park_cand,idx);
    
    clusterMask = getClusteringResults( attr, sc_labels, mec_labels );
    park_seg_labels =immultiply(attr.labels,park_cand); 
    park_seg_labels_idx = unique(park_seg_labels);park_seg_labels_idx(1)=[];
    park_clusters = immultiply(clusterMask,park_cand);
    park_clusters_idx = unique(park_clusters); park_clusters_idx(1)=[];
    
    if DEBUG_FLAG
        figure,imshow(attr.im3b);
        figure,imshow(park_cand);
        figure,imshow(label2rgb(park_seg_labels,'jet','c','shuffle'));
        figure,imshow(label2rgb(park_clusters,'jet','c','shuffle'));
        figure,imshow(taxiroute_mask);
    end
    
%     imwrite(attr.im3b,['outputs\detection_results\' attr.name '\' attr.name '_img3b.png'],'png');
%     imwrite(label2rgb(park_seg_labels,'jet','c','shuffle'),['outputs\detection_results\' attr.name '\' attr.name '_park_seg_labels.png'],'png');
%     imwrite(label2rgb(park_clusters,'jet','c','shuffle'),['outputs\detection_results\' attr.name '\' attr.name '_park_clusters.png'],'png');
    
    % take adjacent original cluster components as parking areas
    d_taxiroute_mask = imdilate(taxiroute_mask,strel('disk',2));
    park_clusters_prime = conlabels(park_clusters);
    adj_cluster_cc_idx = unique(immultiply(d_taxiroute_mask,park_clusters_prime));
    adj_cluster_cc_idx(1)=[];
    parkarea_mask =ismember(park_clusters_prime,adj_cluster_cc_idx); 
%     imwrite(parkarea_mask,['outputs\detection_results\' attr.name '\' attr.name '_parkarea_mask.png'],'png');
    parkarea_surplus_mask = imsubtract(logical(park_cand),logical(parkarea_mask));
    if DEBUG_FLAG
        figure,imshow(parkarea_mask);
        figure,imshow(parkarea_surplus_mask);
    end
%     imwrite(parkarea_surplus_mask,['outputs\detection_results\' attr.name '\' attr.name '_parkarea_surplus_mask.png'],'png');
    
    % cluster surplus segmentation regions with entropy
%     features = extractRegionsFeatureAll(attr.im, attr.labels, 'entropy');
%     region_features = features(park_seg_labels_idx ,1:end-1);
%     mec_labels = mec(region_features, 'c', length(park_clusters_idx));
%     relabeled_parks_ent = zeros(size(attr.labels));
%     for i = 1:length(mec_labels)
%         relabeled_parks_ent(park_seg_labels==park_seg_labels_idx(i)) = mec_labels(i);
%     end
%     imwrite(label2rgb(relabeled_parks_ent),['outputs\detection_results\' attr.name '\' attr.name '_relabeled_parks_ent.png'],'png');
%     figure('name','entropy'),imshow(label2rgb(relabeled_parks_ent));
  
    % cluster surplus segmentation regions with meanstd
%     features = load(['inputs\features\' attr.segAlgo '\' attr.name '\' attr.name '_meanstd.mat']);
%     means = features.feature_i; 
%     mec_labels = mec(means(park_seg_labels_idx,:), 'c', length(park_clusters_idx));
%     relabeled_parks_meanstd = zeros(size(attr.labels));
%     for i = 1:length(mec_labels)
%         relabeled_parks_meanstd(park_seg_labels==park_seg_labels_idx(i)) = mec_labels(i);
%     end
%     imwrite(label2rgb(relabeled_parks_meanstd),['outputs\detection_results\' attr.name '\' attr.name '_relabeled_parks_meanstd.png'],'png');
%     figure('name','meanstd'),imshow(label2rgb(relabeled_parks_meanstd));
    
end

