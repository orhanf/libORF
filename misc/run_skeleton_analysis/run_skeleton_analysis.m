%==================================================================== >
%> @brief This function initiates skeleton analysis on the airfield mask in
%> order to obtain final detections.
%>
%> @author orhanf&okant
%>
%> Change                                      Date       Performed By 
%>
%> @param attr Struct that includes all image attributes
%> @param sc_labels Labelmap obtained by Spectral Clustering 
%> @param mec_labels Labelmap obtained by MEC
%>
%> @retval detections Struct that includes all the detection results
%> 
%> @example 
%> filename = 'test.png';
%> im = imread(filename);
%> roimask = imread(roi_filename);
%> gtiffinfo = geotiffinfo(filename);
%> imres = gtiffinfo.PixelScale(1);
%> meanshift_labels = segment(im, filename, imres);
%> roi_segments = extract_roi_segments(meanshift_labels, roimask);
%> [feature_i]  = extract_initial_features_meanshift(im, meanshift_labels, {'meanstd'});
%> [vegetation_map filteredSegs] = extract_vegetation_map(im, meanshift_labels);
%> [mec_label_map newLabels labels_sc labels_mec] = run_spectral_clustering(feature_i, filteredSegs, roi_segments, meanshift_labels);
%> [initial_mask additional_mask] = extract_overlapping_cluster_with_segs(labels_sc, labels_mec);
%> [detections] = run_skeleton_analysis(attr, sc_labels, mec_labels);
%====================================================================
function [detections] = run_skeleton_analysis(attr, sc_labels, mec_labels)

% addPath;
% images = dir('inputs\images\*.tif');
% 
% % segmentation type
% segAlgo = {'meanshift'};
% 
% % param for segmentation
% segParam = {'_2_2_50'}; 
% 
% % param for vegetation classification
% vegAlgo = 'building_detection';
% 
% mkdir('outputs\detection_results');
% 
% %process each image 
% for i=1:numel(images)
%         
%     [~,filename,~] = fileparts(images(i).name);    
%     mkdir(['outputs\detection_results\' filename]);
% 
%     % get all image attributes
%     attr = getImageAttributes(filename,segAlgo{1},segParam{1},vegAlgo);
%     airfield_mask = logical(imread(['outputs\final_masks\' filename '_final_mask.png']));
    airfield_mask = attr.airfield_mask;
%     runway detection
    [runway_mask runways_idx runways_prop]= detect_runways(attr,airfield_mask);
    
%     taxiroute detection
    [taxiroute_mask taxiroute_surplus_mask] = detect_taxiroutes(airfield_mask,runway_mask);
     
%     park area detection
    [parkarea_mask parkarea_surplus_mask] = detect_parkareas(attr,airfield_mask,...
                                                runway_mask,taxiroute_mask,taxiroute_surplus_mask, sc_labels, mec_labels);

    
    [new_parkarea_mask new_taxiroute_mask] = rectify_detections(airfield_mask,runway_mask,...
                                                taxiroute_mask,taxiroute_surplus_mask, parkarea_mask);

    % dispersal detection
    dispersal_pts  = detect_dispersals(airfield_mask,runway_mask,new_taxiroute_mask);
%     save(['outputs\detection_results\' filename '\' filename '_dispersal_pts.mat'],'dispersal_pts');
    dispersal_mask = zeros(size(airfield_mask));
    if ~isempty(dispersal_pts)
        dispersal_mask(sub2ind(size(dispersal_mask),dispersal_pts(:,2),dispersal_pts(:,1)))=1;
    end
    dispersal_mask = imdilate(dispersal_mask, strel('disk',5));
%     imwrite(dispersal_mask,['outputs\detection_results\' filename '\' filename '_dispersal_mask.png']);
% 
%     close all;
    detections.airfield_mask = airfield_mask;
    detections.runway_mask = runway_mask;
    detections.runways_idx = runways_idx;
    detections.taxiroute_mask = taxiroute_mask;
    detections.taxiroute_surplus_mask = taxiroute_surplus_mask;
    detections.parkarea_mask = parkarea_mask;
    detections.parkarea_surplus_mask = parkarea_surplus_mask;
    detections.new_parkarea_mask = new_parkarea_mask;
    detections.new_taxiroute_mask = new_taxiroute_mask;
    detections.dispersal_mask = dispersal_mask;
    detections.runways_prop = runways_prop;
end
