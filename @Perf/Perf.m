classdef Perf
%==========================================================================
%
% Performance calculation class, precision, recall and fscore is provided
% along with some static member functions. Instantiated in @Img class. 
%
% orhanf - (c) 2012 Orhan FIRAT <orhan.firat@ceng.metu.edu.tr>
%==========================================================================
   
    properties        
        params;     % parameters that led the results for algorithm        
        precision;
        recall;
        fscore;
        tp;
        fp;
        fn;
    end
    
    methods
        
        
        %==================================================================
        % constructor
        %==================================================================
        function obj = Perf(gtMask, detMask, params)
            if nargin >= 2 && Perf.isValid(gtMask,detMask)
                [   obj.precision,...
                    obj.recall, ...
                    obj.fscore, ... 
                    obj.tp, ... 
                    obj.fp, ... 
                    obj.fn ] = Perf.calculate_performance(gtMask,detMask);
               
                if nargin == 3
                    obj.params = params;
                end                                
            end    
            
            if nargin == 1 % for test only
                obj.params = gtMask;
            end
        end
        
    end
    
    
    methods(Static)
        
    
        %==================================================================
        % calculates performance measures according to detection result as
        % area and target as small objects. If area is smaller than a
        % threshold and includes any target object then all objects within
        % area are counted as tp. If area does not encapsulate any objects
        % then whole area is counted as a fp. All missing objects are
        % counted as a distinct fn. USE IT FOR DISPERSALS
        %==================================================================
        function [precision, recall, fscore, tp, fp, fn] = calculate_performance_obj_within_area(gtMask, detMask, areaRatioTh)
        
            if nargin<3
                areaRatioTh = 10;
            end                
            
            gtLabels  = bwlabel(gtMask,4);
            detLabels = bwlabel(detMask,4);
            
            nTarget  = max(max(gtLabels));
            nPredict = max(max(detLabels));
            
            gtInds = find(gtMask);
                       
            tp = 0;
            fp = 0;                         
               
            % process each detection region for area threshold
            for i=1:nPredict
                
                currDetMask  = detLabels==i;
                currInds = find(currDetMask);
                currArea = length(currInds(:));
                
                intersectInds = intersect(currInds,gtInds);                
                
                if isempty(intersectInds)   % false positive
                    fp = fp + 1;                    
                else                        % true positive                                                            
                    
                    tpCount = length(setdiff(unique(gtLabels(intersectInds)),0));
                    tp = tp + tpCount;
            
%                     [X,Y] = find(currDetMask .* gtMask);
%                     
%                     [nouse, convHullArea] = convhull(X,Y);
%                     
%                     if currArea > (areaRatioTh * convHullArea) % false positive
%                         fp = fp + 1;    
%                     end
                end                
            end
            
            fn = nTarget - tp;
            
            precision = Perf.calculate_precision(tp, fp);
            recall    = Perf.calculate_recall(tp, fn);
            fscore    = Perf.calculate_fscore(precision, recall);
            
        end
        
        
        %==================================================================
        % calculates performance measures according to detection result as
        % area and target as area too. If the gt area component overlaps
        % with any of the detection areas then it is counted as tp. Rest of
        % the detected areas are counted as a distinct fp. Undetected gt
        % components are counted as fn. USE IT FOR PARKAREAS
        %==================================================================
        function [precision, recall, fscore, tp, fp, fn] = calculate_performance_area_within_area(gtMask, detMask)

            gtLabels  = bwlabel(gtMask,4);
            detLabels = bwlabel(detMask,4);
            
            nTarget  = max(max(gtLabels));
            nPredict = max(max(detLabels));
            
            overlappingGTMap = gtLabels .* detMask;
            overlappingDTMap = detLabels .* gtMask;
            
            fn = numel(setdiff(1:nTarget,unique(overlappingGTMap(:))));
            tp = nTarget - fn;
            fp = numel(setdiff(1:nPredict,unique(overlappingDTMap(:))));
                        
            precision = Perf.calculate_precision(tp, fp);
            recall    = Perf.calculate_recall(tp, fn);
            fscore    = Perf.calculate_fscore(precision, recall);
            
        end
        
        
        %==================================================================
        % calculates performance measures according to detection result as
        % area and target as area too. If the gt area component overlaps
        % with any of the detection areas then it is counted as tp. Rest of
        % the detected areas are counted as a distinct fp. Undetected gt
        % components are counted as fn. notFPmask is a secondary mask that
        % is to be used for components that should not be counted as FP,
        % such components do not count for TP, just ignore them.
        % USE IT FOR PARKAREAS
        %==================================================================
        function [precision, recall, fscore, tp, fp, fn] = calculate_performance_area_within_area_usingNotFPmask(gtMask, notFPmask, detMask)

            gtLabels  = bwlabel(gtMask,4);
            detLabels = bwlabel(detMask,4);
            
            nTarget  = max(max(gtLabels));
            nPredict = max(max(detLabels));
            
            overlappingGTMap = gtLabels .* detMask;
            overlappingDTMap = detLabels .* gtMask;
            overlappingNotFPMap = notFPmask .* detMask;
            
            fn = numel(setdiff(1:nTarget,unique(overlappingGTMap(:))));
            tp = nTarget - fn;
            fp = numel(setdiff(1:nPredict,unique([unique(overlappingDTMap(:)); unique(overlappingNotFPMap(:))])));
                        
            precision = Perf.calculate_precision(tp, fp);
            recall    = Perf.calculate_recall(tp, fn);
            fscore    = Perf.calculate_fscore(precision, recall);
            
        end
        
        
        %==================================================================
        % calculates performance measures
        %==================================================================
        function [precision, recall, fscore, tp, fp, fn] = calculate_performance(gtMask, detMask)
            
            gtLabels  = bwlabel(gtMask,4);
            detLabels = bwlabel(detMask,4);
            
            nTarget  = max(max(gtLabels));
            nPredict = max(max(detLabels));
                        
            tp = length(setdiff(unique(immultiply(gtLabels,detMask)),0));
            fp = length(setdiff(1:nPredict,unique(immultiply(detLabels,gtMask)))); 
            fn = nTarget - tp;
            
            precision = Perf.calculate_precision(tp, fp);
            recall    = Perf.calculate_recall(tp, fn);
            fscore    = Perf.calculate_fscore(precision, recall);
                        
        end
        
        
        %==================================================================
        % calculates precision
        %==================================================================
        function precision = calculate_precision(tp, fp)            
            
            precision = tp ./ (tp + fp);    
            
            % eliminate NaN s
            precision(isnan(precision)) = 0;
        end
        
        
        %==================================================================
        % calculates recall
        %==================================================================
        function recall = calculate_recall(tp, fn)
            
            recall = tp ./ (tp + fn);
            
            % eliminate NaN s
            recall(isnan(recall)) = 0;
        end
        
        
        %==================================================================
        % calculates fscore
        %==================================================================
        function fscore = calculate_fscore(precision, recall)
            
            fscore = 2.*(precision.*recall)./(precision+recall);
            
            % eliminate NaN s
            fscore(isnan(fscore)) = 0;
        end
        
        
        %==================================================================
        % checks the boundary conditions for mask and gt
        %==================================================================
        function result = isValid(gtMask,detMask)
            result = false;            
            if size(gtMask,1) == size(detMask,1) &&...
                    size(gtMask,2) == size(detMask,2)
               result = true; 
            end
        end
                        
    end

end

