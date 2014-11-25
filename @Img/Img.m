classdef Img
   
    properties(Hidden)        
        
        % class members
        data; 
        nCols;
        nRows;       
        nBands;
        hFigure;              
        imgname;
        filename;
       
        % mask and skeleton
        airfieldMask; 
        runwayMask;
        anaskels;
        skg;
        rad;
        skgStack;
        circStack;
        dispersalProbMap;
        labelMap;
        
        % performance results
        Perfs;
        
    end % properties   
        
       
    methods        
                
        % constructor
        function obj = Img(input)                
           if nargin>0 
               if isstruct(input)                                                                

                   obj.data         = input.attr.im;
                   obj.nCols        = size(input.attr.im,1);
                   obj.nRows        = size(input.attr.im,2);
                   obj.nBands       = size(input.attr.im,3);
                   obj.airfieldMask = input.attr.airfield_mask;
                   obj.runwayMask   = input.detections.runway_mask;               
                   obj.labelMap     = input.attr.filtered_labelmap;                    
                   
                   [~, name, ext] = fileparts(input.attr.name);

                   obj.imgname      = name;
                   obj.filename     = [name ext];               

               elseif (~isempty(input) && size(input,3)>1)
                        % is a rgb/rgbnir image
                   obj.data         = input;
                   obj.nCols        = size(input,1);
                   obj.nRows        = size(input,2);
                   obj.nBands       = size(input,3);                              
               end       

               obj.Perfs = Perf.empty(1,0);
           end
        end
                                       
        % display image
        function show_image(obj)        
           if ~obj.isEmpty() 
                obj.display_img();
           end               
        end        
        
        % overlay to selected band
        function overlay_detection(obj, detection, band)
            if ~obj.isEmpty()                                 
                obj.display_img(Utility.overlay_img(Utility.correct_img(obj.data), detection, band));
           end
        end
        
        % overdraw to selected band
        function overdraw_detection(obj, detection, band)
            if ~obj.isEmpty()                                 
                obj.display_img(Utility.overdraw_img(Utility.correct_img(obj.data), detection, band));
           end
        end
        
        % checks obj data field
        function res = isEmpty(obj)
           res = true;            
           if ~isempty(obj.data)
                res = false;
           end
        end
         
        % obtain gt map and return as a binary mask
        function targetMask = obtain_gt_mask(obj, targetName)
            foldername = obj.imgname(1:regexp(obj.imgname,'_MS_meydan_cevre_')-1);
            a = dir(['airport_detections\Clipped\' foldername '\clip\*.tif']);
            TiffName = ['airport_detections\Clipped\' foldername '\clip\' a(end).name];
            shapeFileName = ['gt\' foldername '\GT\' targetName '.shp'];
            targetMask = logical(shp2mask(TiffName,shapeFileName));
        end
        
        
        %
        % getters
        %
        function nCols = get.nCols(obj)            
            nCols = obj.nCols; 
        end % nCols get method
        
        function nRows = get.nRows(obj)            
            nRows = obj.nRows; 
        end % nRows get method
        
        function nBands = get.nBands(obj)            
            nBands = obj.nBands; 
        end % nBands get method
        
        function data = get.data(obj)            
            data = obj.data; 
        end % data get method
        
        function hFigure = get.hFigure(obj)            
            hFigure = obj.hFigure; 
        end % hFigure get method
        
        function airfieldMask = get.airfieldMask(obj)
            airfieldMask = obj.airfieldMask;
        end % airfieldMask get method
        
        function imgname = get.imgname(obj)
            imgname = obj.imgname;
        end % imgname get method

        function filename = get.filename(obj)
            filename = obj.filename;
        end % filename get method
        
        function anaskels = get.anaskels(obj)
            anaskels = obj.anaskels;
        end % anaskels get method
        
        function skg = get.skg(obj)
            skg = obj.skg;
        end % skg get method
        
        function rad = get.rad(obj)
            rad = obj.rad;
        end % rad get method                
        
        function skgStack = get.skgStack(obj)
            skgStack = obj.skgStack;
        end % skgStack get method                                
        
        function circStack = get.circStack(obj)
            circStack = obj.circStack;
        end % circStack get method                                
                 
        function runwayMask = get.runwayMask(obj)
            runwayMask = obj.runwayMask;
        end % runwayMask get method                                
        
        function dispersalProbMap = get.dispersalProbMap(obj)
            dispersalProbMap = obj.dispersalProbMap;
        end % dispersalProbMap get method                                
        
        function Perfs = get.Perfs(obj)
            Perfs = obj.Perfs;
        end % Perfs get method                                                
        
        function labelMap = get.labelMap(obj)
            labelMap = obj.labelMap;
        end % labelMap get method                                                
                
        function img = get_img_3b(obj)
            img = [];
            if ~obj.isEmpty()
                img = Utility.correct_img(obj.data);
            end
        end
        
                    
        
        
        %
        % setters
        %
        function obj = set.nCols(obj,nCols)            
            obj.nCols = nCols; 
        end % nCols set method
        
        function obj = set.nRows(obj,nRows)            
            obj.nRows = nRows; 
        end % nRows set method
        
        function obj = set.nBands(obj,nBands)            
            obj.nBands = nBands; 
        end % nBands set method
        
        function obj = set.data(obj,data)            
            obj.data = data; 
        end % data set method
        
        function obj = set.hFigure(obj,hFigure)            
             obj.hFigure = hFigure; 
        end % hFigure set method
        
        function obj = set.airfieldMask(obj,airfieldMask)
            obj.airfieldMask = airfieldMask;
        end % airfieldMask set method
        
        function obj = set.imgname(obj,imgname)
            obj.imgname = imgname;
        end % imgname set method

        function obj = set.filename(obj,filename)
            obj.filename = filename;
        end % filename set method
        
        function obj = set.anaskels(obj,anaskels)
            obj.anaskels = anaskels;
        end % anaskels set method
        
        function obj = set.skg(obj,skg)
            obj.skg = skg;
        end % skg set method
        
        function obj = set.rad(obj,rad)
            obj.rad = rad;
        end % rad set method        

        function obj = set.skgStack(obj,skgStack)
            obj.skgStack = skgStack;
        end % skgStack set method        
        
        function obj = set.circStack(obj,circStack)
            obj.circStack = circStack;
        end % circStack set method        
                
        function obj = set.runwayMask(obj,runwayMask)
            obj.runwayMask = runwayMask;
        end % runwayMask set method        

        function obj = set.dispersalProbMap(obj,dispersalProbMap)
            obj.dispersalProbMap = dispersalProbMap;
        end % dispersalProbMap set method        
        
        function obj = set.Perfs(obj,Perfs)
            obj.Perfs = Perfs;
        end % Perfs set method        
        
        function obj = set.labelMap(obj,labelMap)
            obj.labelMap = labelMap;
        end % labelMap set method        
        
        
    end % methods    
    
end
