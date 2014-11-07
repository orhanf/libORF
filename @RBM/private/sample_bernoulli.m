function samples = sample_bernoulli(obj,x,seed)
%
%
% orhanf
%%

    % the first case is used when a randomness source is given, which
    % meaning that each run will have deterministic outputs 
    if ~isempty(obj.randSource)
        if nargin<3
            seed = sum(x(:));
        end
        startIdx = mod(round(seed), round(size(obj.randSource, 2) / 10)) + 1;
        samples = reshape(obj.randSource(startIdx : startIdx+numel(x)-1), size(x));
    else % pseudo-random case
        samples = rand(size(x));
    end    
    
end