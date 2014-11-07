function confEnergy = configuration_energy( obj, visibleStates, hiddenStates, W, b, c )
%CONFÝGURATÝON_ENERGY Summary of this function goes here
%   Detailed explanation goes here
%
% orhanf
%%

if nargin < 4   
    W = obj.rbm_W;
    b = obj.rbm_b;
    c = obj.rbm_c;
    if ~obj.addBias
       b = zeros(size(b)); 
       c = zeros(size(c)); 
    end    
end

confEnergy = mean(sum(W * visibleStates .* hiddenStates)) + ...   % Energy for 1-1 state visible-hidden units
             mean(sum(b' * hiddenStates)) + mean(sum(c' * visibleStates)); % Energy for *-1 state bias units 


end

