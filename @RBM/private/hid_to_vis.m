function [visibleStates,visibleProbs,confEnergy] = hid_to_vis( obj, hiddenStates )
%HÝD_TO_VÝS Summary of this function goes here
%   Detailed explanation goes here
%
% orhanf
%%

nSamples  = size(hiddenStates,2);
addBias   = obj.addBias;

W = obj.rbm_W;
b = obj.rbm_b;
c = obj.rbm_c;
if ~addBias
   b = zeros(size(b)); 
   c = zeros(size(c)); 
end

visibleActs   = W' * hiddenStates + repmat(c, [1, nSamples]); % Calculate the logit for hidden units
visibleProbs  = sigmoid(visibleActs);               % Calculate probabilities of turning the hidden units on
visibleStates = double(visibleProbs>obj.sample_bernoulli(visibleProbs)); % Sample bernoulli for converting into binary units

if nargout > 2
    confEnergy = obj.configuration_energy(visibleStates,hiddenStates, W, b, c);
end
         
end

