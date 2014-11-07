function [hiddenStates,hiddenProbs,confEnergy] = vis_to_hid( obj, data )
%
%
% orhanf
%%

nSamples  = size(data,2);
addBias   = obj.addBias;

W = obj.rbm_W;
b = obj.rbm_b;
c = obj.rbm_c;
if ~addBias
   b = zeros(size(b)); 
   c = zeros(size(c)); 
end

hiddenActs   = W * data + repmat(b, [1, nSamples]); % Calculate the logit for hidden units
hiddenProbs  = sigmoid(hiddenActs);                 % Calculate probabilities of turning the hidden units on
hiddenStates = double(hiddenProbs>obj.sample_bernoulli(hiddenProbs)); % Sample bernoulli for converting into binary units

if nargout > 2
    confEnergy = obj.configuration_energy(data,hiddenStates, W, b, c);
end

end

