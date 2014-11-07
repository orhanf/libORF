function [sampledVisibleStates,sampledVisibleProbs,confEnergy] = sample_visible_data( obj, nSamples )
%SAMPLE_VÝSÝBLE_DATA Summary of this function goes here
%   Detailed explanation goes here
%
% orhanf
%%


visibleSize = obj.visibleSize;
hiddenSize  = obj.hiddenSize;
sample0 = rand(visibleSize,1);
sampledVisibleStates = [ zeros(visibleSize, nSamples)];
sampledVisibleProbs  = [sample0,zeros(visibleSize, nSamples)];
sampledHiddenStates  = zeros(hiddenSize, nSamples);

for i=1:nSamples
    visibleState   = sampledVisibleProbs(:,i);    
    sampledHiddenStates(:,i)   = obj.vis_to_hid(visibleState);  % up
    [sampledVisibleStates(:,i),...
     sampledVisibleProbs(:,i+1)] = obj.hid_to_vis(sampledHiddenStates(:,i)); % down         
end

sampledVisibleProbs(:,1) = []; % remove initial sample

confEnergy = obj.configuration_energy(sampledVisibleStates, sampledHiddenStates);

end

