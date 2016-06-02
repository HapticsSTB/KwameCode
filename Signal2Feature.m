function features = Signal2Feature( rawDataDir, csvFileName, task, rounding )
    %Converts collected data from da Vinci to features
    %RawDataDir is the directory to the file
    
    %initialize extractedFeatures to be a 
    
    %load relevant files so I can get PCA features
    load 'ReconstructVecs.mat' %gives me "eigvecs" and "offset" (PCA)
    
    data = STBData(rawDataDir, csvFileName, 'task', task);
    data = data(~cellfun(@(x)any(isnan(x(:))), {data.score}));
    
    %assuming data is good here, extract features
    features = staticFeatures(data, rounding);
    
    %in case there's a batch of scores (normally there should only be one),
    %use a for loop
    %and then add the PCA features on
    for i = 1:length(features)
        pcScores = CalcPcScore(features(i), eigvecs, offset);
        features(i).pc1 = pcScores(:, 1);
        features(i).pc2 = pcScores(:, 2);
        features(i).pc3 = pcScores(:, 3);
    end
       
end

