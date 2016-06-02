function pcScores = CalcPcScore( features, eigvecs, offset )
    %calculates the first 3 pca scores given features and eigenvectors
    
    %initialize pcScores
    pcScores = zeros(length(features), 3);
    
    %put features in vector form
    [feature_vector, ~, ~] = featureVector(features);
    
    %shift by offset to calculate score
    feature_vector = bsxfun(@minus, feature_vector, offset);
    
    %calculate pcScores
    pcScores(:, 1) = feature_vector*eigvecs.eigvec1;
    pcScores(:, 2) = feature_vector*eigvecs.eigvec2;
    pcScores(:, 3) = feature_vector*eigvecs.eigvec3;    
end

