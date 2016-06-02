function [feat, eigvecs, offset] = featurePCA(features)

    [feature_vector, ratings, index] = featureVector(features);
    [coeff,score,latent,tsquared,explained] = pca(feature_vector);

    pc1 = score(:,1);
    pc2 = score(:,2);
    pc3 = score(:,3);
    
    eigvecs = struct('eigvec1', coeff(:, 1), 'eigvec2', coeff(:, 2), 'eigvec3', coeff(:, 3));

    for i = 1:length(features)
        features(i).pc1 = pc1(i);
        features(i).pc2 = pc2(i);
        features(i).pc3 = pc3(i);
    end
    
    feat = features;
    offset = mean(feature_vector, 1);
end