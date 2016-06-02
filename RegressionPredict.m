function ratings_pred = RegressionPredict( features, rounding )
    %REGRESSIONPREDICT uses the regression models trained to predict the
    %ratings of the surgeon given extracted feature data
    
    load 'SelectFeaturesMean.mat' % this file contains selectFeatures
    load 'RegEnsemble.mat' % this file contains models, muX_final, sigmaX_final
    [feature_vector, ~, ~] = featureVector(features);
    nMetric = 5;
    featLen = length(features);
    
    pred1 = zeros(featLen, nMetric);
    pred2 = zeros(featLen, nMetric);
    pred3 = zeros(featLen, nMetric);
    pred4 = zeros(featLen, nMetric);
    
    for i = 1:nMetric
        selectfeature_vector = feature_vector(:, selectFeatures{i});
        %standardize data here
        X = bsxfun(@rdivide,bsxfun(@minus, selectfeature_vector, muX_final{i}), sigmaX_final{i}); 
        [pred1(:, i), ~, ~] = svmpredict(zeros(featLen, 1) , X, models1{i}, '-q');
        pred2(:, i) = cvglmnetPredict(models2{i}, X, 'lambda_1se');
        pred3(:, i) = predict(models3{i}, X);
        pred4(:, i) = predict(models4{i}, X);
    end
    ratings_pred = (pred1 + pred2 + pred3 + pred4)/4;
    
    if rounding == 1
        ratings_pred = round(ratings_pred);
    end
    
end

