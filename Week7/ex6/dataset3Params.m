function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = [.01 .03 .1 .3 1 3 10 30];
sigma = C;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
this_model = 1;
for this_C = C
    for this_sigma = sigma
        
        model= svmTrain(X, y, this_C, @(x1, x2) gaussianKernel(x1, x2, this_sigma));
        predictions = svmPredict(model, Xval);
        prediction_error = mean(double(predictions ~= yval));
        
        resultsSum(this_model, :) = [this_C, this_sigma, prediction_error];
        this_model = this_model +1;
    end
end

%Best resutls
[~, idx] = min(resultsSum(:,3));
C = resultsSum(idx ,1);
sigma = resultsSum(idx ,2);
% =========================================================================

end
