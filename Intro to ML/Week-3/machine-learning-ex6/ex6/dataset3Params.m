function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

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

x1 = [1 2 1]; x2 = [0 4 -1];

c_list = [0.01 0.03 0.1 0.3 1 3 10 30];
sigma_list = [0.01 0.03 0.1 0.3 1 3 10 30];

%result matrix
result = zeros(length(c_list) * length(sigma_list), 3);

r = 1;

for c_val = c_list
   for sigma_val = sigma_list
      %train using c_val and sigma_val
      model = svmTrain(X, y, c_val,@(x1, x2) gaussianKernel(x1, x2, sigma_val));
      predictions = svmPredict(model, Xval);
      
      %compute the error between your predictions and yval
      err_val = mean(double(predictions ~= yval));
      result(r,:) = [c_val sigma_val err_val];
      r = r + 1;
    endfor

endfor
 
[val, index] = min(result(:,3));
C = result(index, 1);
sigma = result(index, 2);
          




% =========================================================================

end
