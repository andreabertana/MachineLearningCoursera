function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

% Define h_theta 
h_theta = sigmoid(X * theta);
% Define regularization 
reg = (lambda /(2*m)) * sum(theta(2:end) .* theta(2:end)) ;

% Compute cost
J = (sum((-y .* log(h_theta)) - ((1 - y) .* log( 1 - h_theta))) ./ m) + reg;

% Compute partial derivatives over theta
reg2 = (lambda /(m)) * theta(2:end) ;

grad(1) = (1 / m) * sum( X(:,1) .* (h_theta - y) );
grad(2:end) = (1 / m) .* sum( X(:,2:end) .* repmat((h_theta - y), 1, size(X,2) - 1) )' + reg2;


% =============================================================

end
