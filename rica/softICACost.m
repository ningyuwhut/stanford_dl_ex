%% Your job is to implement the RICA cost and gradient
function [cost,grad] = softICACost(theta, x, params)

% unpack weight matrix
W = reshape(theta, params.numFeatures, params.n);
%n是输入数据的维度
%numFeatures可以看做是基向量

% project weights to norm ball (prevents degenerate bases)
Wold = W;
W = l2rowscaled(W, 1);
%l2rowscaled对每个基向量进行归一化

z=W*x;
%%% YOUR CODE HERE %%%
l1_matrix=sqrt(z.^2+params.epsilon);
l1=params.lambda*sum( l1_matrix(:) );
%为什么这么算
error=W'*z-x;
square_error=error.^2;
penalty=0.5* sum(square_error(:)); %矩阵的frobenius norm
%计算方式是矩阵中的所有元素的平方和再开根号
cost=l1+penalty;
Wgrad = params.lambda * ((z./ l1_matrix) * x');
Wgrad+=W*error*x'+W*x*error';
% unproject gradient for minFunc
grad = l2rowscaledg(Wold, W, Wgrad, 1);
grad = grad(:);
