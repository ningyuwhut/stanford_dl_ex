function [f,g] = logistic_regression(theta, X,y)
  %
  % Arguments:
  %   theta - A column vector containing the parameter values to optimize.
  %   X - The examples stored in a matrix.  
  %       X(i,j) is the i'th coordinate of the j'th example.
  %   y - The label for each example.  y(j) is the j'th example's label.
  %

  m=size(X,2);
  
  % initialize objective value and gradient.
  f = 0;
  g = zeros(size(theta));


  %
  % TODO:  Compute the objective function by looping over the dataset and summing
  %        up the objective values for each example.  Store the result in 'f'.
  %
  % TODO:  Compute the gradient of the objective by looping over the dataset and summing
  %        up the gradients (df/dtheta) for each example. Store the result in 'g'.
  %
%%% YOUR CODE HERE %%%
 
  n=size(X,1);%特征数
  for i=1:m
    theta_X=0;
    for j=1:n
	theta_X+=theta(j)*X(j,i);
    endfor
    hi=1/(1+exp(-theta_X));

    f-=y(i)*log(hi)+( 1-y(i) )*log(1-hi);

    for j=1:n
	g(j)+=(hi-y(i))*X(j,i);
    endfor
  endfor
endfunction

     
