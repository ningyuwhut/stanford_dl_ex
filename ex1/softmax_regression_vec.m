function [f,g] = softmax_regression(theta, X,y)
  %
  % Arguments:
  %   theta - A vector containing the parameter values to optimize.
  %       In minFunc, theta is reshaped to a long vector.  So we need to
  %       resize it to an n-by-(num_classes-1) matrix.
  %       Recall that we assume theta(:,num_classes) = 0.
  %
  %   X - The examples stored in a matrix.  
  %       X(i,j) is the i'th coordinate of the j'th example.
  %   y - The label for each example.  y(j) is the j'th example's label.
  %
  m=size(X,2);
  n=size(X,1);

  % theta is a vector;  need to reshape to n x num_classes.
  % Specify [] for the second dimension size to let reshape automatically calculate the appropriat    e number of columns.
  theta=reshape(theta, n, []); %一行一个特征，一列是一个类
  num_classes=size(theta,2)+1;
  
  % initialize objective value and gradient.
  f = 0;
  g = zeros(size(theta));

  %
  % TODO:  Compute the softmax objective function and gradient using vectorized code.
  %        Store the objective function value in 'f', and the gradient in 'g'.
  %        Before returning g, make sure you form it back into a vector with g=g(:);
  %
%%% YOUR CODE HERE %%%
  %y是一个行向量

    y_hat=[exp(theta'*X);ones(1,size(theta'*X,2))]; %最后一行加上最后一个类的概率的分子，即1
    p=y_hat ./sum(y_hat,1); %按列求和,每列表示一个样本属于每个类的概率,结果为分子的每一列除以分母的对应的列,得到每个样本属于每个类的概率，每一列表示一个一个样本
    size(p')
    size(p',1)
    size(y)
    I=sub2ind(size(p'),1:size(p',1),y); %I表示
    value=p'(I);
    f=-sum(value);
    indicator=zeros(size(p'));
    indicator(I)=1; %一行表示一个样本
    g=-X*(indicator-p');
    g=g(:,1:end-1); %去掉最后一个类的值，因为最后一个类的值都为0
  %运行大约一分钟
  g=g(:); % make gradient a vector for minFunc

