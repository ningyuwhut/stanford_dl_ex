%
%This exercise uses a data from the UCI repository:
% Bache, K. & Lichman, M. (2013). UCI Machine Learning Repository
% http://archive.ics.uci.edu/ml
% Irvine, CA: University of California, School of Information and Computer Science.
%
%Data created by:
% Harrison, D. and Rubinfeld, D.L.
% ''Hedonic prices and the demand for clean air''
% J. Environ. Economics & Management, vol.5, 81-102, 1978.
%
addpath ../common
addpath ../common/minFunc_2012/minFunc
addpath ../common/minFunc_2012/minFunc/compiled

% Load housing data from file.
data = load('housing.data');
data=data'; % put examples in columns

% Include a row of 0s as an additional intercept feature.
%size(data,2) is the column number of data, which is also the sample number.
data = [ ones(1,size(data,2)); data ];

% Shuffle examples.
%randperm返回一个行向量，向量中的元素是对1:size(data,2)中的元素的随机排列
data = data(:, randperm(size(data,2)));

% Split into train and test sets
% The last row of 'data' is the median home price.
train.X = data(1:end-1,1:400);
train.y = data(end,1:400);

test.X = data(1:end-1,401:end);
test.y = data(end,401:end);

m=size(train.X,2); %样本数
n=size(train.X,1); %特征数

% Initialize the coefficient vector theta to random values.
theta = rand(n,1);

% Run the minFunc optimizer with linear_regression.m as the objective.
%
% TODO:  Implement the linear regression objective and gradient computations
% in linear_regression.m
%

%@linear_regression是一个函数
tic;
options = struct('MaxIter', 200,'useMex',0 ); %一个标量结构体，成员MaxIter的值为200
theta = minFunc(@linear_regression, theta, options, train.X, train.y);
fprintf('Optimization took %f seconds.\n', toc);

% Run minFunc with linear_regression_vec.m as the objective.
%
% TODO:  Implement linear regression in linear_regression_vec.m
% using MATLAB's vectorization features to speed up your code.
% Compare the running time for your linear_regression.m and
% linear_regression_vec.m implementations.
%
% Uncomment the lines below to run your vectorized code.
%Re-initialize parameters
%theta = rand(n,1);
%tic;
%theta = minFunc(@linear_regression_vec, theta, options, train.X, train.y);
%fprintf('Optimization took %f seconds.\n', toc);

% Plot predicted prices and actual prices from training set.
actual_prices = train.y;
predicted_prices = theta'*train.X;

% Print out root-mean-squared (RMS) training error.
%x .^ y表示元素乘方，这里表示平方
train_rms=sqrt(mean((predicted_prices - actual_prices).^2));
fprintf('RMS training error: %f\n', train_rms);

% Print out test RMS error
actual_prices = test.y;
predicted_prices = theta'*test.X;
test_rms=sqrt(mean((predicted_prices - actual_prices).^2));
fprintf('RMS testing error: %f\n', test_rms);


% Plot predictions on test data.
plot_prices=true;
if (plot_prices)
  [actual_prices,I] = sort(actual_prices); %sort函数对参数进行升序排列,返回的I表示排序后的元素在原数组上的下标
  predicted_prices=predicted_prices(I); %返回对应的预测价格
  plot(actual_prices, 'rx'); 
  hold on; %hold on表示后面的图也显示在这个plot上
  plot(predicted_prices,'bx');
  legend('Actual Price', 'Predicted Price');
  xlabel('House #');
  ylabel('House price ($1000s)');
end
