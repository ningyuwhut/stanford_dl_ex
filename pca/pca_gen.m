%%================================================================
%% Step 0a: Load data
%  Here we provide the code to load natural image data into x.
%  x will be a 784 * 600000 matrix, where the kth column x(:, k) corresponds to
%  the raw image data from the kth 12x12 image patch sampled.
%  You do not need to change the code below.

addpath(genpath('../common'))
x = loadMNISTImages('../common/train-images-idx3-ubyte');

figure('name','Raw images');
randsel = randi(size(x,2),200,1); % A random selection of samples for visualization
x(randsel);
%返回一个200行一列的向量
display_network(x(:,randsel));
%这一行在运行时出错

%%================================================================
%% Step 0b: Zero-mean the data (by row)
%  You can make use of the mean and repmat/bsxfun functions.

%%% YOUR CODE HERE %%%
mean_x=mean(x); %行向量
x=x-repmat(mean_x, size(x,1), 1);

%%================================================================
%% Step 1a: Implement PCA to obtain xRot
%  Implement PCA to obtain xRot, the matrix in which the data is expressed
%  with respect to the eigenbasis of sigma, which is the matrix U.

%%% YOUR CODE HERE %%%
sigma=x*x'/size(x,2); %协方差矩阵
[U,S,V]=svd(sigma);
%U的每一列是一个特征向量，从左到右排列
%S对角线上是特征值，从大到小排列
xRot=U'*x;
%sigma为n维实对称方阵，n是特征个数，那么它的特征值个数也为n，特征向量也是n维,且每个特征向量线性无关

%%================================================================
%% Step 1b: Check your implementation of PCA
%  The covariance matrix for the data expressed with respect to the basis U
%  should be a diagonal matrix with non-zero entries only along the main
%  diagonal. We will verify this here.
%  Write code to compute the covariance matrix, covar. 
%  When visualised as an image, you should see a straight line across the
%  diagonal (non-zero entries) against a blue background (zero entries).

%%% YOUR CODE HERE %%%
%求协方差矩阵就应该减去均值啊,为什么减去均值之后协方差矩阵反而出现非对角线元素不为0呢
mean_xRot=mean(xRot);

%xRot_mean_zero=xRot-repmat(mean_xRot, size(xRot,1),1);
%covar=xRot_mean_zero*xRot_mean_zero'/size(xRot,2);

covar=xRot*xRot'/size(xRot,2);
% Visualise the covariance matrix. You should see a line across the
% diagonal against a blue background.
figure('name','Visualisation of covariance matrix of xRot');
imagesc(covar);

%%================================================================
%% Step 2: Find k, the number of components to retain
%  Write code to determine k, the number of components to retain in order
%  to retain at least 99% of the variance.

%%% YOUR CODE HERE %%%
eigen_value=diag(S); %列向量
%variation_percentage=eigen_value/sum(eigen_value);
eigen_value_sum=sum(eigen_value);
partial_sum=0.0;
for k=1:size(eigen_value)
%    k
%    eigen_value(k)
    partial_sum+=eigen_value(k);
%    partial_sum/eigen_value_sum
    if partial_sum/eigen_value_sum > 0.99
	break;
    endif
endfor

%%================================================================
%% Step 3: Implement PCA with dimension reduction
%  Now that you have found k, you can reduce the dimension of the data by
%  discarding the remaining dimensions. In this way, you can represent the
%  data in k dimensions instead of the original 144, which will save you
%  computational time when running learning algorithms on the reduced
%  representation.
% 
%  Following the dimension reduction, invert the PCA transformation to produce 
%  the matrix xHat, the dimension-reduced data with respect to the original basis.
%  Visualise the data and compare it to the raw data. You will observe that
%  there is little loss due to throwing away the principal components that
%  correspond to dimensions with low variation.

%%% YOUR CODE HERE %%%
xRot(k+1:size(xRot),:)=0;
xHat=U*xRot;
%这里的xHat是把从压缩后的数据再还原回之前的数据
%xHat=xRot(1:k,:);
% Visualise the data, and compare it to the raw data
% You should observe that the raw and processed data are of comparable quality.
% For comparison, you may wish to generate a PCA reduced image which
% retains only 90% of the variance.

figure('name',['PCA processed images ',sprintf('(%d / %d dimensions)', k, size(x, 1)),'']);
display_network(xHat(:,randsel));
figure('name','Raw images in xHat');
display_network(x(:,randsel));

%%================================================================
%% Step 4a: Implement PCA with whitening and regularisation
%  Implement PCA with whitening and regularisation to produce the matrix
%  xPCAWhite. 

epsilon = 1e-1; 
%%% YOUR CODE HERE %%%
xPCAWhite=diag(1./sqrt(diag(S)+epsilon))*U'*x;
%diag的第一个参数为向量，第二个参数为空时，表示产生以第一个向量为对角元素的对角矩阵
%% Step 4b: Check your implementation of PCA whitening 
%  Check your implementation of PCA whitening with and without regularisation. 
%  PCA whitening without regularisation results a covariance matrix 
%  that is equal to the identity matrix. PCA whitening with regularisation
%  results in a covariance matrix with diagonal entries starting close to 
%  1 and gradually becoming smaller. We will verify these properties here.
%  Write code to compute the covariance matrix, covar. 
%
%  Without regularisation (set epsilon to 0 or close to 0), 
%  when visualised as an image, you should see a red line across the
%  diagonal (one entries) against a blue background (zero entries).
%  With regularisation, you should see a red line that slowly turns
%  blue across the diagonal, corresponding to the one entries slowly
%  becoming smaller.

%%% YOUR CODE HERE %%%
epsilon=0;
%xPCAWhite_no_regularization=diag(1./sqrt(diag(S)))*U'*x
%mean_xPCAWhite_no_regularization=mean(xPCAWhite_no_regularization)
%xPCAWhite_mean_zero=xPCAWhite_no_regularization-repmat(mean_xPCAWhite_no_regularization, size(xPCAWhite_no_regularization,1),1)
covar=xPCAWhite*xPCAWhite'/size(xPCAWhite,2);

%mean_xPCAWhite=mean(xPCAWhite)
%xPCAWhite_mean_zero=xPCAWhite-repmat(mean_xPCAWhite, size(xPCAWhite,1),1)
%covar=xPCAWhite_mean_zero*xPCAWhite_mean_zero'/size(xPCAWhite_mean_zero,2)

% Visualise the covariance matrix. You should see a red line across the
% diagonal against a blue background.
figure('name','Visualisation of covariance matrix of PCAWhite');
imagesc(covar);

%%================================================================
%% Step 5: Implement ZCA whitening
%  Now implement ZCA whitening to produce the matrix xZCAWhite. 
%  Visualise the data and compare it to the raw data. You should observe
%  that whitening results in, among other things, enhanced edges.

%%% YOUR CODE HERE %%%
xZCAWhite=U*diag(1./sqrt(diag(S)+epsilon))*U'*x;
% Visualise the data, and compare it to the raw data.
% You should observe that the whitened images have enhanced edges.
figure('name','ZCA whitened images');
display_network(xZCAWhite(:,randsel));
figure('name','Raw images in last step');
display_network(x(:,randsel));
