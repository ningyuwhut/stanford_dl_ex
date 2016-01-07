function [cost, grad, preds] = cnnCost(theta,images,labels,numClasses,...
                                filterDim,numFilters,poolDim,pred)
% Calcualte cost and gradient for a single layer convolutional
% neural network followed by a softmax layer with cross entropy
% objective.
%                            
% Parameters:
%  theta      -  unrolled parameter vector
%  images     -  stores images in imageDim x imageDim x numImges
%                array
%  numClasses -  number of classes to predict
%  filterDim  -  dimension of convolutional filter                            
%  numFilters -  number of convolutional filters
%  poolDim    -  dimension of pooling area
%  pred       -  boolean only forward propagate and return
%                predictions
%如果pred为true，表示只进行前向传播，而不进行后向传播
%
%
% Returns:
%  cost       -  cross entropy cost
%  grad       -  gradient with respect to theta (if pred==False)
%  preds      -  list of predictions for each example (if pred==True)


if ~exist('pred','var')
    pred = false;
end;


imageDim = size(images,1); % height/width of image
numImages = size(images,3); % number of images

%% Reshape parameters and setup gradient matrices

% Wc is filterDim x filterDim x numFilters parameter matrix
% bc is the corresponding bias,长度为filter

% Wd is numClasses x hiddenSize parameter matrix where hiddenSize
% is the number of output units from the convolutional layer
% bd is corresponding bias
[Wc, Wd, bc, bd] = cnnParamsToStack(theta,imageDim,filterDim,numFilters,...
                        poolDim,numClasses);
%把向量theta每部分还原回之前的参数

% Same sizes as Wc,Wd,bc,bd. Used to hold gradient w.r.t above params.
Wc_grad = zeros(size(Wc));
Wd_grad = zeros(size(Wd));
bc_grad = zeros(size(bc));
bd_grad = zeros(size(bd));

%%======================================================================
%% STEP 1a: Forward Propagation
%  In this step you will forward propagate the input through the
%  convolutional and subsampling (mean pooling) layers.  You will then use
%  the responses from the convolution and pooling layer as the input to a
%  standard softmax layer.

%% Convolutional Layer
%  For each image and each filter, convolve the image with the filter, add
%  the bias and apply the sigmoid nonlinearity.  Then subsample the 
%  convolved activations with mean pooling.  Store the results of the
%  convolution in activations and the results of the pooling in
%  activationsPooled.  You will need to save the convolved activations for
%  backpropagation.
convDim = imageDim-filterDim+1; % dimension of convolved output
outputDim = (convDim)/poolDim; % dimension of subsampled output

% convDim x convDim x numFilters x numImages tensor for storing activations
activations = zeros(convDim,convDim,numFilters,numImages);

%size(activations)
% outputDim x outputDim x numFilters x numImages tensor for storing
% subsampled activations
activationsPooled = zeros(outputDim,outputDim,numFilters,numImages);

%%% YOUR CODE HERE %%%
activations=cnnConvolve(filterDim, numFilters, images, Wc,bc); %计算卷积
activationsPooled=cnnPool(poolDim, activations); %池化

%hiddenSize是卷积层的输出单元个数
% Reshape activations into 2-d matrix, hiddenSize x numImages,
% for Softmax layer
%activationsPooled由outputDim*outputDim*numFilters*numImages的矩阵
%变成hiddenSize*numImages的矩阵
%hiddenSize=outputDim*outputDim*numFilters
activationsPooled = reshape(activationsPooled,[],numImages);

%% Softmax Layer
%  Forward propagate the pooled activations calculated above into a
%  standard softmax layer. For your convenience we have reshaped
%  activationPooled into a hiddenSize x numImages matrix.  Store the
%  results in probs.

% numClasses x numImages for storing probability that each image belongs to
% each class.
probs = zeros(numClasses,numImages);

%%% YOUR CODE HERE %%%
size(Wd)
size(activationsPooled)
z=Wd*activationsPooled+repmat(bd,1,numImages);
%z= bsxfun(@minus, z, max(z)); %这个操作是为什么
y_hat=exp(z); %Wd:numClasses*hiddenSize, activationsPooled:hiddenSize*numImages
probs=y_hat./sum(y_hat,1); %一列表示一个样本属于每个类的概率

%%======================================================================
%% STEP 1b: Calculate Cost
%  In this step you will use the labels given as input and the probs
%  calculate above to evaluate the cross entropy objective.  Store your
%  results in cost.

cost = 0; % save objective into cost

%%% YOUR CODE HERE %%%

lambda=0.00001;
size(probs)
labels
size(labels)
I=sub2ind(size(probs), labels', 1:size(probs,2)); %后面两个参数维度必须相同，这里都是行向量
%注意sub2ind的后两个参数
value=log(probs(I));
wCost=lambda/2*(sum(Wd(:).^2)+sum(Wc(:).^2)); %weight_decay
cost=-sum(value)+wCost;
% Makes predictions given probs and returns without backproagating errors.
if pred
    [~,preds] = max(probs,[],1);
    preds = preds';
    grad = 0;
    return;
end;

%%======================================================================
%% STEP 1c: Backpropagation
%  Backpropagate errors through the softmax and convolutional/subsampling
%  layers.  Store the errors for the next step to calculate the gradient.
%  Backpropagating the error w.r.t the softmax layer is as usual.  To
%  backpropagate through the pooling layer, you will need to upsample the
%  error with respect to the pooling layer for each filter and each image.  
%  Use the kron function and a matrix of ones to do this upsampling 
%  quickly.

%%% YOUR CODE HERE %%%
y=full(sparse(1:length(labels),labels,1)); %一行一个样本，属于哪个类那这个列的值就为1
delta_d=(probs-y'); %输出层的误差
%Wd的每一行是一个类,每一列是一个hiddensize中的一维,numClasses*hiddensize
%hiddenSize=outputDim*outputDim*numFilters
delta_pool=Wd'*delta_d; %池化层的误差
%hiddenSize*numImages
%delta_pool是一个hiddenSize*numImages的二维矩阵
delta_pool=reshape(delta_pool, outputDim, outputDim, numFilters, numImages);
delta_i=zeros(convDim, convDim, numFilters, numImages);
%注意delta_i的维度，outputDim=convDim/poolDim
for i=1:numImages
 for j=1:numFilters
    size(kron(delta_pool(:,:,j,i),ones(poolDim)))
    size(activations)
    delta_i(:, :, j,i)=(1/poolDim^2)*kron(delta_pool(:,:,j,i),ones(poolDim)) .* activations(:,:,j,i).* (1-activations(:,:,j,i)); %卷积层的误差;
 endfor
endfor

%%======================================================================
%% STEP 1d: Gradient Calculation
%  After backpropagating the errors above, we can use them to calculate the
%  gradient with respect to all the parameters.  The gradient w.r.t the
%  softmax layer is calculated as usual.  To calculate the gradient w.r.t.
%  a filter in the convolutional layer, convolve the backpropagated error
%  for that filter with each image and aggregate over images.

%%% YOUR CODE HERE %%%
Wd_grad= delta_d*activationsPooled'+lambda*Wd; %* (classes*numImages) *(hiddenSize*numImages)^T
%先不加正则项
bd_grad= sum(delta_d,2); %delta_d的维度是numClasses*numImages,按行求和

%convolved_error=zeros(filterDim, filterDim, numFilters, numImages);
for i=1:numImages
 for j =1:numFilters
    delta_filter=delta_i(:,:,j,i); %convDim*convDim
    bc_grad(j)+=sum(delta_filter(:)); %bias的值是所有这个filter上的误差的和
    delta_filter=rot90(delta_filter, 2);
    im=images(:,:,i);
    convolved_filter_error=conv2(im, delta_filter,'valid'); %结果的维度是filterDim*filterDim
    disp("here") 
    size(convolved_filter_error)
    size(Wc_grad(:,:,j))
    size(delta_filter)
    convDim
    filterDim
    Wc_grad(:,:,j)+=convolved_filter_error; %把每个图像这个filter上误差的卷积都加起来
 endfor
endfor
Wc_grad+=lambda*Wc;
%cost在计算的时候没有除以图像数目，所以这里在求偏导时也不用除以图像数目

%% Unroll gradient into grad vector for minFunc
grad = [Wc_grad(:) ; Wd_grad(:) ; bc_grad(:) ; bd_grad(:)];

end
