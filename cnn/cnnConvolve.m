function convolvedFeatures = cnnConvolve(filterDim, numFilters, images, W, b)
%cnnConvolve Returns the convolution of the features given by W and b with
%the given images
% Parameters:
%  filterDim - filter (feature) dimension %这个是表示卷积单元(filter)的大小？比如在8*8的patch上进行卷积
%  numFilters - number of feature maps %
%  images - large images to convolve with, matrix in the form
%           images(r, c, image number)
%  W, b - W, b for features from the sparse autoencoder
%         W is of shape (filterDim,filterDim,numFilters)
%         b is of shape (numFilters,1)
%
% Returns:
%  convolvedFeatures - matrix of convolved features in the form
%                      convolvedFeatures(imageRow, imageCol, featureNum, imageNum)
numImages = size(images, 3);
imageDim = size(images, 1); %images的行具有多少像素
convDim = imageDim - filterDim + 1;

convolvedFeatures = zeros(convDim, convDim, numFilters, numImages);
%四维矩阵

%(imageDim - filterDim + 1) x (imageDim - filterDim + 1)是每个图片上的卷积单元个数
%numFeatures是filter的个数
% Instructions:
%   Convolve every filter with every image here to produce the 
%   (imageDim - filterDim + 1) x (imageDim - filterDim + 1) x numFeatures x numImages
%   matrix convolvedFeatures, such that 
%   convolvedFeatures(imageRow, imageCol, featureNum, imageNum) is the
%   value of the convolved featureNum feature for the imageNum image over
%   the region (imageRow, imageCol) to (imageRow + filterDim - 1, imageCol + filterDim - 1)
%在区域(imageRow, imageCol) to (imageRow + filterDim - 1, imageCol + filterDim - 1)上计算第featureNum个卷积特征
%
% Expected running times: 
%   Convolving with 100 images should take less than 30 seconds 
%   Convolving with 5000 images should take around 2 minutes
%   (So to save time when testing, you should convolve with less images, as
%   described earlier)

for imageNum = 1:numImages
  for filterNum = 1:numFilters %每个filter

    % convolution of image with feature matrix
    convolvedImage = zeros(convDim, convDim); %保存对imageNum图像进行filterNum特征卷积的结果

    % Obtain the feature (filterDim x filterDim) needed during the convolution
    %%% YOUR CODE HERE %%%
    filter=W(:,:,filterNum);
    %squeeze函数把矩阵中维数为1的那一维给去掉，但是对于二维矩阵并没有影响，如果是一个向量或者标量，那么也没有影响
    % Flip the feature matrix because of the definition of convolution, as explained later
    filter = rot90(squeeze(filter),2);
      
    % Obtain the image
    im = squeeze(images(:, :, imageNum));

    % Convolve "filter" with "im", adding the result to convolvedImage
    % be sure to do a 'valid' convolution

    %%% YOUR CODE HERE %%%
    %%im的维度是imageDim*imageDim，filter的维度是filterDim*filterDim,卷积后的结果矩阵维度是(imageDim-filterDim+1)*(imageDim-filterDim+1)
    convolvedImage = convolvedImage + conv2(im, filter, 'valid');
    % Add the bias unit
    % Then, apply the sigmoid function to get the hidden activation

    %%% YOUR CODE HERE %%%
    convolvedImage = convolvedImage+b(filterNum);
    convolvedImage=1./(1+exp(-convolvedImage ) ); %sigmoid(convolvedImage)
    convolvedFeatures(:, :, filterNum, imageNum) = convolvedImage;
  end
end
end

