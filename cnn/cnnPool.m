function pooledFeatures = cnnPool(poolDim, convolvedFeatures)
%cnnPool Pools the given convolved features
%
% Parameters:
%  poolDim - dimension of pooling region
%  convolvedFeatures - convolved features to pool (as given by cnnConvolve)
%                      convolvedFeatures(imageRow, imageCol, featureNum, imageNum)
%
% Returns:
%  pooledFeatures - matrix of pooled features in the form
%                   pooledFeatures(poolRow, poolCol, featureNum, imageNum)

numImages = size(convolvedFeatures, 4);
numFilters = size(convolvedFeatures, 3);
convolvedDim = size(convolvedFeatures, 1);

pooledFeatures = zeros(convolvedDim / poolDim, ...
        convolvedDim / poolDim, numFilters, numImages);

% Instructions:
%   Now pool the convolved features in regions of poolDim x poolDim,
%   to obtain the 
%   (convolvedDim/poolDim) x (convolvedDim/poolDim) x numFeatures x numImages 
%   matrix pooledFeatures, such that
%   pooledFeatures(poolRow, poolCol, featureNum, imageNum) is the 
%   value of the featureNum feature for the imageNum image pooled over the
%   corresponding (poolRow, poolCol) pooling region. 
%   
%   Use mean pooling here.
poolLen=floor(convolvedDim/poolDim);
%%% YOUR CODE HERE %%%
 for imageNum = 1:numImages
  for filterNum = 1:numFilters
    featuremap = squeeze(convolvedFeatures(:,:,filterNum,imageNum));
    pooledFeaturemap = conv2(featuremap,ones(poolDim)/(poolDim^2),'valid');
    pooledFeatures(:,:,filterNum,imageNum) = pooledFeaturemap(1:poolDim:end,1:poolDim:end);

%用下面的循环程序跑一天也没跑完。。。。
%    for poolRow = 1:poolLen
%     RowIndexBegin=1+(poolRow-1)*poolDim;
%     RowIndexEnd=RowIndexBegin+poolDim-1;
%     for poolCol = 1:poolLen
%	ColIndexBegin=1+(poolCol-1)*poolDim;
%	ColIndexEnd=ColIndexBegin+poolDim-1;
%	pooledFeatures(poolRow, poolCol, filterNum, imageNum)=mean( mean(convolvedFeatures(RowIndexBegin:RowIndexEnd, ColIndexBegin:ColIndexEnd, filterNum, imageNum) ) );
%
%     endfor
%    endfor
  endfor
 endfor
end
