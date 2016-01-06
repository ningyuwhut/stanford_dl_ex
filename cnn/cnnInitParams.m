function theta = cnnInitParams(imageDim,filterDim,numFilters,...
                                poolDim,numClasses)
% Initialize parameters for a single layer convolutional neural
% network followed by a softmax layer.
%                            
% Parameters:
%  imageDim   -  height/width of image
%  filterDim  -  dimension of convolutional filter                            
%  numFilters -  number of convolutional filters
%  poolDim    -  dimension of pooling area
%  numClasses -  number of classes to predict
%
%
% Returns:
%  theta      -  unrolled parameter vector with initialized weights

%% Initialize parameters randomly based on layer sizes.
assert(filterDim < imageDim,'filterDim must be less that imageDim');

Wc = 1e-1*randn(filterDim,filterDim,numFilters); %randn返回符合标准正态分布的随机数矩阵,这是卷积层的参数，注意维度

outDim = imageDim - filterDim + 1; % dimension of convolved image ,卷积层的特征维度outDim*outDim

% assume outDim is multiple of poolDim , outDim必须是poolDim的倍数
assert(mod(outDim,poolDim)==0,...
       'poolDim must divide imageDim - filterDim + 1');

outDim = outDim/poolDim; %这是降采样层的维度
hiddenSize = outDim^2*numFilters; %降采样层的特征个数,这就是隐层的输出单元个数?

% we'll choose weights uniformly from the interval [-r, r]
r  = sqrt(6) / sqrt(numClasses+hiddenSize+1);
Wd = rand(numClasses, hiddenSize) * 2 * r - r; %rand返回0，1之间均值分布的参数矩阵，这里的操作就是把Wd变成了[-r,r]之间的均匀分布

bc = zeros(numFilters, 1);
bd = zeros(numClasses, 1);

% Convert weights and bias gradients to the vector form.
% This step will "unroll" (flatten and concatenate together) all 
% your parameters into a vector, which can then be used with minFunc. 
theta = [Wc(:) ; Wd(:) ; bc(:) ; bd(:)];

end

