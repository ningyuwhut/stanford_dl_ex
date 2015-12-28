function [ stack ] = initialize_weights( ei )
%INITIALIZE_WEIGHTS Random weight structures for a network architecture
%   eI describes a network via the fields layerSizes, inputDim, and outputDim 
%   
%   This uses Xavier's weight initialization tricks for better backprop
%   See: X. Glorot, Y. Bengio. Understanding the difficulty of training 
%        deep feedforward neural networks. AISTATS 2010.

%% initialize hidden layers,包括输出层
stack = cell(1, numel(ei.layer_sizes)); %numel 返回对象A中的元素个数
%%cell貌似是创建一个结构体数组
for l = 1 : numel(ei.layer_sizes)
    if l > 1
        prev_size = ei.layer_sizes(l-1);
    else
        prev_size = ei.input_dim;
    end;
    cur_size = ei.layer_sizes(l);
    % Xaxier's scaling factor
    s = sqrt(6) / sqrt(prev_size + cur_size);
    stack{l}.W = rand(cur_size, prev_size)*2*s - s; %生成一个行数为cur_size,列数为prev_size的矩阵
    stack{l}.b = zeros(cur_size, 1);
    %隐藏层或者输出层中的每个单元都有一个b
end
