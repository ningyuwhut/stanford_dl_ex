function stack = params2stack(params, ei)

% Converts a flattened parameter vector into a nice "stack" structure 
% for us to work with. This is useful when you're building multilayer
% networks.
%
% stack = params2stack(params, netconfig)
%
% params - flattened parameter vector
% ei - auxiliary variable containing 
%             the configuration of the network
%

% Map the params (a vector into a stack of weights)
depth = numel(ei.layer_sizes);
stack = cell(depth,1);%depth*1的cell矩阵
% the size of the previous layer
prev_size = ei.input_dim; 
% mark current position in parameter vector
cur_pos = 1;

for d = 1:depth
    % Create layer d
    stack{d} = struct;

    hidden = ei.layer_sizes(d); %隐藏层d节点单元个数
    % Extract weights
    wlen = double(hidden * prev_size);%隐藏层d的权重参数个数
    stack{d}.W = reshape(params(cur_pos:cur_pos+wlen-1), hidden, prev_size);
    cur_pos = cur_pos+wlen;

    % Extract bias
    blen = hidden;%隐藏层中的每个输出单元都有一个偏置项
    stack{d}.b = reshape(params(cur_pos:cur_pos+blen-1), hidden, 1);
    cur_pos = cur_pos+blen;
    
    % Set previous layer size
    prev_size = hidden;
end
end
