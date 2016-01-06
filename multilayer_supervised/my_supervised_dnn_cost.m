function [ cost, grad, pred_prob] = my_supervised_dnn_cost( theta, ei, data, labels, pred_only)
%SPNETCOSTSLAVE Slave cost function for simple phone net
%   Does all the work of cost / gradient computation
%   Returns cost broken into cross-entropy, weight norm, and prox reg
%        components (ceCost, wCost, pCost)

%% default values
po = false;
if exist('pred_only','var')
  po = pred_only;
end;

%% reshape into network
stack = params2stack(theta, ei); %存储参数，w是一个矩阵，b是一个列向量
numHidden = numel(ei.layer_sizes) - 1; %隐藏层的个数,不包括输入层和输出层
hAct = cell(numHidden+1, 1);
gradStack = cell(numHidden+1, 1);

%% forward prop
%%% YOUR CODE HERE %%%
for d = 1:numel(stack)
    W=stack{d}.W;
    b=stack{d}.b; %b是一个列向量
    if d==1
	hAct{d}=1./( 1+exp(  -(bsxfun(@plus, W*data, b)) ) ); 
	%W*data+b的每一列表示该样本在第一个隐层上面每个单元的数值
    elseif d < length(stack) %隐藏层
	hAct{d}=1./( 1+exp(  -(bsxfun(@plus, W*cell2mat(hAct{d-1}), b)) ) );
    else %输出层使用softmax
	y_hat=exp(bsxfun(@plus, W*cell2mat(hAct(d-1)),b));
	hAct{d}=bsxfun(@rdivide, y_hat, sum(y_hat));
    endif
endfor
%display("hAct");
%display(size(hAct));
pred_prob=hAct{end};
%最后一层的hAct的每一列表示一个样本属于每个类的概率
%% return here if only predictions desired.
if po
  cost = -1; ceCost = -1; wCost = -1; numCorrect = -1;
  grad = [];
  return;
end;

%% compute cost
%%% YOUR CODE HERE %%%

output=hAct{end}; %每列表示每个样本属于每个类的概率
trans_labels=labels'; %label是一个列向量
I=sub2ind(size( output' ), 1:size(output',1), trans_labels);
%label是列向量
trans_output=output';
ceCost=-sum(log(trans_output(I)));
ceCost;

wCost=0; %正则项

for d = 1:numel(stack)
    w=stack{d}.W;
    square_w=w.^2;
    wCost+=sum(square_w(:));
endfor

wCost*=(ei.lambda/2);
%weight_decay;
cost=ceCost+wCost;
cost;
%% compute gradients using backpropagation
%%% YOUR CODE HERE %%%

for d = numel(stack)+1:-1:1
    if d == numel(stack)+1 %输出层
	y=full(sparse(1:length(trans_labels), trans_labels, 1)); %把y扩展,一行一个样本
	delta=hAct{d-1}-y';
%delta是一个矩阵，每个样本关于每个类都有一个误差
%	delta=sum(hAct{d}-y',2) %按样本即按行求所有样本的残差和,得到一个列向量
    elseif d== 1
	gradStack{d}.W=delta*(data)'; %+ei.lambda*stack{d}.W;%delta的每一行对应一个输出单元，每列对应一个样本，hAct的每一列对应一个样本，每行对应一个输出单元
	gradStack{d}.b=sum(delta,2); %按样本求和
    else
	gradStack{d}.W=delta*(hAct{d-1})';%delta的每一行对应一个输出单元，每列对应一个样本，hAct的每一列对应一个样本，每行对应一个输出单元
	gradStack{d}.b=sum(delta,2); %按样本求和
	delta=(stack{d}.W)' * delta .* hAct{d-1}.*(1-hAct{d-1});
%    else %输出层没有权重衰减项
%	gradStack{d}.W=delta*(hAct{d-1})';%delta的每一行对应一个输出单元，每列对应一个样本，hAct的每一列对应一个样本，每行对应一个输出单元
%	gradStack{d}.b=sum(delta,2); %按样本求和
%	delta=(stack{d}.W)' * delta .* ( (hAct{d-1}.*(1-hAct{d-1})) );
    endif
endfor
%+ei.lambda*stack{d}.W
size(gradStack);
size(stack);
size(hAct);
%% compute weight penalty cost and gradient for non-bias terms
%%% YOUR CODE HERE %%%


for l = numHidden : -1 : 1
    gradStack{l}.W = gradStack{l}.W + ei.lambda * stack{l}.W;%softmax没用到权重衰减项
end

%% reshape gradients into vector
[grad] = stack2params(gradStack);
%display("grad");
%display("grad");
end
