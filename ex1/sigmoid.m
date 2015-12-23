function h=sigmoid(a)
  h=1./(1+exp(-a));

%右除表示C=X*B的解
%这里X=C./B,等价于C*inv(B)
%如果是标量的话，那么就是除以向量中的每个元素
