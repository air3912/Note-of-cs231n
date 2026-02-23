### 线性分类
#### KNN
我就我们将一个图片展平然后得到一个向量，我们将之作为高维度下的一个坐标，然后我们需要先给模型喂（实际上让计算机记住就行了）很多的图片以及对应的种类，然后高维下每个图片都有自己的坐标，找出与目标图片距离最近的k个，最后根据这k个的种类决定目标图片的种类
#### nn.Linear(in_features,out_features)
输入一维向量，需要事先把图片展平nn.Flatten,得到向量，输入矩阵中，得到各个种类的概率得分，这里有个处理偏置b的技巧，至于矩阵怎么来的我们接下来说，

### 损失与优化（训练）
```python
for epoch in range(num_epochs)
    outputs=model(inputs)#这里一个线性分类就是一个nn.linear(inputs,num_class)
    loss=criterion(outputs,labels)
    #softmax - nn.CrossEntropyLoss交叉熵损失
    #SVM - nn.MultoLabelMarginLoss折页损失
    opertimizer.zero_grad()
    loss.backward()
    optimizer.step()

```
- 损失：

 

- 反向传播：




- 优化：







### 神经网络
实际上就是通过激活函数使线性函数呈非线性，有了激活函数理论上可以拟合任意函数



### CNN
课堂上也说了很多经典的backbone
nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding)
一共需要讲的就这几个参数
- 第一个参数深度：唯一需要说的是我们卷积核的深度必须和输入深度相同，一个卷积核只能得到一层特征，想要更多的必须再用别的卷积核
- 第二个：输出通道数取决于我们卷积核的数量
- 第三个：只需要WH两个参数，因为深度固定，如果传入一个参数代表是正方形
- 第四个：步长，也是可以接受一个元组参数，（两个参数，代表横向和纵向）正常我们每次卷积操作卷积核移动的距离
- 第五个：填充，参数同上，代表需要在外围填充的长度

然后再讲讲池化
常见有：
- 最大池化
- 平均池化
有时候还可以配合重叠池化（就是步长小于长度，这样就重叠了）
### RNN

注(important)：最开始学深度学习的时候很多时候都有种违和感，后来才发现只因为在之前线性代数的学习中我们提到向量默认他是列向量，但是在深度学习的工程实践中我们提到的的向量是默认行向量的


核心概念：时间步，隐藏状态，嵌入embedding


#### 原始RNN
nn.RNN(input_size,hidden_size,batch_first)
nn.RNN(10, 20, batch_first=True)
输入 Shape： (3, 5, 10) 输出 Shape： (3, 5, 20)
这里解释一下输入，首先10是embedding维度，然后3是3句话，5是每句话的token数，rnn是每句话一起开始运算的
再解释一下输出，这里nn.RNN的输出就默认指的是隐藏状态，而不是最终输出

#### LSTM
扩展慨念：细胞状态



#### GRU


### transformer
#### attention

#### encoder

#### decoder

