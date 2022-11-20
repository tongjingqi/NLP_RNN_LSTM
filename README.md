# NLP_RNN_LSTM
一、系统设计  
分别使用调库和自己实现的方式，实现了基于RNN和LSTM的语言模型。  
系统模块介绍： 
（一）数据处理模块  
1.建立词表   
```
word2number_dict, number2word_dict = make_dict(train_path) #use the make_dict function to make the dict   
```
使用make_dict函数，将训练集合中的词的集合编号。   
其中，0和<pad>相对应，1和<unk_word>相对于，作用分别是填充训练样本空缺词用pad，验证集合中有未见过的词为unk_word。   
2.将数据集转为每个batch的形式  
```
all_input_batch, all_target_batch = make_batch(train_path, word2number_dict, batch_size, n_step) # make the batch
```
将数据集中的句子转化成多个batch，每个batch里有batch_size个n_step长度的句子，句子标签是最后一个词。如果句子长度不够则padding。 对于每一个数据集的句子，以n_step为滑动窗口采集数 据。  
（二）模型定义模块   
1.LSTM的从头实现   

```
class TextRNN(nn.Module):
    def __init__(self):
        super(TextRNN, self).__init__()#！！！！注意改名
        self.C = nn.Embedding(n_class, embedding_dim=emb_size,device=device)

   
        '''define the parameter of LSTM'''
        '''begin'''
        self.W_xi=nn.Linear(emb_size,n_hidden)
        self.W_hi=nn.Linear(n_hidden,n_hidden)
        self.b_i=nn.Parameter(torch.ones([n_hidden]))

        self.W_xf=nn.Linear(emb_size,n_hidden)
        self.W_hf=nn.Linear(n_hidden,n_hidden)
        self.b_f=nn.Parameter(torch.ones([n_hidden]))

        self.W_xo=nn.Linear(emb_size,n_hidden)
        self.W_ho=nn.Linear(n_hidden,n_hidden)
        self.b_o=nn.Parameter(torch.ones([n_hidden]))

        self.W_xc=nn.Linear(emb_size,n_hidden)
        self.W_hc=nn.Linear(n_hidden,n_hidden)
        self.b_c=nn.Parameter(torch.ones([n_hidden]))

        self.tanh=nn.Tanh()
        self.sigmoid=nn.Sigmoid()
        '''end'''
        self.W = nn.Linear(n_hidden, n_class, bias=False,device=device)
        self.b = nn.Parameter(torch.ones([n_class])).to(device)
        self.softmax=nn.Softmax(dim=-1)

    def forward(self, X):

        X = self.C(X)
        #print(X.is_cuda)
        X = X.transpose(0, 1) # X : [n_step, batch_size, n_class]
        sample_size = X.size()[1]
        '''do this RNN forward'''
        '''begin'''
        ##Complete your code with the hint: a^(t) = tanh(W_{ax}x^(t)+W_{aa}a^(t-1)+b_{a})  y^(t)=softmx(Wa^(t)+b)
        c_t = torch.zeros(1,n_hidden,device=device)
        h_t= torch.zeros(1,n_hidden,device=device)
        model_output=0
        for x in X:
            i_t=self.sigmoid(self.W_xi(x)+self.W_hi(h_t)+self.b_i)
            f_t=self.sigmoid(self.W_xf(x)+self.W_hf(h_t)+self.b_f)
            o_t=self.sigmoid(self.W_xo(x)+self.W_ho(h_t)+self.b_o)
            c_t=f_t*c_t+i_t*self.tanh(self.W_xc(x)+self.W_hc(h_t)+self.b_c)
            h_t=o_t*self.tanh(c_t)
            #a_t=self.tanh(self.W_ax(x)+self.W_aa(a_t)+self.b_a)
        model_output = self.W(h_t) + self.b
        '''end'''
        return model_output
```
2.双层LSTM的实现
```
class TextRNN(nn.Module):
    def __init__(self):
        super(TextRNN, self).__init__()#！！！！注意改名
        self.C = nn.Embedding(n_class, embedding_dim=emb_size,device=device)

        # '''define the parameter of RNN'''
        # '''begin'''
        # ##Complete this code
        # self.W_ax = nn.Linear(emb_size,n_hidden,bias=False,device=device)
        # self.W_aa = nn.Linear(n_hidden,n_hidden,bias=False,device=device)
        # self.b_a = nn.Parameter(torch.ones([n_hidden])).to(device)
        # '''end'''
        #
        # self.tanh = nn.Tanh()
        '''define the parameter of LSTM'''
        '''begin'''
        self.W_xi=nn.Linear(emb_size,n_hidden)
        self.W_hi=nn.Linear(n_hidden,n_hidden)
        self.b_i=nn.Parameter(torch.ones([n_hidden]))

        self.W_xf=nn.Linear(emb_size,n_hidden)
        self.W_hf=nn.Linear(n_hidden,n_hidden)
        self.b_f=nn.Parameter(torch.ones([n_hidden]))

        self.W_xo=nn.Linear(emb_size,n_hidden)
        self.W_ho=nn.Linear(n_hidden,n_hidden)
        self.b_o=nn.Parameter(torch.ones([n_hidden]))

        self.W_xc=nn.Linear(emb_size,n_hidden)
        self.W_hc=nn.Linear(n_hidden,n_hidden)
        self.b_c=nn.Parameter(torch.ones([n_hidden]))

        self.W_xi2 = nn.Linear(n_hidden, n_hidden)
        self.W_hi2 = nn.Linear(n_hidden, n_hidden)
        self.b_i2 = nn.Parameter(torch.ones([n_hidden]))

        self.W_xf2 = nn.Linear(n_hidden, n_hidden)
        self.W_hf2 = nn.Linear(n_hidden, n_hidden)
        self.b_f2 = nn.Parameter(torch.ones([n_hidden]))

        self.W_xo2 = nn.Linear(n_hidden, n_hidden)
        self.W_ho2 = nn.Linear(n_hidden, n_hidden)
        self.b_o2 = nn.Parameter(torch.ones([n_hidden]))

        self.W_xc2 = nn.Linear(n_hidden, n_hidden)
        self.W_hc2 = nn.Linear(n_hidden, n_hidden)
        self.b_c2 = nn.Parameter(torch.ones([n_hidden]))

        self.tanh=nn.Tanh()
        self.sigmoid=nn.Sigmoid()
        '''end'''
        self.W = nn.Linear(n_hidden, n_class, bias=False,device=device)
        self.b = nn.Parameter(torch.ones([n_class])).to(device)
        self.softmax=nn.Softmax(dim=-1)

    def forward(self, X):

        X = self.C(X)
        #print(X.is_cuda)
        X = X.transpose(0, 1) # X : [n_step, batch_size, n_class]
        sample_size = X.size()[1]
        '''do this RNN forward'''
        '''begin'''
        ##Complete your code with the hint: a^(t) = tanh(W_{ax}x^(t)+W_{aa}a^(t-1)+b_{a})  y^(t)=softmx(Wa^(t)+b)
        c_t = torch.zeros(1,n_hidden,device=device)
        h_t= torch.zeros(1,n_hidden,device=device)
        model_output=0
        H=list()
        for x in X:
            i_t=self.sigmoid(self.W_xi(x)+self.W_hi(h_t)+self.b_i)
            f_t=self.sigmoid(self.W_xf(x)+self.W_hf(h_t)+self.b_f)
            o_t=self.sigmoid(self.W_xo(x)+self.W_ho(h_t)+self.b_o)
            c_t=f_t*c_t+i_t*self.tanh(self.W_xc(x)+self.W_hc(h_t)+self.b_c)
            h_t=o_t*self.tanh(c_t)
            H.append(h_t)
            #a_t=self.tanh(self.W_ax(x)+self.W_aa(a_t)+self.b_a)
        h_t2 = torch.zeros(1, n_hidden, device=device)
        c_t2 = torch.zeros(1, n_hidden, device=device)
        for x in H:
            i_t2=self.sigmoid(self.W_xi2(x)+self.W_hi2(h_t2)+self.b_i2)
            f_t2=self.sigmoid(self.W_xf2(x)+self.W_hf2(h_t2)+self.b_f2)
            o_t2=self.sigmoid(self.W_xo2(x)+self.W_ho2(h_t2)+self.b_o2)
            c_t2=f_t2*c_t2+i_t2*self.tanh(self.W_xc2(x)+self.W_hc2(h_t2)+self.b_c2)
            h_t2=o_t2*self.tanh(c_t2)
            #a_t=self.tanh(self.W_ax(x)+self.W_aa(a_t)+self.b_a)
        model_output = self.W(h_t2) + self.b
        '''end'''
        return model_output
```


（三）训练，验证与测试模块   
在GPU上进行

二、实验结果与分析   

初始参数为：
```
n_step = 5 # number of cells(= number of Step)
n_hidden = 5 # number of hidden units in one cell
batch_size = 512 #batch size
learn_rate = 0.001
all_epoch = 100 #the all epoch for training
emb_size = 128 #embeding size
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learn_rate)
ppl =math.exp(total_loss / count_loss)
```

第一组实验：
对于从头实现的LSTM，将初始参数逐个单独调大，看测试集的损失、PPL值以及耗费的总时间：（这里应该是不对的，不能用测试集做实验，应该用验证集调参，最后在测试集上看效果，所以这里的测试集也相对于验证集了）
	   

![图片](https://user-images.githubusercontent.com/66310692/202902028-2cc19f8e-d26b-41ba-ba88-561f01b4cbeb.png)

实验分析：相对于初始参数，
n_step从5调到10反而会让性能整体下降
n_hidden从5变大到10会让ppl性能显著变好，同时时间无显著影响
Batch_size变大让ppl性能变差，但是时间会变快(使用GPU)
lr变大10倍，ppl性能显著下降
epoch变大，性能无明显变化，时间显著变长
Embeding_size变大，性能无明显变化，时间变长一些
使用SGD优化器性能非常显著的变差

第二组实验
对于从头实现的LSTM，将各个参数相对于初始参数调小，观察训练集ppl，验证ppl，测试ppl，以及记录总时间。
	
![图片](https://user-images.githubusercontent.com/66310692/202902125-1c25350c-5a80-4213-bfd5-2e873a6ea671.png)

实验分析：相对初始参数，
N_step调小，train_ppl显著变大，但是验证和测试ppl略微变大。
H_hidden调小，ppl值三个数据集都变大，性能下降
Batch_size调小一半性能略微下降，但是时间变成原来的两倍
lr调小一半，性能下降，时间不变
epoch调小一半，性能有所下降，但是时间也减半
embsize调小一半，性能有所下降，时间影响不大

第三组实验：
不再控制变量，进行逐步调优
	
![图片](https://user-images.githubusercontent.com/66310692/202902244-a7002cb0-d499-4292-a515-93b3879704bd.png)

实验分析：
其他参数相同前提下，epoch在25时的结果有时比50效果好。
H_hidden对结果影响较大，h_hidden较小时，其值和性能正相关而且对时间影响不大。最终找到一个相对较优解h_hidden=50
Batch_size增大可以提速，但是性能可能会下降
而其他参数的调整对性能提升的作用不大
最终ppl值可以接近370 。

第四组实验
分别用之前调优的参数去分别测试从头实现、调库实现的LSTM和RNN，进行四组实验
	
![图片](https://user-images.githubusercontent.com/66310692/202906304-7dd7fa0b-9a74-453f-8fa3-4d7c40df1d84.png)
实验分析：LSTM的调库和从头实现对比，调库实现LSTM速度快很多，在性能上手动实现的略高
RNN的调库和从头实现差距不大
LSTM比RNN性能略高


实验结论
RNN和LSTM总体性能差距不大，LSTM在PPL值方面更优秀，而RNN速度更快
调库实现LSTM和从头实现LSTM性能差距不大，而调库实现速度明显更快
手动实现可能性能略优于调库
参数方面，h_hidden调参效果明显，随着h_hidden增大，性能先上升后下降，在大约50处相对较优。
epoch变大性能不一定变好，但是速度会近似与其成反比，因此实验选取较小epoch较好
Batch_size变大性能可能会变差，但是训练速度正相关于其大小。
优化器需要选Adam，选择SGD损失极大
