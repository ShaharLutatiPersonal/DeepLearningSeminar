import torch.nn as nn
import torch.autograd as grad
import torch
"""
According to the article they defined an LSTM block as the following

[i;f;o;g] = [sigm;sigm;sigm;tanh] * T_2n,4n * [h_t^l-1;h_t-1^l]
where t - is subscript of time (first word is in time t = 0)
and l is upscript of layer index.
the LSTM cell defined as two main components
c_t^l = f (*) c_t-1^l + i(*)g 
h_t^l = o(*)tanh(c_t^l)
where (*) is the operator for elementwise multiplication (Hadamard product)
we can see that it is a kind of ARMA filter of the state
where the moving average is the f(*)c part and the AR part is the i(*)g part
with the use of tanh as the non linearity. As the coefficents of f is getting smaller
the LSTM cell will exponenitally remember less in long term.

In the article they introduced a new technique for dropout
[i;f;o;g] = [sigm;sigm;sigm;tanh] * T_2n,4n * [D(h_t^l-1);h_t-1^l]
meaning they doing a dropout on the inputs for the next layer.
"""
class RNN_Zam(nn.Module):
    def __init__(self,dict_size,dp_prob = 0.0):
        super(RNN_Zam, self).__init__()
        self.embedding = nn.Embedding(num_embeddings = dict_size,embedding_dim = 200)
        self.dropout = nn.Dropout(dp_prob)
        # According to pytorch documentaion when applying a dropout !=0 it applies a dropout on the output except for the last layer.
        # It does not apply the dropout to the state variable
        # We double checked it on the source code of pytorch :)  (Google search: pytorch LSTM source -> _VF.lstm())
        self.lstm = nn.LSTM(input_size = 200,hidden_size = 200 ,num_layers =  2,dropout = dp_prob)
        self.fc = nn.Linear(in_features= 200,out_features= dict_size)
        # According to the paper we should use weights initialization
        # Reading Xavier weight initialization paper for symmetric non linearty (if we used relu we use Kaimen)
        # we normalize by using the factor sqrt(6)/sqrt(num_input + num_output)
        for parameter in self.parameters():
            nn.init.uniform_(parameter,-.1,.1)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.xavier_uniform_(self.embedding.weight)
    def forward(self, input,inner_states):
        output = self.embedding(input)
        output = self.dropout(output)
        output,(h_t,c) = self.lstm(output,inner_states)
        output = self.dropout(output)
        output = output.view(-1,200) 
        output = self.fc(output)
        return output,(h_t,c)
class GRU(nn.Module):
    def __init__(self,dict_size,dp_prob = 0.0):
        super(GRU, self).__init__()
        self.embedding = nn.Embedding(num_embeddings = dict_size,embedding_dim = 200)
        self.dropout = nn.Dropout(dp_prob)
        # According to pytorch documentaion when applying a dropout !=0 it applies a dropout on the output except for the last layer.
        # It does not apply the dropout to the state variable
        # We double checked it on the source code of pytorch :)  (Google search: pytorch LSTM source -> _VF.lstm())
        self.gru = nn.GRU(input_size = 200,hidden_size = 200 ,num_layers =  2,dropout = dp_prob)
        self.fc = nn.Linear(in_features= 200,out_features= dict_size)
        # According to the paper we should use weights initialization
        # Reading Xavier weight initialization paper for symmetric non linearty (if we used relu we use Kaimen)
        # we normalize by using the factor sqrt(6)/sqrt(num_input + num_output)
        for parameter in self.parameters():
            nn.init.uniform_(parameter,-.1,.1)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.xavier_uniform_(self.embedding.weight)
    def forward(self, input,inner_states):
        output = self.embedding(input)
        output = self.dropout(output)
        output,(h_n) = self.gru(output,inner_states)
        output = self.dropout(output)
        output = output.view(-1,200) 
        output = self.fc(output)
