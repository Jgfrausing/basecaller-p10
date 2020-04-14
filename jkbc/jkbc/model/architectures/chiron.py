from fastai.basics import *

NF = 256
HIDDEN_UNITS = 200

def model(window_size, dim_pred_out, device, alphabet_size, bs):
    return model = nn.Sequential(
        ResidualBlock(1, NF)
        ,ResidualBlock(NF, NF)
        ,ResidualBlock(NF, NF)
        ,ResidualBlock(NF, NF)
        ,ResidualBlock(NF, dim_pred_out)
        ,nn.BatchNorm1d(dim_pred_out)

        ,LstmBlock(window_size, bs, HIDDEN_UNITS, no_of_layers=4, device=device)

        ,nn.Linear(HIDDEN_UNITS*2,alphabet_size)
    ).to(device=device)

def conv(ni, nf, ks=1, padding=0): return nn.Conv1d(ni, nf, kernel_size=ks, stride=1, padding=padding)
def conv_layers(ni, nf): 
    return nn.Sequential(
        conv(ni, NF)
        ,nn.BatchNorm1d(NF)
        ,nn.ReLU()
        ,conv(NF, NF, 3, padding=1)
        ,nn.BatchNorm1d(NF)
        ,nn.ReLU()
        ,conv(NF, nf)
    )

class ResidualBlock(nn.Module):
    def __init__(self, ni, nf):
        super().__init__()
        self.module = conv_layers(ni, nf)
        self.residual = conv(ni, nf)
    
    def forward(self, x):
        out_a = self.module(x)
        out_b = self.residual(x)
        
        return nn.ReLU()(out_a + out_b)

class LstmBlock(nn.Module):
    def __init__(self, input_size, batch_size, hidden_units, no_of_layers, device):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_units, no_of_layers, bidirectional=True, batch_first=True)
        
        ## Multiply by 2 because of bidirectional
        h0 = torch.zeros(2*no_of_layers, batch_size, hidden_units).to(device=device)
        c0 = torch.zeros(2*no_of_layers, batch_size, hidden_units).to(device=device)
        
        self.hidden=(h0,c0)
        
    def forward(self, x):                
        res, _ = self.lstm(x, self.hidden)
        
        return res

