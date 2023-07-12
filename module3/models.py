import torch
import torch.nn as nn
import numpy as np

class Baseline(nn.Module):
    def __init__(self, out_features=1, name="palceholder",  training_args={}):
        super(Baseline, self).__init__()
        self.kernel_size = 3
        self.name = f'Baseline_F{out_features}'
        self.kvargs = {
            'name': self.name,
            'training_args': training_args,
        }
        self.out_features = out_features

        # conv layer
        downsample = self._downsample(4096, 128)
        self.conv1 = nn.Conv1d(in_channels=8, 
                               out_channels=32, 
                               kernel_size=self.kernel_size, 
                               stride=downsample,
                               padding=self._padding(downsample),
                               bias=False)
        
        # linear layer
        self.lin = nn.Linear(in_features=32*128,
                             out_features=out_features)
        
        # ReLU
        self.relu = nn.ReLU()

    def _padding(self, downsample):
        return max(0, int(np.floor((self.kernel_size - downsample + 1) / 2)))

    def _downsample(self, seq_len_in, seq_len_out):
        return int(seq_len_in // seq_len_out)


    def forward(self, x):
        x= x.transpose(2,1)

        x = self.relu(self.conv1(x))
        x_flat= x.view(x.size(0), -1)
        x = self.lin(x_flat)

        return x
    
class Conv1Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, steps=2, stride=1):
        super().__init__()
        self.steps = steps
        self.layers = nn.Sequential(
            nn.Conv1d(in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=kernel_size//2,
                        bias=False),
            nn.ReLU())
        self.skip_layers = nn.Sequential(
            *[ 
                nn.Sequential(
                nn.BatchNorm1d(out_channels), 
                nn.Conv1d(in_channels=out_channels,
                            out_channels=out_channels,
                            kernel_size=kernel_size,
                            stride=1,
                            padding=kernel_size//2,
                            bias=False),
                nn.ReLU()
            )for _ in range(steps)]
        )
    
    def forward(self, x):
        x = self.layers(x)
        if self.steps > 0:
            x = x + self.skip_layers(x)
        return x
    
class Model(nn.Module):
    def __init__(self, 
                 kernel_size=3, 
                 num_layers=5, 
                 steps=2, 
                 dropout=0.5, 
                 lin_steps=1, 
                 in_channels=8, 
                 out_features=1, 
                 data_width=4096, 
                 name="palceholder", 
                 training_args={},
                 ):
        super().__init__()
        self.name = f"Model_F{out_features}"
        self.kvargs = {
            "name": self.name,
            "kernel_size": kernel_size,
            "num_layers": num_layers,
            "steps": steps,
            "dropout": dropout,
            "lin_steps": lin_steps,
            "in_channels": in_channels,
            "out_features": out_features,
            "data_width": data_width,
            "training_args": training_args,
        }
        self.out_features = out_features

        self.encoder = nn.Sequential(
            *[ Conv1Block(in_channels*(2**i), in_channels*(2**(i+1)), kernel_size, steps, stride=2) for i in range(num_layers)])
        self.reducer = nn.Sequential(
            *[ Conv1Block(in_channels*(2**i), in_channels*(2**(i-1)), kernel_size, steps, stride=1) for i in range(num_layers, 0, -1)], 
            Conv1Block(8, 1, kernel_size, steps, stride=1)
            )
        # linear layer
        out_channels = data_width//(2**(num_layers))
        self.lin = nn.Sequential(
            *[nn.Sequential(
                nn.Linear(in_features=out_channels,
                                out_features=out_channels), 
                nn.Dropout(dropout), 
                nn.ReLU()
            ) for _ in range(lin_steps)],
            nn.Linear(in_features=out_channels,
                                out_features=out_features),
        )
        

    def forward(self, x):
        x= x.transpose(2,1)
        x = self.encoder(x)
        x = self.reducer(x)
        x_flat= x.view(x.size(0), -1)
        x = self.lin(x_flat)
        return x

class Model_V2(nn.Module):
    def __init__(self, 
                 kernel_size=3, 
                 encode_layers=3,
                 encoder_out_channels=256,
                 reduce_layers=3,
                 reduce_out_channels=4,
                 steps=1, 
                 dropout=0.5, 
                 lin_steps=1, 
                 lin_dims=256, 
                 in_channels=8,
                 out_features=1, 
                 data_width=4096, 
                 name="palceholder", 
                 training_args={}
                 ):
        super().__init__()
        self.name = f"Model_F{out_features}_V2"
        self.kvargs = {
            "name": self.name,
            "kernel_size": kernel_size,
            "encode_layers": encode_layers,
            "encoder_out_channels": encoder_out_channels,
            "reduce_layers": reduce_layers,
            "reduce_out_channels": reduce_out_channels,
            "steps": steps,
            "dropout": dropout,
            "lin_steps": lin_steps,
            "lin_dims": lin_dims,
            "in_channels": in_channels,
            "out_features": out_features,
            "training_args": training_args,
        }
        self.out_features = out_features
        encoder_channels = np.linspace(in_channels, encoder_out_channels, encode_layers+1, dtype=int)
        self.encoder = nn.Sequential(
            *[ Conv1Block(in_c, out_c, kernel_size, steps, stride=2) for in_c, out_c in zip(encoder_channels[:-1], encoder_channels[1:])])
        
        reducer_channels = np.linspace(encoder_out_channels, reduce_out_channels, reduce_layers+1, dtype=int)
        self.reducer = nn.Sequential(
            *[ Conv1Block(in_c, out_c, kernel_size, steps,  stride=1) for in_c, out_c in zip(reducer_channels[:-1], reducer_channels[1:])]
            )
        # linear layer
        out_channels = reduce_out_channels*data_width//(2**(encode_layers))
        self.lin = nn.Sequential(
            nn.Linear(in_features=out_channels,
                                out_features=lin_dims),
            nn.ReLU(),
            *[nn.Sequential(
                nn.Linear(in_features=lin_dims,
                                out_features=lin_dims), 
                nn.Dropout(dropout), 
                nn.ReLU()
            ) for _ in range(lin_steps)],
            nn.Linear(in_features=lin_dims,
                                out_features=1),
        )
        

    def forward(self, x):
        x= x.transpose(2,1)
        x = self.encoder(x)
        x = self.reducer(x)
        x_flat= x.view(x.size(0), -1)
        x = self.lin(x_flat)
        return x
    