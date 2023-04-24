import torch 
from torch import nn


class Lstm_model(nn.Module): 
# SOURCE: https://github.com/closeheat/pytorch-lstm-text-generation-tutorial/blob/master/model.py
    
    """
        *Vanilla LSTM*
        Implementation of Vanilla LSTM architecture
    """
    
    def __init__(self, dataset, lstm_size = 128, num_layers = 3):
        """
            :param dataset: dataset with parsed args
            :param lstm_size: number of cells 
            :param num_layers: number of layers
        """
        super(Lstm_model, self).__init__() 
        self.lstm_size = lstm_size # number of expected features in the input x
        self.embedding_dim = lstm_size # size of each embedding vector
        self.num_layers = num_layers # number of stacked LSTM layers
        n_vocab = len(dataset.unique_words) 
        
        # Embedding layer converts word indexes to word vectors.
        
        self.embedding = nn.Embedding(
            num_embeddings=n_vocab, 
            embedding_dim=self.embedding_dim, 
        ) 
        
        self.lstm = nn.LSTM(
            input_size=self.lstm_size, 
            hidden_size=self.lstm_size, 
            num_layers=self.num_layers, 
        )
        
        self.fc = nn.Linear(self.lstm_size, n_vocab) 
    
    def forward(self, x, prev_state): 
        embed = self.embedding(x) 
        output, state = self.lstm(embed, prev_state)
        logits = self.fc(output) 
        # logits = vector of raw (non-normalized) predictions that the model generates.
        return logits, state 
            
            
    def init_state(self, sequence_length): 
    # function called at the start of every epoch to initialize the right shape of the state.
        return (torch.zeros(self.num_layers, sequence_length, self.lstm_size),
                    torch.zeros(self.num_layers, sequence_length, self.lstm_size))
        
        
class Bi_lstm_model(nn.Module): 
# SOURCE: https://towardsdatascience.com/text-generation-with-bi-lstm-in-pytorch-5fda6e7cc22c
    """
        *Bi-LSTM*
        Implementation of Bi-LSTM architecture
    """
    
    def __init__(self, dataset, lstm_size=128, num_layers=3): 
        """
            :param dataset: dataset with parsed args
            :param lstm_size: number of cells 
            :param num_layers: number of layers
        """
        super(Bi_lstm_model, self).__init__() 
        self.lstm_size = lstm_size # number of expected features in the input x
        self.embedding_dim = lstm_size # size of each embedding vector
        self.num_layers = num_layers # number of stacked LSTM layers
        
        n_vocab = len(dataset.unique_words) 
        
        # Embedding layer converts word indexes to word vectors.
        self.embedding = nn.Embedding(
            num_embeddings=n_vocab, 
            embedding_dim=self.embedding_dim, 
        ) 
        
        # Bi-LSTM
        self.bilstm = nn.LSTM( 
            input_size = self.lstm_size,
            hidden_size = self.lstm_size,
            num_layers=self.num_layers,
            bidirectional = True
        ) 
        
        self.fc = nn.Linear(self.lstm_size * 2, n_vocab) 
    
    def forward(self, x, prev_state): 
        embed = self.embedding(x) 
        output, state = self.bilstm(embed, prev_state)
        logits = self.fc(output) 
        # logits = vector of raw (non-normalized) predictions that the model generates.
        return logits, state 
    
    
    def init_state(self, sequence_length): 
    # function called at the start of every epoch to initialize the right shape of the state.
        return (torch.zeros(self.num_layers*2, sequence_length, self.lstm_size),
                torch.zeros(self.num_layers*2, sequence_length, self.lstm_size))
        


        
        