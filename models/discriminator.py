from torch import nn

def get_discriminator_block(input_dim, output_dim):
    '''
    Discriminator Block
    Function for returning a neural network of the discriminator given input and output dimensions.
    Parameters:
        input_dim: the dimension of the input vector, a scalar
        output_dim: the dimension of the output vector, a scalar
    Returns:
        a discriminator neural network layer, with a linear transformation 
          followed by an nn.LeakyReLU activation with negative slope of 0.2 
    '''
    return nn.Sequential(
        nn.Linear(input_dim, output_dim),
        nn.LeakyReLU(0.2)
    )

class Discriminator(nn.Module):
    '''
    Discriminator Class
    Values:
        in_dim: the input dimension, a scalar
        hidden_dim: the inner dimension, a scalar
    '''
    def __init__(self, in_dim=20, hidden_dim=32):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            get_discriminator_block(in_dim, hidden_dim * 4),
            get_discriminator_block(hidden_dim * 4, hidden_dim * 2),
            get_discriminator_block(hidden_dim * 2, hidden_dim),
            nn.Linear(hidden_dim,1)
        )

    def forward(self, pred):
        '''
        Function for completing a forward pass of the discriminator: Given a prediction, 
        returns a 1-dimension tensor representing fake/real.
        Parameters:
            pred: a flattened prediction tensor with dimension (in_dim = T_pred * 2)
        '''
        return self.disc(pred)
    