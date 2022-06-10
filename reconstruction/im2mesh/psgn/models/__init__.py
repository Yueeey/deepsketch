import torch.nn as nn
from im2mesh.psgn.models.decoder import Decoder
from im2mesh.psgn.models.psgn_2branch import PCGN_2Branch
import torch.nn.functional as F

decoder_dict = {
    'simple': Decoder,
    'psgn_2branch': PCGN_2Branch
}


class PCGN(nn.Module):
    r''' The Point Set Generation Network.

    For the PSGN, the input image is first passed to a encoder network,
    e.g. restnet-18 or the CNN proposed in the original publication. Next,
    this latent code is then used as the input for the decoder network, e.g.
    the 2-Branch model from the PSGN paper.

    Args:
        decoder (nn.Module): The decoder network
        encoder (nn.Module): The encoder network
    '''

    def __init__(self, decoder, encoder):
        super().__init__()
        self.decoder = decoder
        self.encoder = encoder

    def forward(self, x):
        latent = self.encoder(x)
        points = self.decoder(latent)
        c = F.normalize(latent)
        return points, c

    # def forward(self, x):
    #     latent = self.encoder(x)
    #     points = self.decoder(latent)
    #     c = F.normalize(latent)
    #     return points, c, latent

    def bias(self, c):
        points = self.decoder(c)
        return points
