"""Configuration files

"""

class Config():
    def __init__(self):
        self.epochs = 200
        self.epochs_g = 50
        self.epochs_d = 100
        self.batch_size = 64
        self.eta_G = 1e-3   # Learning Rate for generator
        self.eta_D = 1e-3   # Learning Rate for discriminator
        self.dir_path = "./data/"
        self.emb_dim = 5
        self.weight_decay = 1e-5
        self.weight_decay_g = 1e-5
        self.weight_decay_d = 1e-5
        self.patience = 300
        self.device = "cuda:0"
        
class BprConfig():
    def __init__(self):
        self.epochs = 200
        self.batch_size = 64
        self.eta = 1e-3   
        self.dir_path = "./data/"
        self.emb_dim = 5
        self.device = "cuda:0"
        self.weight_decay = 1e-5
bpr_config = BprConfig()
irgan_config = Config()        