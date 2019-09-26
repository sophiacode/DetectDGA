import random
import time
from tqdm import tqdm

import numpy as np
import pandas as pd

class CharBot(object):
    def __init__(self, seed=None):
        if seed is not None:
            self.seed = seed
        else:
            self.seed = (int)(time.time())
    
    def generate(self, dga_num, save_path, benign_num=10000, min_len=6):
        print("Loading Alexa top 1 million domains...")
        
        """ Load and preprocess """
        alexa_path = "./dataset/CharBot/Alexa.csv"
        # domains = list(pd.read_csv("./dataset/original/Alexa-top-1m.csv", header=None)[1])
        # for i in range(len(domains)):
        #     domains[i] = domains[i].split(".")[0]
    
        # domains = np.asarray(domains)
        # lens = np.array([len(d) for d in domains])
        # idx = np.argwhere(lens >= 6)
        # domains = domains[idx]
        
        # df = pd.DataFrame(domains)
        # df.to_csv(alexa_path, header=None, index=None)
        
        domains = list(pd.read_csv(alexa_path, header=None)[0])
        benign_num = min(benign_num, 1000000)
        domains = random.sample(domains, benign_num)

        """ Generate """
        dgas = list()
        valid_chars = list("abcdefghijklmnopqrstuvwxyz1234567890-")
        for i in tqdm(range(dga_num)):
            random.shuffle(domains)
            d = list(random.choice(domains))
            i, j = random.sample(range(1, len(d)), 2)
            d[i] = random.choice(valid_chars)
            d[j] = random.choice(valid_chars)
            dgas.append(''.join(d))
      
        df = pd.DataFrame(dgas)
        df.to_csv(save_path, index=None, header=None)


if __name__ == "__main__":
    CharBot().generate(100000, "./dataset/CharBot.csv")
        




        
    
