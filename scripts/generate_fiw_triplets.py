import torch
import os
import argparse
import pandas
import numpy as np
import glob
from PIL import Image
import itertools
import pickle
import pandas as pd
def main(args):
    print(args)

    # load model
    #model.eval()

    #load dataset
    father_son = pandas.read_csv(args.fs_path)
    father_daughter = pandas.read_csv(args.fd_path)
    mother_son = pandas.read_csv(args.ms_path)
    mother_daughter = pandas.read_csv(args.md_path)

    triplets = []

    for (father, son) in father_son.to_numpy():
        print(father, son)
        mother_idx = np.where(mother_son.to_numpy() == son)[0]
        print(f'\t {mother_idx}')
        if len(mother_idx) == 1:
            print('here')
            mother = mother_son.to_numpy()[mother_idx[0]][0]
            
            triplets.append([father,mother,son])
    for (father, daughter) in father_daughter.to_numpy():
        print(father, daughter)
        mother_idx = np.where(mother_daughter.to_numpy() == daughter)[0]
        print(f'\t {mother_idx}')
        if len(mother_idx) == 1:
            print('here')
            mother = mother_daughter.to_numpy()[mother_idx[0]][0]
            
            triplets.append([father,mother,daughter])

    triplet_dirs = np.stack(triplets)

    triples = []
    
    # for row in triplet_dirs:
    #   concate permutations
    def load_image(path):
        img = Image.open(path)
        return np.array(img)

    true_triplets = []
    print(len(triplet_dirs))
    i = 0
    for row in triplet_dirs:
        i+=1 
        print(i)
        all_images = []
        for indiv in row:
            
            paths = glob.glob((args.fiw_root + '/' + indiv + '/*'))
            indiv_images = []
            # #   open all files
            # for path in paths:
            #     img = Image.open(path)
            #     indiv_images.append(np.array(img))
            all_images.append(paths)
        true_triplets += list(itertools.product(*all_images))
    pd.DataFrame(true_triplets).to_csv("triplets.csv")
    

    

    #import pdb; pdb.set_trace()
    



        
        
    
    
	

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Inference with StyleDNA')

    parser.add_argument('--device', help='Device to be used by the model (default=cuda:0)', default="cuda")
    parser.add_argument('--fs_path', help='csv file where father son relationships are stored', default=r"./recognizing-faces-in-the-wild/test-public-lists/fs.csv")
    parser.add_argument('--fd_path', help='csv file where father daughter relationships are stored', default=r"./recognizing-faces-in-the-wild/test-public-lists/fd.csv")
    parser.add_argument('--ms_path', help='csv file where mother son relationships are stored', default=r"./recognizing-faces-in-the-wild/test-public-lists/ms.csv")
    parser.add_argument('--md_path', help='csv file where mother daughter relationships are stored', default=r"./recognizing-faces-in-the-wild/test-public-lists/md.csv")
    parser.add_argument('--fiw_root', help='root directory of families in the while training set.', default=r"./recognizing-faces-in-the-wild/train-faces")
    args = parser.parse_args()

    ### RUN
    main(args)