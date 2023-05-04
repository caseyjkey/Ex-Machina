import torch
import os
import argparse

from torchvision.utils import save_image
from model.model import pSp, condi
from model.DNAnet import DNAnet,VAE
from utils.utils import align_face, totensor
from utils.FiW_Dataset import FiW_Dataset
import torch.nn as nn
import PIL
from model.model_irse import Backbone
from model.model_arcface import *
from torchvision.transforms import Resize

def main(args):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"using device: {device}")

    print("Loading models...")
    # Load model weights
    vae = VAE(512,device=device)
    beta = vae.beta
    # DNA.load_state_dict(torch.load(args.dna), strict=True)

    mapper = nn.DataParallel(condi()).to(device)
    mapper.load_state_dict(torch.load(args.map), strict=True)
    mapper.eval()
    mapper.requires_grad_(False)

    net = nn.DataParallel(pSp(3, args.enc, args.gan)).to(device)
    net.eval()
    net.requires_grad_(False)

    arcface = Backbone(num_layers=50,drop_ratio=0.6,mode="ir_se").to(device)
    arcface.load_state_dict(torch.load(args.irse50))
    arcface.eval()
    arcface.requires_grad_(False)

    batch_size = 4
    num_epochs = 1
    training_data = FiW_Dataset(args.trip,args.fiw, align_faces=True,device=device)
    train_dataloader = torch.utils.data.DataLoader(training_data, batch_size=batch_size, shuffle=True)

    l2_w = 1
    l_id_w = 2
    optimizer = torch.optim.Adam(vae.parameters(), 1e-5)
    CoSim = torch.nn.CosineSimilarity()
    mse = torch.nn.MSELoss()
    resize = torch.nn.functional.interpolate

    E = net.module.encoder # encoder

    def D(W, age, gender):
        cW_hat_expand = W.repeat(18, 1, 1).permute(1, 0, 2)
        cW_hat_delta = mapper(cW_hat_expand, age,  gender)
        cImg_hat = net(cW_hat_expand + cW_hat_delta)
        return cImg_hat
    
    
    for epoch in range(num_epochs):
        running_loss = 0.
        last_loss = 0.
        for i, triplet in enumerate(train_dataloader):           
            optimizer.zero_grad()

            triplet_cat = torch.cat(triplet,dim=0)

            with torch.no_grad():
                dW, mW, cW = E(triplet_cat).split(batch_size)

            dW = dW.clone().detach().requires_grad_(True)
            mW = mW.clone().detach().requires_grad_(True)
            cW = cW.clone().detach().requires_grad_(True)

            # gather age and gender random samples
            age = torch.randint(0,100,size=[batch_size,1]).to(device)
            gender = torch.bernoulli(torch.ones(batch_size,1)*0.5).to(device)
            
            # concat parent encodings for fusion
            pW = torch.cat((dW,mW),dim=1)

            # forward pass of Vae and stylegan (D)
            outputs = vae(pW)
            cW_hat = outputs['rec']
            cImg_hat = D(cW_hat,age,gender)

            loss, losses = vae.loss(cW,outputs)

            rec_loss = losses['rec_loss']
            kl_loss =  losses['kl_loss'].mean(dim=1)

            #RI_ori = arcface(resize(triplet[2], (112,112)))
            #RI_gen = arcface(resize(cImg_hat, (112,112)))

            #l_id = 1 - CoSim(RI_ori,RI_gen)
            l_id = 0

            loss = l2_w*rec_loss + l_id_w*l_id + beta*kl_loss
            loss = loss.mean()
            loss.backward()

            optimizer.step()
                         
 
            # Gather data and report
            running_loss += loss.item()
            if (i+1) % 5 == 0:
                last_loss = running_loss / 5 # loss per batch
                print('epoch {} batch {} loss: {}'.format(epoch+1, i + 1, last_loss))
                running_loss = 0.
            if (i+1) % 200 == 0:
                torch.save(vae.state_dict(),f"saved_models/vae_{i+1}.pth")

            
        


            

        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Inference with StyleDNA')

    # Image Path and Saving Path
    parser.add_argument('-m', '--mom_path', help='Path of mom image', default = r'C:\Users\tjtom\OneDrive\Desktop\File_Cabinet\Code\Ex-Machina-GAN\mom.png')
    parser.add_argument('-d', '--dad_path', help='Path of dad image')
    parser.add_argument('-a', '--target_gender', help='Predicted child age', default= 'male')
    parser.add_argument('-g', '--target_age', help='Predicted child gender', default = 15)
    parser.add_argument('-o', '--outputs', help='Output directory', default='./result')

    # Model weights Path
    parser.add_argument('-E', '--enc', help='Path of pretrained encoder model', default='./pretrained_model/enc_4_2.pth')
    parser.add_argument('-M', '--map', help='Path of pretrained mapper model', default='./pretrained_model/condi_4_2.pth')
    parser.add_argument('-G', '--gan', help='Path of pretrained stylegan2 model', default='./pretrained_model/stylegan2-ffhq-config-f.pt')
    parser.add_argument('-D', '--dna', help='Path of pretrained DNA model', default='./pretrained_model/10.pth')
    parser.add_argument('-IR', '--irse50', help='IRSE50 pretrained model weights', default='./pretrained_model/model_ir_se50.pth')

    parser.add_argument('-T', '--trip', help='Path of FiW triplets', default='triplets.csv')
    parser.add_argument('-F', '--fiw', help='Path of FiW Data', default='')

    # Device
    parser.add_argument('--device', help='Device to be used by the model (default=cuda:0)', default="cuda")
    args = parser.parse_args()

    ### RUN
    main(args)