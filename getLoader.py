""" Goal: We load the dataset and preprocess for caption generator model. Convert text -> numerical values
    1. Build a Vocabulary mapping each word to a index
    2. Setup a Pytorch dataset to load the data
    3. Setup padding of every batch (all samples should be equal same seq_len and setup dataloader)
"""
# Import relevant libraries
import os # when loading file paths
import pandas as pd # for looking up in annotation file
import spacy # for tokenizer
import torch
from torch.nn.utils.rnn import pad_sequence # pad batch
from torch.utils.data import DataLoader,Dataset # for loading dataset
from PIL import Image
import torchvision.transforms as transforms


# Initialize the english loader which can separate words in a sentence
spacy_eng = spacy.load("en_core_web_sm")


# 1. Build a Vocabulary mapping each word to a index
class Vocabulary:
    def __init__(self,freq_threshold):
        self.itos = {0:"<PAD>", 1:"<SOS>", 2:"<EOS>",3:"<UNK>"}
        self.stoi = {"<PAD>":0, "<SOS>":1, "<EOS>":2, "<UNK>":3}
        self.freq_threshold = freq_threshold
    
    def __len__(self):
        return len(self.itos)
    
    @staticmethod
    def tokenizer_eng(text):
        return [tok.text.lower() for tok in spacy_eng.tokenizer(text)]
    
    def build_vocabulary(self,sentence_list):
        frequencies = {}
        idx = 4
        
        for sentence in sentence_list:
            for word in self.tokenizer_eng(sentence):
                if word not in frequencies:
                    frequencies[word]=1
                else:
                    frequencies[word] += 1
                
                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1
                    
    def numericalize(self,text):
        tokenized_text = self.tokenizer_eng(text)
        
        word_after_numericalized = []
        for token in tokenized_text:
            if token in self.stoi:
                word_after_numericalized.append(self.stoi[token])
            else:
                word_after_numericalized.append(self.stoi["<UNK>"])
                
        return word_after_numericalized


# 2. Setup a Pytorch dataset to load dataset following our structure
class FlickrDataset(Dataset):
    def __init__(self, root_dir, captions_file, transform=None,freq_threshold=5):
        self.root_dir = root_dir
        self.df_captions = pd.read_csv(captions_file)
        self.transform = transform

        # Get image file name, captions
        self.img_name = self.df_captions['image']
        self.captions = self.df_captions['caption']
        
        # Initialize vocabulary and build vocab
        self.vocab = Vocabulary(freq_threshold)
        self.vocab.build_vocabulary(self.captions.tolist())
        
    def __len__(self):
        return len(self.df_captions)
    
    def __getitem__(self, index):
        caption = self.captions[index]
        img_id = self.img_name[index]
        img = Image.open(os.path.join(self.root_dir,img_id)).convert("RGB")
        
        if self.transform is not None:
            img = self.transform(img)
            
        numericalized_caption = [self.vocab.stoi["<SOS>"]]
        numericalized_caption += self.vocab.numericalize(caption)
        numericalized_caption.append(self.vocab.stoi["<EOS>"])
        
        return img, torch.tensor(numericalized_caption)


# 3. Setup padding of every batch (all samples should be equal same seq_len)
class MyCollate:
    def __init__(self,pad_idx):
        self.pad_idx = pad_idx
    
    def __call__(self,batch):
        imgs = [item[0].unsqueeze(0) for item in batch]
        imgs = torch.cat(imgs,dim=0)
        targets = [item[1] for item in batch]
        targets = pad_sequence(targets,batch_first=False,padding_value=self.pad_idx)

        return imgs, targets


# 4. Create function to we load data automatically when we use in another file
def get_loader(root_folder,annotation_file, transform_method,
               batch_size = 32, num_workers = 8, shuffle = True, pin_memory = True):
    
    dataset = FlickrDataset(root_folder,annotation_file,transform=transform_method)
    
    pad_idx = dataset.vocab.stoi["<PAD>"]
    
    loader = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=num_workers,
                        shuffle=shuffle, pin_memory=pin_memory, collate_fn= MyCollate(pad_idx=pad_idx))
    
    return loader


def main():
    # Declare the transform method we wanna use with our images
    my_transform_method = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
    ])
    
    my_dataloader = get_loader(root_folder="Images",annotation_file="captions.txt",transform_method=my_transform_method)
    
    for idx, (imgs,captions) in enumerate(my_dataloader):
        print(imgs.shape)
        print(captions.shape)
        
        
if __name__ == "__main__":
    main()