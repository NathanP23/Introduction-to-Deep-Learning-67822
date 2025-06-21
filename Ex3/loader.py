import torch
# CODE_CHANGE: removed the following line to support newest python versions.
"""################################################
from torchtext.legacy.data import Field
################################################"""

import torchtext as tx
from torchtext.vocab import GloVe
from torchtext.datasets import IMDB
from torchtext.data.utils import get_tokenizer
import re
from torch.utils.data import DataLoader
from torchtext.data.functional import to_map_style_dataset
import pandas as pd


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_LENGTH = 100
embedding_size = 100
Train_size=30000


def review_clean(text):
    text = re.sub(r'[^A-Za-z]+', ' ', text)  # remove non alphabetic character
    text = re.sub(r'https?:/\/\S+', ' ', text)  # remove links
    text = re.sub(r"\s+[a-zA-Z]\s+", ' ', text)  # remove singale char
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def tokinize(s):
    s = review_clean(s).lower()
    splited = s.split()
    return splited[:MAX_LENGTH]


def load_data_set(load_my_reviews=False):
    data=pd.read_csv("IMDB Dataset.csv")
    train_data=data[:Train_size]
    train_iter=ReviewDataset(train_data["review"],train_data["sentiment"])
    test_data=data[Train_size:]
    if load_my_reviews:
        # CODE_CHANGE: Return ONLY custom reviews, not mixed with test data
        ################################################
        my_data = pd.DataFrame({"review": my_test_texts, "sentiment": my_test_labels})
        test_iter = ReviewDataset(my_data["review"], my_data["sentiment"])
        return train_iter, test_iter
        ################################################"""
    test_data=test_data.reset_index(drop=True)
    test_iter=ReviewDataset(test_data["review"],test_data["sentiment"])
    return train_iter, test_iter


embadding = GloVe(name='6B', dim=embedding_size)
tokenizer = get_tokenizer(tokenizer=tokinize)


def preprocess_review(s):
    cleaned = tokinize(s)
    embadded = embadding.get_vecs_by_tokens(cleaned)
    if embadded.shape[0] != 100 or embadded.shape[1] != 100:
        embadded = torch.nn.functional.pad(embadded, (0, 0, 0, MAX_LENGTH - embadded.shape[0]))
    return torch.unsqueeze(embadded, 0)

def preprocess_label(label):
    return [0.0, 1.0] if label == "negative" else [1.0, 0.0]



def collact_batch(batch):
    label_list = []
    review_list = []
    embadding_list=[]
    for  review,label in batch:
        label_list.append(preprocess_label(label))### label
        review_list.append(tokinize(review))### the  actuall review
        processed_review = preprocess_review(review).detach()
        embadding_list.append(processed_review) ### the embedding vectors
    
    label_list = torch.tensor(label_list, dtype=torch.float32).reshape((-1, 2))

    embadding_tensor= torch.cat(embadding_list)
    return label_list.to(device), embadding_tensor.to(device) ,review_list


##########################
# ADD YOUR OWN TEST TEXT #
##########################

my_test_texts = [
    # Professor's required examples (must include)
    "this movie is very very bad the worst movie",
    "this movie is so great",
    "I really liked the fish and animations the anther casting was not so good",
    
    # Our additions for RNN vs GRU comparison
    "terrible awful boring start but wait the ending completely redeems everything making this a masterpiece worth watching",
    "amazing brilliant perfect beginning however the rest was completely unwatchable terrible waste of potential",
    "started slow but the ending was incredible and moving",
    "worst film ever made complete waste of time and money",
    "brilliant masterpiece with outstanding performances",
    
    # Additional examples testing long-range dependencies
    "boring boring boring boring boring but then suddenly it becomes absolutely phenomenal best movie ever definitely watch this",
    "masterpiece masterpiece masterpiece until the last act which ruins everything terrible ending worst conclusion ever seen",
    
    # Testing "despite/although" understanding
    "despite numerous flaws including bad acting poor script terrible direction this somehow works becoming genuinely entertaining and memorable",
    "although it starts promising with great cinematography excellent cast the film ultimately fails delivering nothing but disappointment",
    
    # Testing gradual sentiment shifts
    "starts okay becomes better then good then great finally reaching absolutely magnificent levels pure cinematic perfection achieved",
    "begins wonderfully gradually loses steam becomes mediocre then bad finally ending as complete disaster avoid this mess",
    
    # Testing conflicting signals throughout
    "great acting terrible script beautiful cinematography awful dialogue stunning visuals horrible plot overall still worth watching though",
    "horrible effects amazing story terrible pacing brilliant characters awful editing great soundtrack overall disappointing waste unfortunately",
    
    # Testing context-dependent words
    "not bad actually pretty good in fact very good honestly great film definitely not disappointing at all",
    "not terrible but not good either not recommended definitely not worth your time not even slightly enjoyable"
]

my_test_labels = [
    # Professor's labels
    "negative",  # very very bad
    "positive",  # so great
    "positive",  # liked despite casting
    
    # Our original labels
    "positive",  # terrible→masterpiece
    "negative",  # amazing→unwatchable
    "positive",  # slow→incredible
    "negative",  # worst film
    "positive",  # brilliant
    
    # New labels
    "positive",  # boring×5 → phenomenal (tests if RNN forgets early boring)
    "negative",  # masterpiece×3 → ruins everything (tests if RNN stuck on early positive)
    
    "positive",  # despite flaws → entertaining (tests understanding of "despite")
    "negative",  # although promising → fails (tests understanding of "although")
    
    "positive",  # gradual improvement → perfection (tests tracking gradual change)
    "negative",  # gradual decline → disaster (tests tracking gradual change)
    
    "positive",  # mixed signals → worth watching (tests final integration)
    "negative",  # mixed signals → disappointing (tests final integration)
    
    "positive",  # multiple "not" → actually positive (tests context understanding)
    "negative"   # multiple "not" → actually negative (tests context understanding)
]

################################################


class ReviewDataset(torch.utils.data.Dataset):
    def __init__(self, review_list, labels):
        'Initialization'
        self.labels = labels
        self.reviews = review_list

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, index):
        X = self.reviews[index]
        y = self.labels[index]
        return X, y



def get_data_set(batch_size, toy=False):
    train_data, test_data = load_data_set(load_my_reviews=toy)

    # CODE_CHANGE: Adjust batch size for custom reviews
    ################################################
    if toy:
        # For custom reviews, use batch size that gets all reviews at once
        test_batch_size = len(my_test_texts)
        shuffle_test = False  # Don't shuffle custom reviews
    else:
        test_batch_size = batch_size
        shuffle_test = True
    ################################################
    train_dataloader = DataLoader(train_data, batch_size=batch_size,
                                  shuffle=True, collate_fn=collact_batch)
    test_dataloader = DataLoader(test_data, batch_size=test_batch_size,
                                 shuffle=shuffle_test, collate_fn=collact_batch)
    return train_dataloader, test_dataloader, MAX_LENGTH, embedding_size