# https://github.com/bebound/textcnn/blob/master/textcnn.ipynb
import os
import time

from torchtext import data, datasets
import torch
import torch.nn as nn
import torch.nn.functional as F

TEXT = data.Field(lower=True, batch_first=True)
LABEL = data.LabelField()

# make splits for data
train, val, test = datasets.SST.splits(TEXT, LABEL, './data/', fine_grained=True)

# TEXT.build_vocab(train, vectors="fasttext.en.300d")
TEXT.build_vocab(train, vectors="glove.6B.300d")
LABEL.build_vocab(train, val, test)

print('len(TEXT.vocab)', len(TEXT.vocab))
print(LABEL.vocab.itos)
print(LABEL.vocab.stoi)
print('len(LABEL.vocab)', len(LABEL.vocab))  # vocab include ''
print('TEXT.vocab.vectors.size()', TEXT.vocab.vectors.size())


class TextCNN(nn.Module):
    def __init__(self, args):
        super().__init__()
        dim = args['dim']
        n_class = args['n_class']
        embedding_matrix = args['embedding_matrix']
        kernels = [3, 4, 5]
        kernel_number = [100, 100, 100]
        self.static_embed = nn.Embedding.from_pretrained(embedding_matrix)
        self.non_static_embed = nn.Embedding.from_pretrained(embedding_matrix, freeze=False)
        self.convs = nn.ModuleList([nn.Conv2d(2, number, (size, dim), padding=(size - 1, 0)) for (size, number) in
                                    zip(kernels, kernel_number)])
        self.dropout = nn.Dropout()
        self.out = nn.Linear(sum(kernel_number), n_class)

    def forward(self, x):
        non_static_input = self.non_static_embed(x)
        static_input = self.static_embed(x)
        x = torch.stack([non_static_input, static_input], dim=1)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x, 1)
        x = self.dropout(x)
        x = self.out(x)
        return x


args = {}
args['vocb_size'] = len(TEXT.vocab)
args['dim'] = 300
args['n_class'] = len(LABEL.vocab)
args['embedding_matrix'] = TEXT.vocab.vectors
args['lr'] = 1e-5
args['batch_size'] = 256
args['epochs'] = 2
args['log_interval'] = 1
args['test_interval'] = 5
args['save_dir'] = './saved/'

print(args['vocb_size'])
print(args['n_class'])

train_iter, val_iter, test_iter = data.BucketIterator.splits(
    (train, val, test), batch_sizes=(args["batch_size"], args["batch_size"], args["batch_size"]), shuffle=True)


def save(m):
    if not os.path.isdir(args['save_dir']):
        os.makedirs(args['save_dir'])
    save_path = os.path.join(args['save_dir'], 'text_cnn.pt')
    torch.save(m, save_path)
    print("saved model.")


model = TextCNN(args)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'])
criterion = nn.CrossEntropyLoss()

best_acc = 0
last_step = 0
model.train()
steps = 0


def eval(data_iter, model):
    model.eval()
    corrects, avg_loss = 0, 0
    for i, data in enumerate(data_iter):
        x, target = data.text, data.label
        start = time.time()
        x = x.to(device)
        target = target.to(device)
        logit = model(x)
        print(len(x), time.time() - start)
        loss = F.cross_entropy(logit, target, reduction='sum')
        avg_loss += loss.item()
        corrects += (torch.max(logit, 1)
                     [1].view(target.size()).data == target.data).sum()

    size = len(data_iter.dataset)
    avg_loss /= size
    accuracy = 100.0 * int(corrects) / size
    print('\nEvaluation - loss: {:.6f}  acc: {:.4f}%({}/{}) \n'.format(avg_loss,
                                                                       accuracy,
                                                                       corrects,
                                                                       size))
    model.train()
    return accuracy


# for epoch in range(1, args['epochs'] + 1):
#     print("train epoch ", epoch)
#     for i, data in enumerate(train_iter):
#         steps += 1
#         x, target = data.text, data.label
#         x = x.to(device)
#         target = target.to(device)
#         optimizer.zero_grad()
#         output = model(x)
#         loss = criterion(output, target)
#         loss.backward()
#         optimizer.step()
# save(model)
#
# print("finish training.")
best_model = torch.load(os.path.join(args['save_dir'], 'text_cnn.pt'))
best_model.eval()
for _ in range(1000):
    eval(test_iter, best_model)
