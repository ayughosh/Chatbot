import torch
import torch.nn as nn
from torch.nn import functional as F
import mmap
import random
import pickle
import os
import json
import argparse
import requests
from bs4 import BeautifulSoup

parser = argparse.ArgumentParser(description="This is a demonstration program")

# Here we add an argument to the parser, specifying the expected type, a help message, etc.
parser.add_argument(
    "-batch_size", type=str, required=True, help="Please provide a batch_size"
)

args = parser.parse_args()

# Now we can use the argument value in our program.
print(f"batch size: {args.batch_size}")
device = "cuda" if torch.cuda.is_available() else "cpu"

# batch_size = args.batch_size # to use the batch_size cmd arg -> python file_name.py -batch_size 32
batch_size = int(args.batch_size)
block_size = 128
max_iters = 4000
learning_rate = 3e-4
eval_iters = 500
n_embd = 384
n_head = 4
n_layer = 4
dropout = 0.2

print(device)


def scrape_wikipedia_article(url):

    response = requests.get(url)

    soup = BeautifulSoup(response.content, "html.parser")

    article_text = soup.find("div", {'class': 'mw-parser-output'})
    paragraphs = article_text.find_all('p')
    content = ' '.join([para.get_text() for para in paragraphs])
    print(content)
    return content

# Example usage

url = "https://en.wikipedia.org/wiki/Machine_learning"

article_text = scrape_wikipedia_article(url)

print(article_text)

# # Load the extracted comments
# with open('comments.json', 'r', encoding='utf-8') as f:
#     comments = json.load(f)

# # Create a single string of text
# text = ' '.join(comments)


# Create the vocabulary and encoding functions
text=""
text += " " + article_text
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(vocab_size)


string_to_int = {ch: i for i, ch in enumerate(chars)}
int_to_string = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [
    string_to_int.get(c, -1) for c in s
]  # Use .get() to handle missing characters
decode = lambda l: "".join(
    [int_to_string[i] for i in l if i != -1]
)  # Skip missing characters

"""# Filter out any characters not in the vocabulary
encoded_data = [string_to_int[c] for c in text if c in string_to_int]
data = torch.tensor(encoded_data, dtype=torch.long)

print("Data encoding complete. Length of data:", len(data))"""

data = torch.tensor(encode(text), dtype=torch.long)
print(data[:100])
print(len(data))


n = int(0.8 * len(data))
train_data = data[:n]
val_data = data[n:]


def get_batch(split):
    data = train_data if split == "train" else val_data
    assert (
        len(data) >= block_size * batch_size
    ), "Data size is smaller than the required block size."

    ix = torch.randint(len(data) - block_size, (batch_size,))
    for i in ix:
        assert i + block_size + 1 <= len(
            data
        ), f"Index {i + block_size + 1} out of bounds."

    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])

    # Ensure tensors are contiguous
    x = x.contiguous()
    y = y.contiguous()

    # Additional checks
    # print(f"Batch indices: {ix}")
    # print(f"x shape: {x.shape}, y shape: {y.shape}")  # Debugging print statement

    # Check for NaNs or infinities
    assert not torch.isnan(x).any(), "x contains NaNs"
    assert not torch.isnan(y).any(), "y contains NaNs"
    assert not torch.isinf(x).any(), "x contains infinities"
    assert not torch.isinf(y).any(), "y contains infinities"

    # Check sizes
    assert (
        x.size(0) == batch_size
    ), f"x batch size mismatch: {x.size(0)} != {batch_size}"
    assert (
        y.size(0) == batch_size
    ), f"y batch size mismatch: {y.size(0)} != {batch_size}"

    # Ensure tensors are of type float
    # x, y = x.float(), y.float()

    # Switch to CUDA for better performance
    try:
        x, y = x.to(device), y.to(device)
    except RuntimeError as e:
        print(f"CUDA error: {e}")
        raise e

    return x, y


# Add the following line to enable CUDA kernel error checking
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# Simplify transfer process
# x, y = get_batch('train')
# print('inputs:')
# print(x)
# print('targets:')
# print(y)


class Head(nn.Module):
    """one head of self-attention"""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B, T, C = x.shape
        k = self.key(x)  # (B,T,hs)
        q = self.query(x)  # (B,T,hs)
        # compute attention scores ("affinities")
        wei = (
            q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        )  # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # (B, T, T)
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x)  # (B,T,hs)
        out = wei @ v  # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out


# [1, 0, 0]
# [1, 0.6, 0]
# [1, 0.6, 0.4]


class MultiHeadAttention(nn.Module):
    """multiple heads of self-attention in parallel"""

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat(
            [h(x) for h in self.heads], dim=-1
        )  # (B, T, F) -> (B, T, [h1, h1, h1, h1, h2, h2, h2, h2, h3, h3, h3, h3])
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    """a simple linear layer followed by a non-linearity"""

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        y = self.sa(x)
        x = self.ln1(x + y)
        y = self.ffwd(x)
        x = self.ln2(x + y)
        return x


# Normalization: Takes place in the __init__ method with self.ln_f = nn.LayerNorm(n_embd), and is applied at the end of the blocks in the transformer architecture.
class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(
            vocab_size, n_embd
        )  # Maps each token in the vocabulary to a dense vector of size n_embd.
        self.position_embedding_table = nn.Embedding(
            block_size, n_embd
        )  # Maps each position in the input sequence to a dense vector of size n_embd
        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head=n_head) for _ in range(n_layer)]
        )  # A sequence of transformer blocks (Block),
        # each consisting of multi-head self-attention and feed-forward layers

        self.ln_f = nn.LayerNorm(
            n_embd
        )  # final layer norm,Applies layer normalization to stabilize and speed up training.
        self.lm_head = nn.Linear(
            n_embd, vocab_size
        )  # language modelling head, doing the final linear transformation,
        # A linear layer that maps the final hidden states to the vocabulary size, used to predict the next token.

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self, index, targets=None
    ):  # Computes the output of the model given an input sequence of token indices. If targets are provided, it also computes the loss.
        # logits=self.token_embedding_table(index)
        # print(f"index shape: {index.shape}")  # Debugging print statement
        B, T = index.shape
        index = index.long()

        # Converts the input token indices (index) into embedding vectors using the token embedding table

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(
            index
        )  # Embeds the token indices (idx), converting them into corresponding embedding vectors.
        pos_emb = self.position_embedding_table(
            torch.arange(T, device=device)
        )  # (T,C) Generates positional encodings for each position in the sequence, ensuring the model can understand the order of tokens.
        x = (
            tok_emb + pos_emb
        )  # (B,T,C)Adds the token embeddings and positional embeddings element-wise to create a combined representation of tokens with their respective positions.
        x = self.blocks(
            x
        )  # (B, T, C)Passes the combined embeddings through the transformer blocks,
        # applying attention mechanisms and other transformations to capture relationships between tokens
        x = self.ln_f(
            x
        )  # (B,T,C)Applies layer normalization to the transformed embeddings, helping stabilize and improve the training process.
        logits = self.lm_head(x)  # (B,T, vocab_size)
        # Passes the normalized embeddings through the final layer (language model head) to produce the logits, which represent the model's raw output before applying any activation functions.
        if targets is None:
            loss = None
        else:
            B, T, C = (
                logits.shape
            )  # If targets are provided, reshapes the logits and targets tensors, and computes the cross-entropy loss.
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    # Decoding: The generate method is responsible for decoding, as it generates new token indices based on the model's predictions.
    def generate(
        self, index, max_new_tokens
    ):  # Generates a sequence of tokens given an initial input context by sampling from the model's predictions.
        # index is (B,T) array of indices in the current context
        for _ in range(max_new_tokens):
            # get the predictions
            logits, loss = self.forward(index)
            # focus only on the last time step
            logits = logits[:, -1, :]  # (B,C)   # focus only on the last time step
            # apply softmax to get probabilities
            # Softmax: Implemented in the generate method using F.softmax(logits, dim=-1) to convert logits to probabilities.
            probs = F.softmax(
                logits, dim=-1
            )  # apply softmax to get probabilities, converts logits to probabilities
            # sample from distribution
            index_next = torch.multinomial(
                probs, num_samples=1
            )  # (B,1) )  # sample the next token index from probability distribution
            # append sampled index to the running sequence
            index = torch.cat(
                (index, index_next), dim=1
            )  # (B, T+1) # append sampled index to the running sequence
        return index


model = GPTLanguageModel(vocab_size)
# print("loading model parameters..")
# Deserialize the model
# with open('model-01.pkl', 'rb') as f:
#    model = pickle.load(f)
# print('model loaded')
m = model.to(device)


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    print(iter)
    if iter % eval_iters == 0:
        losses = estimate_loss()
        print(
            f"step: {iter}, train loss: {losses['train']:.3f}, val loss: {losses['val']:.3f}"
        )

    # sample a batch of data
    xb, yb = get_batch("train")

    # evaluate the loss
    logits, loss = model.forward(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
print(loss.item())


# Assuming `model` is your trained model
with open("model-01.pkl", "wb") as f:
    pickle.dump(model, f)
print("model saved")
