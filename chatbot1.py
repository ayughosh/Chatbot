import pickle
import torch
import torch.nn as nn
from torch.nn import functional as F
from bs4 import BeautifulSoup
import requests

block_size = 128
max_iters = 10000
learning_rate = 3e-5
eval_iters = 500
n_embd = 384
n_head = 4
n_layer = 4
dropout = 0.2

# Set the device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

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
)

# Define the GPTLanguageModel class (simplified version)
class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size, n_embd=384, n_head=4, n_layer=4, block_size=128, dropout=0.2):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, index, targets=None):
        B, T = index.shape
        index = index.long()
        if T > block_size:
            index = index[:, -block_size:]
        tok_emb = self.token_embedding_table(index)
        pos_emb = self.position_embedding_table(torch.arange(T, device=index.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, index, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, loss = self.forward(index)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            index_next = torch.multinomial(probs, num_samples=1)
            index = torch.cat((index, index_next), dim=1)
        return index

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
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

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out

class FeedForward(nn.Module):
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

# Load the pre-trained model
with open('model-01.pkl', 'rb') as file:
    model = pickle.load(file)
model.to(device)

def get_response(user_input):
    # Encode the user input
    encoded_input = torch.tensor([encode(user_input)], dtype=torch.long).to(device)
    
    # Generate a response
    generated_indices = model.generate(encoded_input, max_new_tokens=50)
    
    # Decode the generated indices to get the response
    response = decode(generated_indices[0].tolist())
    
    return response
def main():
    print("Welcome to the Terminal Chatbot!")
    print("Type 'exit' to end the conversation.\n")

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Chatbot: Goodbye!")
            break

        response = get_response(user_input)
        print(f"Chatbot: {response}")

if __name__ == "__main__":
    main()
