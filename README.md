# Chatbot
Created a chatbot by implementing a GPT model and fine tuning it to my specific needs.
Chatbot uses data from wikipedia
Created another GPT model which uses data from Reddit
Used post normalization instead of Pre normalization.(GPT-3 uses pre normalization)
Pre normalization is Normalizing data before it is fed into the model, it ensures that all features contribute equally to the learning process and prevents any single feature from dominating due to its scale.
Post normalization is Normalizing data after certain operations or transformations have been performed, it is Useful for ensuring that outputs of certain layers or steps in a process are standardized, particularly in iterative processes or neural network layers.
Post normalization is used for deep learning whereas Pre normalization is used for Traditional Machine Learning model.
Used Learnable encodings instead of Sinocodial encodings(used in transformers)

Research Papers:
Attention is All You Need - https://arxiv.org/pdf/1706.03762.pdf

A Survey of LLMs - https://arxiv.org/pdf/2303.18223.pdf

QLoRA: Efficient Finetuning of Quantized LLMs - https://arxiv.org/pdf/2305.14314.pdf

