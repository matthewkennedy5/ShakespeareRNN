import pickle

vocab = []

with open('shakespeare.txt') as f:
    text = f.read()
    for ch in text:
        if ch not in vocab:
            vocab.append(ch)

pickle.dump(vocab, open('vocab.pkl', 'wb'))