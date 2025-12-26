import os
import re
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from transformers import DistilBertTokenizer, DistilBertModel
from tqdm import tqdm
import amrlib
import spacy
torch.manual_seed(0)

device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
bert = DistilBertModel.from_pretrained('distilbert-base-uncased').to(device)
stog = amrlib.load_stog_model()
nlp = spacy.load("en_core_web_md")

EPOCHS = 10
BATCH_SIZE = 24
MAX_LENGTH = 128
LR = 1e-5
CKPT_DIR = "./ckpt"
NUM_CLASSES = 3

os.makedirs(CKPT_DIR, exist_ok=True)

POS_DIM = 16   # embedding size for POS tags
DEP_DIM = 16   # embedding size for dependency labels
AMR_DIM = 16   # embedding size for AMR tokens

def get_ud_features(sen, nlp):
    doc = nlp(sen) #get the dependencies of the sentence
    features = {
        'pos_tags': [token.pos_ for token in doc], #tags like NOUN, VERB, ADJ
        'dep_rels': [token.dep_ for token in doc], #dependency relations (how words relate to each other)
        'is_stop': [token.is_stop for token in doc], #ex: is, the, a
        'is_punct': [token.is_punct for token in doc],
        'head_distances': [token.i - token.head.i for token in doc]
    }
    return features

def create_ud_vocab(sentences, nlp):
    pos_tag_set = set()
    dep_rels_set = set()

    for sent in sentences:
        features = get_ud_features(sent, nlp)
        pos_tag_set.update(features['pos_tags']) # add pos_tag to set for each sent
        dep_rels_set.update(features['dep_rels']) # add dep_rels to set for each sent

    #sorts sets to then maps either the pos_tag or the dep_rel to a number value
    pos_to_idx = {tag: idx for idx, tag in enumerate(sorted(pos_tag_set))}
    dep_to_idx = {dep: idx for idx, dep in enumerate(sorted(dep_rels_set))}

    #padding option it is at the end of the array in case padding is needed for sentences of different lengths
    pos_to_idx['<PAD>'] = len(pos_to_idx)
    dep_to_idx['<PAD>'] = len(dep_to_idx)

    return pos_to_idx, dep_to_idx

def encode_ud_features(sentence, nlp, pos_to_idx, dep_to_idx, max_length):
    #this method encodes features as tensors
    features = get_ud_features(sentence, nlp)

    #loop through pos_tags, get tag otherwise use <PAD>
    pos_ids = []
    for tag in features['pos_tags']:
        pos_ids.append(pos_to_idx.get(tag, pos_to_idx['<PAD>']))

    #loop through dep_rels, get dep otherwise use <PAD>
    dep_ids = []
    for dep in features['dep_rels']:
        dep_ids.append(dep_to_idx.get(dep, dep_to_idx['<PAD>']))

    #normalize the head_distances incase anything is too large and distracting
    head_dists = [min(max(d, -10), 10) / 10.0 for d in features['head_distances']]

    #pad/truncate based on the max_length
    pos_ids = (pos_ids + [pos_to_idx['<PAD>']] * max_length)[:max_length]
    dep_ids = (dep_ids + [dep_to_idx['<PAD>']] * max_length)[:max_length]
    head_dists = (head_dists + [0.0] * max_length)[:max_length] #uses 0.0 for the padded head distances

    return { #reformat at tensors
        'pos_ids': torch.tensor(pos_ids, dtype=torch.long),
        'dep_ids': torch.tensor(dep_ids, dtype=torch.long),
        'head_dists': torch.tensor(head_dists, dtype=torch.float)
    }


class NN(nn.Module):
    def __init__(self, pos_vocab_size, dep_vocab_size, amr_vocab_size, pos_pad_idx, dep_pad_idx, amr_pad_idx, use_bert=True, use_ud=True, use_amr=True):
        super().__init__()

        self.use_bert = use_bert
        self.use_ud = use_ud
        self.use_amr = use_amr

        # BERT embeddings
        if use_bert:
            self.bert = bert
            bert_dim = 768
        else:
            bert_dim = 0

        # UD embeddings 
        if use_ud:
            self.pos_emb = nn.Embedding(
                num_embeddings=pos_vocab_size,
                embedding_dim=POS_DIM,
                padding_idx=pos_pad_idx,
            )

            self.dep_emb = nn.Embedding(
                num_embeddings=dep_vocab_size,
                embedding_dim=DEP_DIM,
                padding_idx=dep_pad_idx,
            )
            ud_dim = POS_DIM + DEP_DIM + 1
        else:
            ud_dim = 0
        
        # AMR embeddings
        if use_amr:
            self.amr_emb = nn.Embedding(
                num_embeddings=amr_vocab_size,
                embedding_dim=AMR_DIM,
                padding_idx=amr_pad_idx,
            )
            amr_dim = AMR_DIM
        else:
            amr_dim = 0

        in_dim = bert_dim + ud_dim + amr_dim # set total dimensions based on model inputs

        self.hidden_layers = nn.Sequential(
            nn.Linear(in_dim, 384),
            nn.ReLU(),
            nn.Linear(384, 192),
            nn.ReLU(),
            nn.Linear(192, 96),
            nn.ReLU(),
            nn.Linear(96, 48),
            nn.ReLU(),
        )
        self.flatten = nn.Flatten()
        self.classifier = nn.Linear(MAX_LENGTH * 48, NUM_CLASSES)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(
            self,
            input_ids,
            attention_mask=None,
            pos_ids=None,
            dep_ids=None,
            head_dists=None,
            amr_ids=None,
        ):
        # shapes:
        # input_ids: [batch, max_len]
        # pos_ids, dep_ids: [batch, max_len]
        # head_dists: [batch, max_len]

        features = []
        if self.use_bert:
            input_ids = input_ids.to(device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            last_hidden_state = outputs.last_hidden_state
            features.append(last_hidden_state)

        if self.use_ud:
            if pos_ids is not None:
                pos_ids = pos_ids.to(device)
            if dep_ids is not None:
                dep_ids = dep_ids.to(device)
            if head_dists is not None:
                head_dists = head_dists.to(device)

            pos_vecs = self.pos_emb(pos_ids)          # [batch, seq_len, POS_DIM]
            dep_vecs = self.dep_emb(dep_ids)          # [batch, seq_len, DEP_DIM]
            head_vecs = head_dists.unsqueeze(-1)      # [batch, seq_len, 1]
            features.extend([pos_vecs, dep_vecs, head_vecs])

        if self.use_amr:
            if amr_ids is not None:
                amr_ids = amr_ids.to(device)
            amr_vecs = self.amr_emb(amr_ids)
            features.append(amr_vecs)

        # concatenate along feature dimension
        combined = torch.cat(features, dim=2)

        processed = self.hidden_layers(combined)  # [batch, seq_len, 48]
        flattened = self.flatten(processed)       # [batch, seq_len*48]
        logits = self.classifier(flattened)       # [batch, NUM_CLASSES]

        return self.log_softmax(logits)

def load_sentences(tok, counter):
    sentences = []
    labels = []
    with open(tok, "r", encoding="utf-8") as ftok:
        for line in ftok:
            line = line.strip()
            if not line:
                continue
            sentences.append(line)
            labels.append(counter)
    return sentences, labels

def make_data(groups):
    sentences = []
    labels = []
    counter = 0
    for tok in groups:
        sents, labs = load_sentences(tok, counter)
        sentences.extend(sents)
        labels.extend(labs)
        counter += 1

    return sentences, labels

def balance_data(data, labels, seed=None, shuffle=True):
    df = pd.DataFrame({"sent": data, "label": labels})
    counts = df["label"].value_counts()
    min_count = int(counts.min())
    # truncate to lowest count
    balanced_parts = []
    for label in sorted(df["label"].unique()):
        group = df[df["label"] == label]
        if len(group) > min_count:
            sampled = group.sample(n=min_count, random_state=seed)
        else:
            sampled = group.copy()
        balanced_parts.append(sampled)

    balanced_df = pd.concat(balanced_parts, ignore_index=True)
    if shuffle:
        balanced_df = balanced_df.sample(
            frac=1, random_state=seed
        ).reset_index(drop=True)
    return balanced_df["sent"].tolist(), balanced_df["label"].tolist()

def prep_bert_data(data, max_length):
    input_ids_list = []
    attention_mask_list = []
    for text in data:
        encoded = tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )
        input_ids_list.append(encoded['input_ids'].squeeze(0))
        attention_mask_list.append(encoded['attention_mask'].squeeze(0))
    
    return torch.stack(input_ids_list), torch.stack(attention_mask_list)

def prep_ud_data(data, max_length, nlp, pos_to_idx, dep_to_idx):
  tensors = []
  for text in data:
    ud_features = encode_ud_features(text, nlp, pos_to_idx, dep_to_idx, max_length)
    tensors.append(ud_features)
  return tensors

def tokenize_amr(amr_string):

    amr_string = ' '.join(amr_string.split())

    # split on special characters
    pattern = r'(\(|\)|/|:[A-Za-z0-9-]+)'

    tokens = []
    for part in re.split(pattern, amr_string):
        part = part.strip()
        if part:
            tokens.append(part)

    return tokens

def build_amr_vocab(amr_strings, vocab_size=5000):
    token_counter = Counter()

    # collect all tokens from all AMR strings
    for amr_str in amr_strings:
        tokens = tokenize_amr(amr_str)
        token_counter.update(tokens)

    vocab = ['<PAD>', '<UNK>']

    # add most common tokens
    most_common = token_counter.most_common(vocab_size - len(vocab))
    vocab.extend([token for token, _ in most_common])

    token_to_idx = {token: idx for idx, token in enumerate(vocab)}

    return token_to_idx

def amr_to_tensor(amr_string, token_to_idx, max_length=128):

    tokens = tokenize_amr(amr_string)
    #truncate
    tokens = tokens[:max_length]

    token_ids = [
        token_to_idx.get(token, token_to_idx['<UNK>'])
        for token in tokens
    ]
    #pad
    pad_id = token_to_idx['<PAD>']
    while len(token_ids) < max_length:
        token_ids.append(pad_id)

    return torch.tensor(token_ids, dtype=torch.long)

def split_data(data, labels, test_size=0.2, seed=None):
    train_data, test_data, train_labels, test_labels = train_test_split(
        data, labels, test_size=test_size, random_state=seed, stratify=labels
    )
    return train_data, train_labels, test_data, test_labels

def get_predicted_label_from_predictions(log_probs):

    return log_probs.argmax(1).item()

def print_performance_by_class(true_labels, pred_labels):
    true_labels = np.array(true_labels)
    pred_labels = np.array(pred_labels)

    acc_list = []
    for c in range(NUM_CLASSES):
        idx = np.where(true_labels == c)[0]
        if len(idx) == 0:
            acc_list.append(0.0)
            continue
        acc = (pred_labels[idx] == true_labels[idx]).mean()
        acc_list.append(acc)

    print("Accuracy by Category:")
    names = {0: "native", 1: "nonnative", 2: "translated"}
    for i, a in enumerate(acc_list):
        label_name = names.get(i, f"Category {i}")
        print(f"{label_name} (class {i}): {a:.4f}")
    return acc_list

def train(dataloader, model,optimizer,epoch, weights):
	loss_fn = nn.NLLLoss(weight=weights)
	model.train()
	with tqdm(dataloader, unit="batch") as tbatch:
		for X, y in tbatch:
			X, y = X.to(device), y.to(device)
			# Compute prediction error
			pred = model(X)
			loss = loss_fn(pred, y)

			# Backpropagation
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
	torch.save({'epoch':epoch,
		'model_state_dict':model.state_dict(),
		'optimizer_state_dict':optimizer.state_dict(),
		'loss':loss,
		},f"{CKPT_DIR}/ckpt_{epoch}.pt")

def test(dataloader,model,dataset_name, weights):
	loss_fn = nn.NLLLoss(weight=weights)
	size = len(dataloader.dataset)
	num_batches = len(dataloader)
	model.eval()
	test_loss, correct = 0, 0
	with torch.no_grad():
		for X, y in dataloader:
			X, y = X.to(device), y.to(device)
			pred = model(X)
			test_loss += loss_fn(pred, y).item()
			correct += (pred.argmax(1) == y).type(torch.float).sum().item()
	test_loss /= num_batches
	correct /= size
	print(f"{dataset_name} Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

def train_epoch(dataloader, model, optimizer, epoch, weights):
    loss_fn = nn.NLLLoss(weight=weights)
    model.train()
    with tqdm(dataloader, unit="batch") as tbatch:
        for batch in tbatch:
            (
                input_ids,
                attention_mask,
                pos_ids,
                dep_ids,
                head_dists,
                amr_ids,
                labels,
            ) = batch

            labels = labels.to(device)

            log_probs = model(
                input_ids,
                attention_mask=attention_mask,
                pos_ids=pos_ids,
                dep_ids=dep_ids,
                head_dists=head_dists,
                amr_ids=amr_ids,
            )
            loss = loss_fn(log_probs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

def evaluate(dataloader, model, weights, name="TEST"):
    loss_fn = nn.NLLLoss(weight=weights)
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []

    with torch.no_grad():
        for batch in dataloader:
            (
                input_ids,
                attention_mask,
                pos_ids,
                dep_ids,
                head_dists,
                amr_ids,
                labels,
            ) = batch

            labels = labels.to(device)

            log_probs = model(
                input_ids,
                attention_mask=attention_mask,
                pos_ids=pos_ids,
                dep_ids=dep_ids,
                head_dists=head_dists,
                amr_ids=amr_ids,
            )
            loss = loss_fn(log_probs, labels)
            total_loss += loss.item()

            preds = log_probs.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_preds.append(preds.cpu())

    avg_loss = total_loss / len(dataloader)
    acc = correct / total
    print(f"{name} accuracy: {acc*100:.2f}%   loss: {avg_loss:.4f}")
    return torch.cat(all_preds, dim=0)

def run_experiment(config_name, use_bert, use_ud, use_amr, 
                   train_loader, test_loader, 
                   pos_vocab_size, dep_vocab_size, amr_vocab_size,
                   pos_pad_idx, dep_pad_idx, amr_pad_idx,
                   test_labels, weights, epochs=5):
    print("\n"+"="*70)
    print(f"EXPERIMENT: {config_name}")
    print("="*70)

    model = NN(
        pos_vocab_size, dep_vocab_size, amr_vocab_size,
        pos_pad_idx, dep_pad_idx, amr_pad_idx,
        use_bert=use_bert, use_ud=use_ud, use_amr=use_amr
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    best_acc = 0.0
    for epoch in range(epochs):
        print(f"\nEpoch {epoch}")
        train_epoch(train_loader, model, optimizer, epoch, weights)
        preds_test = evaluate(test_loader, model, weights, name="TEST")
        print_performance_by_class(test_labels, preds_test.numpy())
        acc = (preds_test.numpy() == test_labels).mean()
        if acc > best_acc:
            best_acc = acc
    print(f"\n{config_name} - Best Test Accuracy: {best_acc*100:.2f}%")
    class_print=print_performance_by_class(test_labels, preds_test.numpy())
    return {
        'config_name': config_name,
        'overall_accuracy': best_acc,
        'native_accuracy': class_print[0],
        'nonnative_accuracy': class_print[1],
        'translated_accuracy': class_print[2],
    }

def run_all_models(train_loader, test_loader, pos_vocab_size, dep_vocab_size, amr_vocab_size,
                   pos_pad_idx, dep_pad_idx, amr_pad_idx, test_labels, weights, epochs=5):
    configurations = [
        ("BERT only", True, False, False),
        ("UD only", False, True, False),
        ("AMR only", False, False, True),
        ("BERT + UD", True, True, False),
        ("BERT + AMR", True, False, True),
        ("UD + AMR", False, True, True),
        ("BERT + UD + AMR", True, True, True),
    ]
    all_results = []
    for config_name, use_bert, use_ud, use_amr in configurations:
        result = run_experiment(
            config_name, use_bert, use_ud, use_amr,
            train_loader, test_loader,
            pos_vocab_size, dep_vocab_size, amr_vocab_size,
            pos_pad_idx, dep_pad_idx, amr_pad_idx,
            test_labels, weights, epochs=epochs
        )
        all_results.append(result)
    
    print("\n" + "="*70)
    print("EXPERIMENT RESULTS")
    print("="*70)

    print(f"{'Configuration':<20} {'Overall Acc':<12} {'Native Acc':<12} {'Non-native Acc':<16} {'Translated Acc':<16}")
    print("-"*70)

    for result in all_results:
        print(f"{result['config_name']:<20} "
              f"{result['overall_accuracy']*100:>10.2f}%  "
              f"{result['native_accuracy']*100:>10.2f}%  "
              f"{result['nonnative_accuracy']*100:>14.2f}%  "
              f"{result['translated_accuracy']*100:>14.2f}%")

def main():
    # 0 = native, 1 = nonnative, 2 = translated
    natives_tok = "natives.tok"
    nonnatives_tok = "nonnatives.tok"
    translations_tok = "translations.tok"

    groups = [natives_tok, nonnatives_tok, translations_tok]

    all_sents, all_labels = make_data(groups)
    print("Raw class counts:", np.bincount(all_labels, minlength=NUM_CLASSES))

    all_sents, all_labels = balance_data(all_sents, all_labels, seed=42, shuffle=True)
    print("Balanced class counts:", np.bincount(all_labels, minlength=NUM_CLASSES))

    # optional extra downsampling while debugging
    # keep = 20 * NUM_CLASSES
    # all_sents = all_sents[:keep]
    # all_labels = all_labels[:keep]


    # split to train / test
    train_texts, train_labels, test_texts, test_labels = split_data(
        all_sents, all_labels, test_size=0.2, seed=42
    )

    train_labels = np.array(train_labels)
    test_labels = np.array(test_labels)

    print("Train counts:", np.bincount(train_labels, minlength=NUM_CLASSES))
    print("Test counts:", np.bincount(test_labels, minlength=NUM_CLASSES))

    # class weights
    class_counts = np.bincount(train_labels, minlength=NUM_CLASSES)
    weights = torch.tensor(
        [1.0 / c for c in class_counts], dtype=torch.float32, device=device
    )

    # BERT tokenization
    train_input_ids, train_attention = prep_bert_data(train_texts, MAX_LENGTH)
    test_input_ids, test_attention = prep_bert_data(test_texts, MAX_LENGTH)

    # UD vocab over ALL sentences (so train/test share vocab)
    pos_to_idx, dep_to_idx = create_ud_vocab(all_sents, nlp)
    pos_pad_idx = pos_to_idx["<PAD>"]
    dep_pad_idx = dep_to_idx["<PAD>"]

    # UD encoding (which loops over sentences)
    train_pos_ids = []
    train_dep_ids = []
    train_head_dists = []
    for s in train_texts:
        feats = encode_ud_features(s, nlp, pos_to_idx, dep_to_idx, MAX_LENGTH)
        train_pos_ids.append(feats["pos_ids"])
        train_dep_ids.append(feats["dep_ids"])
        train_head_dists.append(feats["head_dists"])

    test_pos_ids = []
    test_dep_ids = []
    test_head_dists = []
    for s in test_texts:
        feats = encode_ud_features(s, nlp, pos_to_idx, dep_to_idx, MAX_LENGTH)
        test_pos_ids.append(feats["pos_ids"])
        test_dep_ids.append(feats["dep_ids"])
        test_head_dists.append(feats["head_dists"])

    train_pos_ids = torch.stack(train_pos_ids, dim=0)
    train_dep_ids = torch.stack(train_dep_ids, dim=0)
    train_head_dists = torch.stack(train_head_dists, dim=0)

    test_pos_ids = torch.stack(test_pos_ids, dim=0)
    test_dep_ids = torch.stack(test_dep_ids, dim=0)
    test_head_dists = torch.stack(test_head_dists, dim=0)

    # AMR
    train_amr = stog.parse_sents(train_texts)
    test_amr = stog.parse_sents(test_texts)
    token_to_idx = build_amr_vocab(train_amr)
    train_amr_tensors = torch.stack([amr_to_tensor(amr, token_to_idx, max_length=MAX_LENGTH) for amr in train_amr])
    test_amr_tensors = torch.stack([amr_to_tensor(amr, token_to_idx, max_length=MAX_LENGTH) for amr in test_amr])

    # labels
    train_labels_t = torch.tensor(train_labels, dtype=torch.long)
    test_labels_t = torch.tensor(test_labels, dtype=torch.long)

    # datasets / loaders
    train_dataset = TensorDataset(
        train_input_ids,
        train_attention,
        train_pos_ids,
        train_dep_ids,
        train_head_dists,
        train_amr_tensors,
        train_labels_t,
    )

    test_dataset = TensorDataset(
        test_input_ids,
        test_attention,
        test_pos_ids,
        test_dep_ids,
        test_head_dists,
        test_amr_tensors,
        test_labels_t,
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    run_all_models(train_loader, test_loader, len(pos_to_idx), len(dep_to_idx), len(token_to_idx), 
                   pos_pad_idx, dep_pad_idx, token_to_idx['<PAD>'], test_labels, weights, epochs=EPOCHS)

if __name__ == "__main__":
    main()