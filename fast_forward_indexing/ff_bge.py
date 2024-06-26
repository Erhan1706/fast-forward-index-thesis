import torch
import pyterrier as pt
from fast_forward.encoder import TransformerEncoder
import torch
from fast_forward import OnDiskIndex, Mode, Indexer
from pathlib import Path
import pandas as pd

"""
Dense indexing using BGE as the query and document encoder for Fast-Forward indexing
"""
class BGEDocumentEncoder(TransformerEncoder):
  def __call__(self, texts):
    encoded_input = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
    # for s2p(short query to long passage) retrieval task, add an instruction to query (not add instruction for passages)
    # encoded_input = tokenizer([instruction + q for q in queries], padding=True, truncation=True, return_tensors='pt')
    encoded_input.to(self.device)

    # Compute token embeddings
    with torch.no_grad():
        model_output = self.model(**encoded_input)
        # Perform pooling. In this case, cls pooling.
        sentence_embeddings = model_output[0][:, 0]
    # normalize embeddings
    sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
    return sentence_embeddings.detach().cpu().numpy()

class BGEQueryEncoder(TransformerEncoder):
  def __call__(self, texts):
    encoded_input = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
    # for s2p(short query to long passage) retrieval task, add an instruction to query (not add instruction for passages)
    # encoded_input = tokenizer([instruction + q for q in queries], padding=True, truncation=True, return_tensors='pt')
    encoded_input.to(self.device)

    # Compute token embeddings
    with torch.no_grad():
        model_output = self.model(**encoded_input)
        # Perform pooling. In this case, cls pooling.
        sentence_embeddings = model_output[0][:, 0]
    # normalize embeddings
    sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
    return sentence_embeddings.detach().cpu().numpy()


if not pt.started():
    pt.init(tqdm="notebook")

torch.cuda.is_available()

dataset = pt.get_dataset("irds:beir/trec-covid") # change  dataset

d_encoder = BGEDocumentEncoder('BAAI/bge-base-en-v1.5', device='cuda:0')
q_encoder = BGEQueryEncoder('BAAI/bge-base-en-v1.5')

# max id length may need to be adjusted based on the dataset
ff_index = OnDiskIndex(
    Path("ffindex_climate_gte.h5"), dim=768, query_encoder=q_encoder, mode=Mode.MAXP, max_id_length=221 
)

def docs_iter():
    for d in dataset.get_corpus_iter():
        yield {"doc_id": d["docno"], "text": d["text"]}

ff_indexer = Indexer(ff_index, d_encoder, batch_size=32)
ff_indexer.index_dicts(docs_iter())