import torch
import pyterrier as pt
from fast_forward.encoder import TransformerEncoder
import torch
from fast_forward import OnDiskIndex, Mode, Indexer
from pathlib import Path
import pandas as pd

"""
Dense indexing using Artic-embed-m as the query and document encoder for Fast-Forward indexing
"""
class SnowFlakeDocumentEncoder(TransformerEncoder):
  def __call__(self, texts):
    document_tokens =  self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt', max_length=512)
    document_tokens.to(self.device)
    # Compute token embeddings
    with torch.no_grad():
        #query_embeddings = self.model(**query_tokens)[0][:, 0]
        doument_embeddings = self.model(**document_tokens)[0][:, 0]

    # normalize embeddings
    #query_embeddings = torch.nn.functional.normalize(query_embeddings, p=2, dim=1)
    doument_embeddings = torch.nn.functional.normalize(doument_embeddings, p=2, dim=1)
    return doument_embeddings.detach().cpu().numpy()

class SnowFlakeQueryEncoder(TransformerEncoder):
  def __call__(self, texts):
    query_prefix = 'Represent this sentence for searching relevant passages: '
    queries_with_prefix = ["{}{}".format(query_prefix, i) for i in texts]
    query_tokens = self.tokenizer(queries_with_prefix, padding=True, truncation=True, return_tensors='pt', max_length=512)

    query_tokens.to(self.device)
    #document_tokens =  self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt', max_length=512)
    # Compute token embeddings
    with torch.no_grad():
        query_embeddings = self.model(**query_tokens)[0][:, 0]
        #doument_embeddings = self.model(**document_tokens)[0][:, 0]

    # normalize embeddings
    query_embeddings = torch.nn.functional.normalize(query_embeddings, p=2, dim=1)
    #doument_embeddings = torch.nn.functional.normalize(doument_embeddings, p=2, dim=1)
    return query_embeddings.detach().cpu().numpy()


if not pt.started():
    pt.init(tqdm="notebook")

torch.cuda.is_available()

dataset = pt.get_dataset("irds:beir/climate-fever") # change  dataset

doc_encoder = SnowFlakeDocumentEncoder('Snowflake/snowflake-arctic-embed-m', device="cuda:0")
q_encoder = SnowFlakeQueryEncoder('Snowflake/snowflake-arctic-embed-m')

# max id length may need to be adjusted based on the dataset
ff_index = OnDiskIndex(
    Path("ffindex_climate_snowflake.h5"), dim=768, query_encoder=q_encoder, mode=Mode.MAXP, max_id_length=221
)


def docs_iter():
    for d in dataset.get_corpus_iter():
        yield {"doc_id": d["docno"], "text": d["text"]}

ff_indexer = Indexer(ff_index, doc_encoder, batch_size=8)
ff_indexer.index_dicts(docs_iter())