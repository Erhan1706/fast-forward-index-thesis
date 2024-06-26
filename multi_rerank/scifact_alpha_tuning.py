import torch
import pyterrier as pt
from fast_forward.util.pyterrier import FFInterpolate, FFScore
from pyterrier.measures import RR, nDCG, MAP
from fast_forward.encoder import TransformerEncoder
from fast_forward import OnDiskIndex, Mode
from pathlib import Path
import pandas as pd
from transformers import AutoModel, AutoTokenizer
from typing import Callable, Sequence, Union
from fast_forward.encoder import Encoder
import numpy as np
import itertools


if not pt.started():
    pt.init(tqdm="notebook")
torch.cuda.is_available()


dataset = pt.get_dataset("irds:beir/scifact/test")
devset= pt.get_dataset("irds:beir/scifact/train")

bm25 = pt.BatchRetrieve("./data/scifact", wmodel="BM25")

class SnowFlakeQueryEncoder(TransformerEncoder):
  def __call__(self, texts):
    query_prefix = 'Represent this sentence for searching relevant passages: '
    queries_with_prefix = ["{}{}".format(query_prefix, i) for i in texts]
    query_tokens = self.tokenizer(queries_with_prefix, padding=True, truncation=True, return_tensors='pt', max_length=512)

    query_tokens.to(self.device)
    self.model.eval()

    #document_tokens =  self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt', max_length=512)
    # Compute token embeddings
    with torch.no_grad():
        query_embeddings = self.model(**query_tokens)[0][:, 0]
        #doument_embeddings = self.model(**document_tokens)[0][:, 0]

    # normalize embeddings
    query_embeddings = torch.nn.functional.normalize(query_embeddings, p=2, dim=1)
    #doument_embeddings = torch.nn.functional.normalize(doument_embeddings, p=2, dim=1)
    return query_embeddings.detach().cpu().numpy()
  
q_encoder_artic = SnowFlakeQueryEncoder('Snowflake/snowflake-arctic-embed-m')

ff_index_artic = OnDiskIndex.load(
    Path("./datam/ffindex_scifact_snowflake_arctic_embed_m.h5"), query_encoder=q_encoder_artic, mode=Mode.MAXP
)
ff_index_artic = ff_index_artic.to_memory()

ff_score_artic = FFScore(ff_index_artic)

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
  

q_encoder_bge = BGEQueryEncoder('BAAI/bge-base-en-v1.5')

ff_index_bge = OnDiskIndex.load(
    Path("./bge/ffindex_scifact_bge_base_en_v1_5.h5"), query_encoder=q_encoder_bge, mode=Mode.MAXP
)

ff_index_bge = ff_index_bge.to_memory()

ff_score_bge = FFScore(ff_index_bge)

def normalize_column(df, column_name):
    min_val = df[column_name].min()
    max_val = df[column_name].max()
    df[column_name] = (df[column_name] - min_val) / (max_val-min_val)


pl_scifact = ~bm25 % 1000 >> FFScore(ff_index_artic)
d_artic = pl_scifact.transform(dataset.get_topics())
normalize_column(d_artic, "score")
normalize_column(d_artic, "score_0")

d_artic_dev = pl_scifact.transform(devset.get_topics())
normalize_column(d_artic_dev, "score")
normalize_column(d_artic_dev, "score_0")

pl_bge = ~bm25 % 1000 >> ff_score_bge
d_bge = pl_bge.transform(dataset.get_topics())
normalize_column(d_bge, "score")
normalize_column(d_bge, "score_0")

d_bge_dev = pl_bge.transform(devset.get_topics())
normalize_column(d_bge_dev, "score")
normalize_column(d_bge_dev, "score_0")

combinations = [(0, 0.5, 0.5), (0.05, 0.425, 0.425), (0.2, 0.4, 0.4), 
                (0.1, 0.2, 0.7), (0.1, 0.7, 0.2), (0, 0.3, 0.7),
                (0.1, 0.4, 0.5), (0.1, 0.5, 0.4), (0, 0.7, 0.3),
                (0.05, 0.45, 0.5), (0.05, 0.5, 0.45), (0.025, 0.275, 0.7),
                (0.025, 0.7, 0.275)]

sc_artic_dev = pt.Transformer.from_df(d_artic_dev)
sc_bge_dev = pt.Transformer.from_df(d_bge_dev)
max_comb = [0, 0, 0]
max_score = 0

for combination in combinations:
  exp = pt.Experiment(
      [combination[0] * ~bm25 + combination[1] * sc_artic_dev + combination[2] * sc_bge_dev],
      devset.get_topics(),
      devset.get_qrels(),
      eval_metrics=[nDCG @ 10],
      names=["Artic + BGE"],
  )
  if exp["nDCG@10"].values[0] > max_score:
    max_score = exp["nDCG@10"].values[0]
    max_comb = combination

print(f"Best alpha for bm25: {max_comb[0]}, artic: {max_comb[1]} and bge: {max_comb[2]}")

sc_artic = pt.Transformer.from_df(d_artic)
sc_bge = pt.Transformer.from_df(d_bge)
pt.Experiment(
   [max_comb[0] * ~bm25 + max_comb[1] * sc_artic + max_comb[2] * sc_bge],
    dataset.get_topics(),
    dataset.get_qrels(),
    eval_metrics=[nDCG @ 10],
    names=["Artic + BGE"],
).to_csv(f"scifact_artic_bge_{max_comb[0]}_{max_comb[1]}_{max_comb[2]}.csv")

