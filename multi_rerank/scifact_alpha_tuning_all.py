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

bm25 = pt.BatchRetrieve("../data/scifact", wmodel="BM25")

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
    Path("../artic_c/snowflake_arctic_embed/ffindex_scifact_snowflake_arctic_embed_m.h5"), query_encoder=q_encoder_artic, mode=Mode.MAXP
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
    Path("../bge_c/bge/ffindex_scifact_bge_base_en_v1_5.h5"), query_encoder=q_encoder_bge, mode=Mode.MAXP
)

ff_index_bge = ff_index_bge.to_memory()

ff_score_bge = FFScore(ff_index_bge)

### GTE
class TransformerEncoder(Encoder):
    """Uses a pre-trained transformer model for encoding. Returns the pooler output."""

    def __init__(
        self, model: Union[str, Path], device: str = "cpu", **tokenizer_args
    ) -> None:
        """Create a transformer encoder.

        Args:
            model (Union[str, Path]): Pre-trained transformer model (name or path).
            device (str, optional): PyTorch device. Defaults to "cpu".
            **tokenizer_args: Additional tokenizer arguments.
        """
        super().__init__()
        self.model = AutoModel.from_pretrained(model, trust_remote_code=True)
        self.model.to(device)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.device = device
        self.tokenizer_args = tokenizer_args

    def __call__(self, texts: Sequence[str]) -> np.ndarray:
        inputs = self.tokenizer(texts, return_tensors="pt", **self.tokenizer_args)
        inputs.to(self.device)
        embeddings = self.model(**inputs).pooler_output.detach().cpu().numpy()
        return embeddings

class GTEQueryEncoder(TransformerEncoder):
  def __call__(self, texts):
    batch_dict = self.tokenizer(texts, max_length=8192, padding=True, truncation=True, return_tensors='pt')
    batch_dict.to(self.device)
    
    with torch.no_grad():
      outputs = self.model(**batch_dict)
      embeddings = outputs.last_hidden_state[:, 0]
    return embeddings.detach().cpu().numpy()

q_encoder_gte = GTEQueryEncoder('Alibaba-NLP/gte-base-en-v1.5')

ff_index_gte = OnDiskIndex.load(
    Path("../gte_c/gte/ffindex_scifact_gte_base_en_v1_5.h5"), query_encoder=q_encoder_gte, mode=Mode.MAXP
)
ff_index_gte = ff_index_gte.to_memory()
ff_score_gte = FFScore(ff_index_gte)


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

pl_gte = ~bm25 % 1000 >> ff_score_gte
d_gte = pl_gte.transform(dataset.get_topics())
normalize_column(d_gte, "score")
normalize_column(d_gte, "score_0")

d_gte_dev = pl_gte.transform(devset.get_topics())
normalize_column(d_gte_dev, "score")
normalize_column(d_gte_dev, "score_0")

combinations = [(0, 0.33, 0.33, 0.34), (0.05, 0.316, 0.316, 0.316), (0.025, 0.325, 0.325, 0.325), 
                (0.1, 0.1, 0.6, 0.1), (0.1, 0.6, 0.1, 0.1), (0.1, 0.1, 0.1, 0.6)
                (0.2, 0.2, 0.4, 0.2), (0.2, 0.4, 0.2, 0.2), (0.2, 0.2, 0.2, 0.4)
                (0.025, 0.175, 0.5, 0.3), (0.025, 0.5, 0.175, 0.3), (0.025, 0.3, 0.175, 0.5),
                (0.025, 0.5, 0.3, 0.175)]

sc_artic = pt.Transformer.from_df(d_artic_dev)
sc_bge = pt.Transformer.from_df(d_bge_dev)
sc_gte = pt.Transformer.from_df(d_gte_dev)
max_comb = [0, 0, 0, 0]
max_score = 0

for combination in combinations:
  exp = pt.Experiment(
      [combination[0] * ~bm25 + combination[1] * sc_artic + combination[2] * sc_bge + combination[3] * sc_gte],
      devset.get_topics(),
      devset.get_qrels(),
      eval_metrics=[nDCG @ 10],
      names=["Artic + BGE + GTE"],
  )
  if exp["nDCG@10"].values[0] > max_score:
    max_score = exp["nDCG@10"].values[0]
    max_comb = combination

print(f"Best alpha for bm25: {max_comb[0]}, artic: {max_comb[1]} gte: {max_comb[3]} and bge: {max_comb[2]}")

sc_artic = pt.Transformer.from_df(d_artic)
sc_bge = pt.Transformer.from_df(d_bge)
sc_gte = pt.Transformer.from_df(d_gte)
pt.Experiment(
   [combination[0] * ~bm25 + combination[1] * sc_artic + combination[2] * sc_bge + combination[3] * sc_gte],
    dataset.get_topics(),
    dataset.get_qrels(),
    eval_metrics=[nDCG @ 10],
    names=["Artic + BGE + GTE"],
).to_csv(f"scifact_artic_gte_bge_{max_comb[0]}_{max_comb[1]}_{max_comb[2]}_{max_comb[3]}.csv")

"""
sc_artic = pt.Transformer.from_df(d_artic_dev)
sc_bge = pt.Transformer.from_df(d_bge_dev)
sc_gte = pt.Transformer.from_df(d_gte_dev)
max_comb = [0, 0, 0]
max_score = 0

for combination in combinations:
  exp = pt.Experiment(
      [combination[0] * ~bm25 + combination[1] * sc_artic + combination[2] * sc_gte],
      devset.get_topics(),
      devset.get_qrels(),
      eval_metrics=[nDCG @ 10],
      names=["Artic + GTE"],
  )
  if exp["nDCG@10"].values[0] > max_score:
    max_score = exp["nDCG@10"].values[0]
    max_comb = combination

sc_artic = pt.Transformer.from_df(d_artic)
sc_bge = pt.Transformer.from_df(d_bge)
sc_gte = pt.Transformer.from_df(d_gte)
pt.Experiment(
   [combination[0] * ~bm25 + combination[1] * sc_gte + combination[2] * sc_bge],
    dataset.get_topics(),
    dataset.get_qrels(),
    eval_metrics=[nDCG @ 10],
    names=["Artic + GTE"],
).to_csv(f"scifact_artic_gte_{max_comb[0]}_{max_comb[1]}_{max_comb[2]}.csv")
"""