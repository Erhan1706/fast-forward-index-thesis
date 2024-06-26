import torch
import pyterrier as pt
import fast_forward
import numpy as np
import pandas as pd

from fast_forward import OnDiskIndex, Mode
from pathlib import Path
from fast_forward.util.pyterrier import FFScore, FFInterpolate
from pyterrier.measures import nDCG, RR, MAP
from typing import Sequence, Union
from fast_forward.encoder import Encoder
from transformers import AutoModel, AutoTokenizer

class SnowFlakeQueryEncoder(fast_forward.encoder.TransformerEncoder):
  def __call__(self, texts):
    query_prefix = 'Represent this sentence for searching relevant passages: '
    queries_with_prefix = ["{}{}".format(query_prefix, i) for i in texts]
    query_tokens = self.tokenizer(queries_with_prefix, padding=True, truncation=True, return_tensors='pt', max_length=512)

    query_tokens.to(self.device)
    self.model.eval()
    # Compute token embeddings
    with torch.no_grad():
        query_embeddings = self.model(**query_tokens)[0][:, 0]
        #doument_embeddings = self.model(**document_tokens)[0][:, 0]

    # normalize embeddings
    query_embeddings = torch.nn.functional.normalize(query_embeddings, p=2, dim=1)
    
    return query_embeddings.detach().cpu().numpy()

class BGEQueryEncoder(fast_forward.encoder.TransformerEncoder):
  def __call__(self, texts):
    encoded_input = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
    # for s2p(short query to long passage) retrieval task, add an instruction to query (not add instruction for passages)
    # encoded_input = tokenizer([instruction + q for q in queries], padding=True, truncation=True, return_tensors='pt')
    encoded_input.to(self.device)

    self.model.eval()
    # Compute token embeddings
    with torch.no_grad():
        model_output = self.model(**encoded_input)
        # Perform pooling. In this case, cls pooling.
        sentence_embeddings = model_output[0][:, 0]
    # normalize embeddings
    sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
    return sentence_embeddings.detach().cpu().numpy()

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
    self.model.eval()
    with torch.no_grad():
      outputs = self.model(**batch_dict)
      embeddings = outputs.last_hidden_state[:, 0]
    return embeddings.detach().cpu().numpy()

if not pt.started():
    pt.init(tqdm="notebook")

dataset = pt.get_dataset("irds:beir/hotpotqa/test")
devset = pt.get_dataset("irds:beir/hotpotqa/dev")
bm25 = pt.BatchRetrieve("../data/hotpotqa", wmodel="BM25")

### Artic-embed
q_encoder_artic = SnowFlakeQueryEncoder('Snowflake/snowflake-arctic-embed-m')

ff_index_artic = OnDiskIndex.load(
    Path("../data/ffindex_hotpot_snowflake.h5"), query_encoder=q_encoder_artic, mode=Mode.MAXP
) 
ff_index_artic = ff_index_artic.to_memory()
ff_score_artic = FFScore(ff_index_artic)


### BGE 
q_encoder_bge = BGEQueryEncoder('BAAI/bge-base-en-v1.5')
ff_index_bge = OnDiskIndex.load(
    Path("../bge_c/bge/ffindex_hotpotqa_bge_base_en_v1_5.h5"), query_encoder=q_encoder_bge, mode=Mode.MAXP
) 

ff_index_bge = ff_index_bge.to_memory()
ff_score_bge = FFScore(ff_index_bge)
## GTE 
q_encoder_gte = GTEQueryEncoder('Alibaba-NLP/gte-base-en-v1.5')

ff_index_gte = OnDiskIndex.load(
    Path("../gte_c/gte/ffindex_hotpotqa_gte_base_en_v1_5.h5"), query_encoder=q_encoder_gte, mode=Mode.MAXP
)
ff_index_gte = ff_index_gte.to_memory()
ff_score_gte = FFScore(ff_index_gte)


def normalize_column(df, column_name):
    min_val = df[column_name].min()
    max_val = df[column_name].max()
    df[column_name] = (df[column_name] - min_val) / (max_val-min_val)

pl_artic = ~bm25 % 1000 >> ff_score_artic
d_artic = pl_artic.transform(dataset.get_topics())
d_artic_dev = pl_artic.transform(devset.get_topics())
normalize_column(d_artic_dev, 'score')
normalize_column(d_artic_dev, 'score_0')

pl_bge = ~bm25 % 1000 >> ff_score_bge
d_bge_dev = pl_bge.transform(devset.get_topics())
normalize_column(d_bge_dev, 'score')
normalize_column(d_bge_dev, 'score_0')
d_bge_dev.to_csv("bge_hotpot_scores_dev_norm.csv")

pl_gte = ~bm25 % 1000 >> ff_score_gte

d_gte_dev = pl_gte.transform(devset.get_topics())
normalize_column(d_gte_dev, 'score')
normalize_column(d_gte_dev, 'score_0')
d_gte_dev.to_csv("gte_hotpot_scores_dev_norm.csv") 

d_gte = pl_gte.transform(dataset.get_topics())
normalize_column(d_gte, 'score')
normalize_column(d_gte, 'score_0')
d_gte.to_csv("gte_hotpot_scores_norm.csv")


d_artic.to_csv("artic_hotpot_scores_norm.csv")
d_artic_dev.to_csv("artic_hotpot_scores_dev_norm.csv")
