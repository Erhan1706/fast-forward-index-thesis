import pyterrier as pt
from fast_forward.encoder import Encoder
import torch
from fast_forward import OnDiskIndex, Mode, Indexer
from pathlib import Path
import pandas as pd
from transformers import AutoModel, AutoTokenizer
from typing import Callable, Sequence, Union
import numpy as np


"""
Dense indexing using GTE as the query and document encoder for Fast-Forward indexing
"""
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


class GTEDocumentEncoder(TransformerEncoder):
  def __call__(self, texts):
    batch_dict = self.tokenizer(texts, max_length=8192, padding=True, truncation=True, return_tensors='pt')
    batch_dict.to(self.device)
    with torch.no_grad():
      outputs = self.model(**batch_dict)
      embeddings = outputs.last_hidden_state[:, 0]
    return embeddings.detach().cpu().numpy()

class GTEQueryEncoder(TransformerEncoder):
  def __call__(self, texts):
    batch_dict = self.tokenizer(texts, max_length=8192, padding=True, truncation=True, return_tensors='pt')
    batch_dict.to(self.device)
    
    with torch.no_grad():
      outputs = self.model(**batch_dict)
      embeddings = outputs.last_hidden_state[:, 0]
    return embeddings.detach().cpu().numpy()


if not pt.started():
    pt.init(tqdm="notebook")

torch.cuda.is_available()

dataset = pt.get_dataset("irds:beir/trec-covid") # change dataset

d_encoder = GTEDocumentEncoder('Alibaba-NLP/gte-base-en-v1.5', device='cuda:0')
q_encoder = GTEQueryEncoder('Alibaba-NLP/gte-base-en-v1.5')

# max id length may need to be adjusted based on the dataset
ff_index = OnDiskIndex(
    Path("./data/ffindex_trec_cov_gte.h5"), dim=768, query_encoder=q_encoder, mode=Mode.MAXP, max_id_length=8
)

def docs_iter():
    for d in dataset.get_corpus_iter():
        yield {"doc_id": d["docno"], "text": d["text"]}

ff_indexer = Indexer(ff_index, d_encoder, batch_size=32)
ff_indexer.index_dicts(docs_iter())