import torch
import pyterrier as pt

from fast_forward.encoder import TransformerEncoder
import torch
from fast_forward import OnDiskIndex, Mode
from pathlib import Path
import pandas as pd
from fast_forward.util.pyterrier import FFScore, FFInterpolate
from pyterrier.measures import nDCG


class SnowFlakeDocumentEncoder(TransformerEncoder):
  def __call__(self, texts):
    document_tokens =  self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt', max_length=512)
    document_tokens.to(self.device)
    # Compute token embeddings
    self.model.eval()
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
  
class BGEDocumentEncoder(TransformerEncoder):
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

class BGEQueryEncoder(TransformerEncoder):
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


if not pt.started():
    pt.init(tqdm="notebook")

dataset = pt.get_dataset("irds:beir/arguana")
bm25 = pt.BatchRetrieve("../data/beir_arguana", wmodel="BM25")

candidates = (bm25 % 5)(dataset.get_topics())

### Artic-embed
doc_encoder_artic = SnowFlakeDocumentEncoder('Snowflake/snowflake-arctic-embed-m', device="cuda:0")
q_encoder_artic = SnowFlakeQueryEncoder('Snowflake/snowflake-arctic-embed-m')

ff_index_artic = OnDiskIndex.load(
    Path("../data/ffindex_arguana_snowflake.h5"), query_encoder=q_encoder_artic, mode=Mode.MAXP
) 
ff_index_artic = ff_index_artic.to_memory()
ff_score_artic = FFScore(ff_index_artic)

alpha_values = [0.05, 0.1, 0.3, 0.5]
dataframes = []

for alpha in alpha_values:
    dataframes.append(FFInterpolate(alpha=alpha))


### BGE 
d_encoder_bge = BGEDocumentEncoder('BAAI/bge-base-en-v1.5', device='cuda:0')
q_encoder_bge = BGEQueryEncoder('BAAI/bge-base-en-v1.5')

ff_index_bge = OnDiskIndex.load(
    Path("../bge_c/bge/ffindex_arguana_bge_base_en_v1_5.h5"), query_encoder=q_encoder_bge, mode=Mode.MAXP
) ## NEW ONE

ff_index_bge = ff_index_bge.to_memory()
ff_score_bge = FFScore(ff_index_bge)

dataframes_bge = []

for alpha in alpha_values:
    dataframes_bge.append(FFInterpolate(alpha=alpha))

for i in range(len(dataframes)):
    pl_artic = ~bm25 % 1000 >> ff_score_artic >> dataframes[i]
    pl_bge = ~bm25 % 1000 >> ff_score_bge >> dataframes_bge[i]
    exp = pt.Experiment(
        [0.5 * pl_artic + 0.5 * pl_bge],
        dataset.get_topics(),
        dataset.get_qrels(),
        eval_metrics=[nDCG @ 10],
        names=["Artic + BGE"],
    )
    exp.to_csv(f"artic_bge_arguana_{dataframes[i].alpha}.csv")

