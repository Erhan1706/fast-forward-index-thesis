import pyterrier as pt
import ir_measures
from ir_measures import *
import plotly.express as px
from fast_forward import OnDiskIndex, Mode
from pathlib import Path
import plotly.express as px
import pandas as pd
from fast_forward.util.pyterrier import FFScore, FFInterpolate
from fast_forward.encoder import TransformerEncoder
import torch

"""
Compute nDCG@10 scores for each individual query in the dataset. Returns csv file with query length and nDCG@10 scores for 
each query.
"""

if not pt.started():
    pt.init(tqdm="notebook")

class SnowFlakeQueryEncoder(TransformerEncoder):
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

q_encoder_artic = SnowFlakeQueryEncoder('Snowflake/snowflake-arctic-embed-m')
dataset = pt.get_dataset("irds:beir/trec-covid")
bm25 = pt.BatchRetrieve("./data/trec-covid", wmodel="BM25") # change dataset

ff_index_artic = OnDiskIndex.load(
    Path("./datam/ffindex_trec-covid_snowflake.h5"), query_encoder=q_encoder_artic, mode=Mode.MAXP
) # Change index path to respective index path
ff_index_artic = ff_index_artic.to_memory()

ff_int = FFInterpolate(alpha=0.1)

def normalize_column(df, column_name):
    min_val = df[column_name].min()
    max_val = df[column_name].max()
    df[column_name] = (df[column_name] - min_val) / (max_val-min_val)


def gen_data(q, t, bm25, ff_score):
  axis = []
  q_rels = q
  topics = t
  # For each query compute individual nDCG@10
  for i, row in topics.iterrows():
      qrel = q_rels[q_rels["qid"] == topics["qid"][i]]
      qrel = qrel.rename(columns={'qid': 'query_id', 'docno': 'doc_id', 'label': 'relevance'})

      sp = bm25.search(topics["query"][i])
      re_ranked = ff_score(sp)
      normalize_column(re_ranked, "score")
      normalize_column(re_ranked, "score_0")
      
      run = ff_int(re_ranked)
      run = run.rename(columns={'qid': 'query_id', 'docno': 'doc_id', 'label': 'relevance'})
      run["query_id"] = topics["qid"][i]

      score = ir_measures.calc_aggregate([nDCG@10], qrel, run)
      axis.append((len(row["query"].split()), score[nDCG@10]))

  return axis

def remove_non_alphanumeric(text):
    return ''.join(ch if ch.isalnum() or ch == " " else "" for ch in text)


topics = dataset.get_topics('text')
# If using reduced topics from query reduction, change path to respective csv score files
reduced_topics = pd.read_csv("./query_reduction/reduced_trec_cov2.csv", usecols=["qid", "query"])
# pre process data to correct format 
reduced_topics["qid"] = reduced_topics["qid"].astype(str)
reduced_topics['query'] = reduced_topics['query'].apply(lambda x: remove_non_alphanumeric(x))
reduced_topics['query'] = reduced_topics['query'].str.lower()

axis = gen_data(dataset.get_qrels(), reduced_topics, bm25, FFScore(ff_index_artic))
df = pd.DataFrame(axis, columns=["Query Length", "nDCG@10"]) 
df.to_csv("cov_axis_reduced.csv", index=False) 