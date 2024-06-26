from llama_cpp import Llama
import json
from typing import (
    AbstractSet,
    cast,
    Collection,
    Dict,
    Iterator,
    List,
    Literal,
    Sequence,
    TypedDict,
    Union,
)
import pyterrier as pt
import pandas as pd
from pprint import pprint

if not pt.started():
    pt.init()

my_model_path = "Meta-Llama-3-8B-Instruct-Q6_K.gguf" # possible to use another quantization of the model
CONTEXT_SIZE = 8000

# remove n_gpu_layers if you cuda is not enabled
model = Llama(model_path=my_model_path, kv_overrides={"tokenizer.ggml.pre":"llama3"}, n_ctx=CONTEXT_SIZE, n_gpu_layers=33)

def generate_text_from_prompt(user_prompt,
                             max_tokens = 5000,
                             temperature = 0.5,
                             top_p = 0.6,
                             echo = True,
                             stop = ["###"]):
   
   
   prompt = [
    {"role": "system", "content": """Perform query reduction in the given queries. Try to retain as much meaning as possible from the original query even if it comes at a cost of smaller reductions. If you consider the query to be too short for reduction you may leave it unchanged. Answer with only the reduced query nothing else. Here are some examples:
Query: human produced carbon might be one of the factors of climate change but there s simply no evidence that it is a significant one
Reduced: human-produced carbon's role in climate change is uncertain, but not proven significant
Query: global warming ceased around the end of the twentieth century and was followed since 1997 by 19 years of stable temperature
Reduces: global warming ceased in late 20th century, with temperatures remaining stable since 1997
Query: extreme melting and changes to the climate like this has released pressure on to the continent allowing the ground to rise up
Reduced: extreme melting and climate changes release pressure to the continent causing rising of the ground"""},
    {"role": "user", "content": f"{user_prompt}"},
   ]
   # Define the parameters
   return model.create_chat_completion(
       prompt,
       max_tokens=max_tokens,
       temperature=temperature,
       top_p=top_p,
       stop=stop,
   )["choices"][0]["message"]["content"]

dataset = pt.get_dataset("irds:beir/climate-fever")
topics = dataset.get_topics()

topics["query"] = topics["query"].apply(generate_text_from_prompt)
topics.to_csv("reduced_queries_climate2.csv")

#generate_text_from_prompt("what are the transmission routes of coronavirus?")