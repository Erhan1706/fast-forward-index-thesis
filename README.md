# Exploring methods to improve effectiveness of ad-hoc retrieval systems for long and complex queries

This is the implementation corresponding to my final BSc thesis as part of the TU Delft 2024 Research Project, the paper can be found [here](https://repository.tudelft.nl/record/uuid:1c94eeda-9c0e-4122-9652-418ac3127a9c). We explore different methods to improve ranking quality of long and complex queries in different ad-hoc retrieval tasks, using the [Fast-Forward index framework](https://arxiv.org/abs/2311.01263). The methods explored include query reduction using large language models and re-ranking utilising multiple semantic models.   


## Installation & Usage
Install all necessary dependencies:
~~~
pip install -r requirements.txt
~~~

In order to run this code, for each dataset both a sparse and dense index  are needed. Sparse indexing was done using PyTerrier using the following script [pt_index.py](sparse_indexing/pt_index.ipynb). All the datasets used are provided by [ir_datasets](https://ir-datasets.com/) and can be accessed in [PyTerrier](https://pyterrier.readthedocs.io/en/latest/datasets.html#available-datasets). 
Dense indexing was done using the [Fast-Forward index framework](https://github.com/mrjleo/fast-forward-indexes), the scripts for all 3 dense encoders are available in the [fast_forward_indexing](/fast_forward_indexing/) directory. 

<u>Note:</u> due to their large storage size, it isn't possible to upload the indexes to this repository. The indexing process is very resource-intensive and was primarily conducted using the [Delft High Performance Computing Centrte](https://doc.dhpc.tudelft.nl/delftblue/). As this may be a limitation for some users, the indexes are also available upon request.
## Overview 
The repository is organized as follows:
<ul>
<li> fast_forward_indexing - includes all the scripts related to dense indexing. <a href="/fast_forward_indexing/script_pt.sh">/fast_forward_indexing/script_pt.sh </a> contains the bash script utilised when indexing in the DelftBlue supercomputer.
<li> length_experiments - collection of scripts that measure the retrieval
quality for each individual query of the dataset and plots it against their respective length.
<li> multi_rerank - contains all the experiments related to utilising multiple dense re-rankers in the Fast-Forward framework. 
<ul>
<li> generate_scores - generate the final ranking scores before interpolation
<li> multi_rank - experiments that compare the ranking performance for various numbers of dense re-rankers.
<li> scifact_alpha_tuning - script that tuned the alpha values in the development set to their optimal values
</ul>
<li> query_reduction - contains all the experiments related to query reduction using LLM's. 
<ul>
<li> llama3_reduce.py - script that generates the reductions using Meta-Llama-3-8B-Instruct model. 
<li> reduced_queries - directory that stores the reduced queries generated in csv format. 
<li> system_prompts.txt - system prompts utilised for each dataset.
<li> eval_reduction_* - scripts that compare ranking quality between the original and reduced queries. 
</ul>
<li> sparse_indexing - includes all the scripts related to sparse indexing. 
</ul>



