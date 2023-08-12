# Graph Embeddings analysis in Social Network

This repository was created for the purpose of my thesis.
<br><br>

<b>Author</b> : Giannis Kontogiorgakis <br>
<b>Grade</b> : 10.0 / 10.0  <br>

My implementation consists of the following steps:
- Export graphs from database, using pymongo <br>
```python graphUtils/export_graphs.py --period feb_mar ```<br>
> run this task for each period (feb_mar, feb_apr, feb_may, feb_jun)

- Graph Embeddings <br>
After graph export, lets assume that your graph files are stored in this format in folder <b>data</b>: <br>

```
├── data
│   ├── feb_mar
│   │   ├── graph_mention_feb_mar.tsv
│   │   ├── graph_quote_feb_mar.tsv
│   │   ├── graph_multy_feb_mar.tsv
│   ├── feb_apr
│   │   ├── ...
│   ├── feb_may
│   │   ├── ...
│   ├── feb_jun
│   │   ├── ...
```



- Model Selection



