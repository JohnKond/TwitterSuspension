""""####################################################################################################################
Author: Alexander Shevtsov ICS-FORTH
E-mail: shevtsov@ics.forth.gr
--------------------------------------
Parse the tsv graph relation file and train the PytorchBigGraph model in order to store the graph embeddings
####################################################################################################################"""

import argparse
import os
import sys

sys.path.append("/home/gkont/TwitterSuspension/")
path_to_storage = "/Storage/gkont/"


os.environ['OPENBLAS_NUM_THREADS'] = '1'

from pathlib import Path
from os import path
from parser import ParseEmbeddings
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')

from torchbiggraph.converters.importers import TSVEdgelistReader, convert_input_data
from torchbiggraph.config import parse_config
from torchbiggraph.train import train
from torchbiggraph.util import SubprocessInitializer, setup_logging


class SocialGraphEmbedding:
    def __init__(self, graph_type, out_dimmensions, graph_period, first):
        self.dims = out_dimmensions
        self.name = graph_period

        self.DATA_DIR = path_to_storage + 'data/'
        self.EMBED_DIR = path_to_storage + 'embeddings/'

        self.period = graph_period
        self.GRAPH_PATH = self.DATA_DIR +"/"+ self.name + "/graph_{}_{}.tsv".format(graph_type, graph_period)

        
        self.MODEL_DIR = 'model_{}'.format(self.name)
        # self.parser = ParseEmbeddings(graph_type, out_dimmensions, graph_period, first)
        if not path.isfile(self.GRAPH_PATH):
            print("Graph :{} not exists!")
            sys.exit(-1)
        self.main()

        # self.parser.store_embeddings()

    """Generates configuration file in form of pytorchBigGraph"""

    def create_config(self):
        return dict(
            # I/O data
            entity_path=self.EMBED_DIR + self.name + "/entity_{}_json/".format(self.name),
            edge_paths=[self.EMBED_DIR + self.name + "/edges_{}_json/".format(self.name)],
            checkpoint_path=self.EMBED_DIR + self.name + "/checkpoint_{}_json/".format(self.name),
            # Graph structure
            entities={"user": {"num_partitions": 1}},
            relations=[
                {
                    "name": "all_edges",
                    "lhs": "user",
                    "rhs": "user",
                    "operator": "complex_diagonal"
                }
            ],
            dynamic_relations=True,
            dimension=self.dims,
            global_emb=False,
            comparator="dot",
            num_epochs=50,
            num_edge_chunks=10,
            batch_size=10000,
            num_batch_negs=500,
            num_uniform_negs=500,
            loss_fn="softmax",
            lr=0.1,
            relation_lr=0.01,
            eval_fraction=0.0005,
            eval_num_batch_negs=10000,
            eval_num_uniform_negs=0,
            # GPU
            num_gpus=1,
        )

    def trainEmbeddings(self):
        setup_logging()
        config = parse_config(self.create_config())
        subprocess_init = SubprocessInitializer()
        input_edge_paths = [Path(self.GRAPH_PATH)]

        convert_input_data(
            config.entities,
            config.relations,
            config.entity_path,
            config.edge_paths,
            input_edge_paths,
            TSVEdgelistReader(lhs_col=0, rel_col=1, rhs_col=2),
            dynamic_relations=config.dynamic_relations,
        )

        ''' train embeddings '''
        train(config, subprocess_init=subprocess_init)

    def main(self):
        print('training graph : ', self.period)
        self.trainEmbeddings()


parser = argparse.ArgumentParser(
    description='Parse extracted Graph files(.tsv) into PytorchBigGraph model and generate node embeddings')
parser.add_argument('--dims', type=int, dest='dims', required=True,
                    help="Number of output embedding dimensions. (Required!)")
parser.add_argument('--second_portion', dest='second_weeks', default=False, action='store_true',
                    help="Flag parameter describe if should be used for second 21 days data extraction")
parser.add_argument('--multy', dest='multy', default=False, action='store_true',
                    help="Flag that describe graph relation that would be utilized (Multy)")
parser.add_argument('--quote', dest='quote', default=False, action='store_true',
                    help="Flag that describe graph relation that would be utilized (Quote)")
parser.add_argument('--mention', dest='mention', default=False, action='store_true',
                    help="Flag that describe graph relation that would be utilized (Mention)")
parser.add_argument('--retweet', dest='retweet', default=False, action='store_true',
                    help="Flag that describe graph relation that would be utilized (Retweet)")

parser.add_argument('--first', dest='first', default=False, action='store_true',
                    help="Flag that describe graph time period : 1st(23/02 - 23/03)")
parser.add_argument('--second', dest='second', default=False, action='store_true',
                    help="Flag that describe graph time period : 2nd(23/02 - 23/04)")
parser.add_argument('--third', dest='third', default=False, action='store_true',
                    help="Flag that describe graph time period : 3rd(23/02 - 23/05)")
parser.add_argument('--fourth', dest='fourth', default=False, action='store_true',
                    help="Flag that describe graph time period : 1st(23/02 - 23/06)")

args = parser.parse_args()

if __name__ == "__main__":
    graph_period = ""
    if args.first:
        graph_period = "feb_mar"
    elif args.second:
        graph_period = "feb_apr"
    elif args.third:
        graph_period = "feb_may"
    elif args.fourth:
        graph_period = "feb_jun"
    if graph_period == "":
        print("Select the graph period (--first or --second or --third or --fourth).")
        sys.exit(-1)

    PTBG = SocialGraphEmbedding("multy", args.dims, graph_period, first=False if args.second_weeks else True)
    print("Done")
