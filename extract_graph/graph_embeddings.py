""""####################################################################################################################
Author: Alexander Shevtsov ICS-FORTH
E-mail: shevtsov@ics.forth.gr
--------------------------------------
Parse the tsv graph relation file and train the PytorchBigGraph model in order to store the graph embeddings
####################################################################################################################"""
import ast
import os, sys, argparse

import h5py

sys.path.append('/mnt/c/Users/giankond/Documents/thesis/TwitterSuspension/')

# os.environ['OPENBLAS_NUM_THREADS'] = '1'

from pathlib import Path
from os import path

from emb_parser import ParseEmbeddings
import json

import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')

from torchbiggraph.converters.importers import TSVEdgelistReader, convert_input_data
from torchbiggraph.config import parse_config
from torchbiggraph.train import train
from torchbiggraph.util import SubprocessInitializer, set_logging_verbosity, setup_logging
from graph_knn import NeighboursEmbeddings

class SocialGraphEmbedding:
    def __init__(self, graph_type, out_dimmensions, graph_period, first):
        self.dims = out_dimmensions
        self.DATA_DIR = '../data'
        self.period = graph_period
        self.GRAPH_PATH = self.DATA_DIR + "/graph_{}_{}.tsv".format(graph_type, graph_period)
        # self.GRAPH_PATH = self.DATA_DIR + "/graph_multy_first.tsv"

        self.name = "{}_{}_{}".format(graph_type, self.dims, graph_period)
        self.MODEL_DIR = 'model_{}'.format(self.name)
        self.parser = ParseEmbeddings(graph_type, out_dimmensions, graph_period, first)
        if not path.isfile(self.GRAPH_PATH):
            print("Graph :{} not exists!")
            sys.exit(-1)
        self.main()

        # self.parser.store_embeddings()

    def pretrained_and_new_nodes(self, pretrained_nodes, new_nodes, entity_file, entity_count, embeddings_path):
        """
        pretrained_nodes:
            A dictionary of nodes and their embeddings
        new_nodes:
            A list of new nodes,each new node must have an embedding in pretrained_nodes.
            If no new nodes, use []
        entity_name:
            The entity's name, for example, WHATEVER_0
        data_dir:
            The path to the files that record graph nodes and edges
        embeddings_path:
            The path to the .h5 file of embeddings
        """
        with open(entity_file, 'r') as source:
            nodes = json.load(source)
        dist = {item: ind for ind, item in enumerate(nodes)}

        if len(new_nodes) > 0:
            # modify both the node names and the node count
            extended = nodes.copy()
            extended.extend(new_nodes)
            with open(entity_file, 'w') as source:
                json.dump(extended, source)
            with open(entity_count, 'w') as source:
                source.write('%i' % len(extended))

        if len(new_nodes) == 0:
            # if no new nodes are added, we won't bother create a new .h5 file, but just modify the original one
            with h5py.File(embeddings_path, 'r+') as source:

                for node, embedding in pretrained_nodes.items():
                    if node in nodes:
                        source['embeddings'][dist[node]] = embedding
        else:
            # if there are new nodes, then we must create a new .h5 file
            # see https://stackoverflow.com/a/47074545/8366805
            with h5py.File(embeddings_path, 'r+') as source:
                embeddings = list(source['embeddings'])
                optimizer = list(source['optimizer'])
            for node, embedding in pretrained_nodes.items():
                if node in nodes:
                    embeddings[dist[node]] = embedding
            # append new nodes in order
            for node in new_nodes:
                if node not in list(pretrained_nodes.keys()):
                    raise ValueError
                else:
                    embeddings.append(pretrained_nodes[node])
            # write a new .h5 file for the embedding
            with h5py.File(embeddings_path, 'w') as source:
                source.create_dataset('embeddings', data=embeddings, )
                optimizer = [item.encode('ascii') for item in optimizer]
                source.create_dataset('optimizer', data=optimizer)

    """Generates configuration file in form of pytorchBigGraph"""

    def create_config(self):
        return dict(
            # I/O data
            entity_path="entity_{}_json/".format(self.name),
            edge_paths=["edges_{}_json/".format(self.name)],
            checkpoint_path="checkpoint_{}_json/".format(self.name),
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
            # num_edge_chunks=10,
            # batch_size=10000,
            # num_batch_negs=500,
            num_uniform_negs=500,
            loss_fn="softmax",
            lr=0.1,
            # relation_lr=0.01,
            eval_fraction=0.0005,
            # eval_num_batch_negs=10000,
            # eval_num_uniform_negs=0,
            # GPU
            # num_gpus=1,
        )

        # with open(nodes_path, 'r') as f:
        #     node_names = json.load(f)
        #
        # with h5py.File(embeddings_path, 'r') as g:
        #     embeddings = g['embeddings'][:]
        #
        # node2embedding_PlanB = dict(zip(node_names, embeddings))
        # print('hey')




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

        # train embeddings
        train(config, subprocess_init=subprocess_init)




    def main(self):

        # if path.exists(self.parser.entity_file) and path.exists(self.parser.embeddings_file):
        if self.period != 'first':
            print('new graph')
            # print("Embedding already exists... Creating new embeddings")
            self.trainEmbeddings()
            return None


        print('first graph')
        # initial embeddings
        self.name = "first"
        self.trainEmbeddings()
        new_nodes = self.newNodes()
        placeNewNodes(self.new_nodes_path, self.new_embeds_path, new_nodes)


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
    graph_type = ""
    if args.multy:
        graph_type = "multy"
    elif args.quote:
        graph_type = "quote"
    elif args.mention:
        graph_type = "mention"
    elif args.retweet:
        graph_type = "retweet"
    if graph_type == "":
        print("Select the graph relation (--multy or --retweet or --quote or --mention).")
        sys.exit(-1)

    graph_period = ""
    if args.first:
        graph_period = "first"
    elif args.second:
        graph_period = "second"
    elif args.third:
        graph_period = "third"
    elif args.fourth:
        graph_period = "fourth"
    if graph_period == "":
        print("Select the graph period (--first or --second or --third or --fourth).")
        sys.exit(-1)

    print(graph_period)
    PTBG = SocialGraphEmbedding(graph_type, args.dims, graph_period, first=False if args.second_weeks else True)
    print("Done")
