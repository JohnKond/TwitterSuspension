import json
import h5py
import faiss

class NeighboursEmbeddings:

    def __init__(self,dims,init_nodes_path,init_embed_path,new_nodes_path,new_embed_path, new_nodes):
        self.dims = dims
        self.init_nodes_path = init_nodes_path
        self.init_embed_path = init_embed_path
        self.new_nodes_path = new_nodes_path
        self.new_embed_path = new_embed_path
        self.new_nodes = new_nodes
        self.main()


    def NewNodes(self):

        ''' Initial embeddings '''
        with open(self.init_nodes_path, 'r') as f:
            node_names = json.load(f)
        with h5py.File(self.init_embed_path, 'r') as g:
            embeddings = g['embeddings'][:]

        self.embeddingsDict = dict(zip(node_names, embeddings))

        ''' New embeddings '''

        with open(self.new_nodes_path, 'r') as f:
            new_node_names = json.load(f)

        new_nodes = []

        with open(self.new_nodes_path, 'r') as f:
            node_names = json.load(f)
        for name in new_node_names:
            if name not in self.embeddingsDict:
                new_nodes.append(name)

        print(new_nodes)
        return new_nodes


    def nearest_neighbours(self,embed_path, entities_path, entity_name, dims):
        # Create FAISS index
        index = faiss.IndexFlatL2(dims)

        with h5py.File(embed_path, "r") as hf:
            index.add(hf["embeddings"][...])

        # Get trained embedding of Paris
        with open(entities_path, "rt") as tf:
            entity_names = json.load(tf)

        target_entity_offset = entity_names.index(entity_name)
        with h5py.File(embed_path, "r") as hf:
            target_embedding = hf["embeddings"][target_entity_offset, :]

        # Search nearest neighbors
        _, neighbors = index.search(target_embedding.reshape((1, dims)), 3)

        # Map back to entity names
        top_entities = [entity_names[index] for index in neighbors[0]][1:]
        print('Nearest neighbours of ', entity_name, ' are :', top_entities)
        return top_entities



    def getEmbeddings(self,node):
        with open(self.init_nodes_path, 'r') as f:
            node_names = json.load(f)
        with h5py.File(self.init_embed_path, 'r') as g:
            embeddings = g['embeddings'][:]

        embeddingsDict = dict(zip(node_names, embeddings))
        return embeddingsDict[node]

    def newVectorSpace(self, neighbours):
        embed_arr = []
        embed_arr = [[1,2,3,4],[3,4,5,6],[1,4,2,4]]
        # for neigh in neighbours:
        #     embed_arr.append(self.getEmbeddings(neigh))

        return None


    def placeNewNodes(self):
        ''' Calculate distance of new nodes and place them in the initial embeddings'''

        new_nodes_dict = {}
        for node in self.new_nodes:
            neighbours = self.nearest_neighbours(self.new_embed_path, self.new_nodes_path, node, self.dims)
            new_vector = self.newVectorSpace(neighbours)
            new_nodes_dict[node] = new_vector

        ''' Place new vectors dict in initial h5 embeddings file'''


    def main(self):
        self.placeNewNodes()


initembedPath = "checkpoint_first_json/embeddings_user_0.v50.h5"
initentityPath = "entity_first_json/entity_names_user_0.json"
newembedPath = "checkpoint_second_json/embeddings_user_0.v50.h5"
newentityPath = "entity_multy_4_second_json/entity_names_user_0.json"
dims = 4

# NE = NeighboursEmbeddings(4,init_embed_path=initembedPath,init_nodes_path=initentityPath,new_embed_path=newembedPath,new)