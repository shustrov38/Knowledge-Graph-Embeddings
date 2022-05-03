import os

from typing import List, Tuple, Dict, Union

import numpy as np


class wordnet_preprocessed:
    """"
    Class for using preprocessed WordNet with some 
    modifications (less entities and more relation types)
    """
    
    POSSIBLE_DATASET_TYPES = ['train', 'valid', 'test']

    
    def __init__(self, path_to_wordnet_folder: str = None) -> None:
        assert os.path.exists(path_to_wordnet_folder)
        assert path_to_wordnet_folder[-1] != '/'
        
        self.path = path_to_wordnet_folder
        
        self.entities = 0
        self.relations = 0
        
        self.synset2idx = {}
        self.idx2synset = {}
        
        self.synset2def = {}
        self.synset2con = {}
        
        self.__load_synsets_and_definitions()
        
        self.__train = self.__load_dataset_by_type('train')
        self.__valid = self.__load_dataset_by_type('valid')
        self.__test  = self.__load_dataset_by_type('test')
        
    def get_train(self) -> List[Tuple[int, int, int]]:
        return self.__train
    
    def get_valid(self) -> List[Tuple[int, int, int]]:
        return self.__valid
    
    def get_test(self) -> List[Tuple[int, int, int]]:
        return self.__test
        
    def __load_synsets_and_definitions(self) -> None:
        synset = set()
        relset = set()
        
        for dataset_type in self.POSSIBLE_DATASET_TYPES:
            with open(self.path + f'/wordnet-mlj12-{dataset_type}.txt', 'r') as f:
                for line in f.readlines():
                    lhs, rel, rhs = self.__parse_line(line)
                    synset.add(lhs)
                    relset.add(rel)
                    synset.add(rhs)
        
        self.entities = len(synset)
        self.relations = len(relset)
        
        id = 0
        for syn in sorted(synset):
            self.synset2idx[syn] = id
            self.idx2synset[id] = syn
            id += 1
        for rel in sorted(relset):
            self.synset2idx[rel] = id   
            self.idx2synset[id] = rel
            id += 1
            
        with open(self.path + '/wordnet-mlj12-definitions.txt', 'r') as f:
            for line in f.readlines():
                synset, concept, definition = self.__parse_line(line)
                self.synset2def.update({synset: definition}) # may be multiple definitions 
                self.synset2con.update({synset: concept}) # may be multiple concepts 
        
    def __load_dataset_by_type(self, dataset_type: str) -> List[Tuple[int, int, int]]:
        assert dataset_type in self.POSSIBLE_DATASET_TYPES
        
        dataset = []
        with open(self.path + f'/wordnet-mlj12-{dataset_type}.txt', 'r') as f:
            for line in f.readlines():
                lhs, rel, rhs = self.__parse_line(line)
                dataset.append((self.synset2idx[lhs], self.synset2idx[rel], self.synset2idx[rhs]))

        return dataset
        
    def __parse_line(self, line: str) -> List[str]:
        return line.rstrip().split('\t')
    

def get_geographical_data_from_wordnet(wn: wordnet_preprocessed,  \
                                       region_names: List[str],   \
                                       region_synsets: List[str], \
                                       region_colors: List[str]=None) -> Tuple[List[Tuple[int, int, int]], \
                                                                               List[int],  \
                                                                               np.ndarray, \
                                                                               np.ndarray, \
                                                                               Dict[str, Dict[str, Union[str, List[int]]]]]:
    ''' 
    Creates dataset with only geographical data. 
    
    dataset extracted from WordNe by the following rule: 
        (subject, relation, object) == (subject, _part_of, geographical region) 
    '''
    
    assert len(region_names) == len(region_synsets)
    
    if region_colors is None:
        random_0xFFFFFF = lambda: np.random.randint(0, 256)
        random_color = lambda: '#{:02x}{:02x}{:02x}'.format(random_0xFFFFFF(), random_0xFFFFFF(), random_0xFFFFFF())
        region_colors = [random_color() for _ in region_names]
    else:
        assert len(region_names) == len(region_colors)
    
    # fill dataset and separate synsets between regions
    
    regions = dict()
    
    dataset = list()
    unique_idxs = set()
    
    # use this to make factorization classes closer
    # 08562243 __orient_NN_2 the hemisphere that includes Eurasia and Africa and Australia
    # 09275016 __eurasia_NN_1 the land mass formed by the continents of Europe and Asia  
    # 08682575 __west_NN_1 the countries of (originally) Europe and (now including) North America and South America
    world_synsets = ['08562243', '09275016', '08682575']
    world_synsets_idxs = list(map(lambda x: wn.synset2idx[x], world_synsets))
    
    relation_synsets = ['_part_of', '_has_part']
    relation_synsets_idxs = list(map(lambda x: wn.synset2idx[x], relation_synsets))
    
    for region_name, synset, color in zip(region_names, region_synsets, region_colors):
        temp_idxs = []
        
        for l, o, r in sorted(wn.get_train()):
            if o in relation_synsets_idxs and (l == wn.synset2idx[synset] or r == wn.synset2idx[synset]):
                if l not in world_synsets_idxs and l != wn.synset2idx[synset]:
                    temp_idxs.append(l)
                if r not in world_synsets_idxs and r != wn.synset2idx[synset]:
                    temp_idxs.append(r)
                dataset.append((l, o, r))
                
                unique_idxs |= {l, o, r}
        
        regions[region_name] = {'color': color, 'idxs': sorted(list(set(temp_idxs)))}
                
    # fill lookup table for indicies 
                                                                          
    unique_idxs = sorted(list(unique_idxs))
    lookup_table = {idx: i for i, idx in enumerate(unique_idxs)}
    
    # translate indicies

    total_size = len(unique_idxs)
    
    for region_name in region_names:
        regions[region_name]['idxs'] = list(map(lambda x: lookup_table[x], regions[region_name]['idxs']))
            
    dataset = list(map(lambda x: (lookup_table[x[0]], total_size, lookup_table[x[2]]), dataset))        

    
    return dataset, unique_idxs, np.arange(total_size), np.arange(total_size, total_size + len(relation_synsets)), regions