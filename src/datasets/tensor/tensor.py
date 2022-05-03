import os
import pickle

from typing import List, Tuple, Dict, Union

import numpy as np
import scipy.sparse as sp

class tensor_datasets:
    """"
    Class for using custom datasets (umls, kinships, nations)
    """
    
    POSSIBLE_DATASETS = ['kinships', 'umls', 'nations']
    POSSIBLE_DATASET_TYPES = ['valid', 'test', 'train-pos', 'train-neg']
    
    
    def __init__(self, path_to_folder: str = None) -> None:
        assert os.path.exists(path_to_folder)
        assert path_to_folder[-1] != '/'
        
        self.path = path_to_folder
        self.data_path = f'{self.path}/../data'
        
    def get(self, dataset: str, dataset_type: str, fold_num: int=0) -> List[Tuple[int, int, int]]:
        assert dataset in self.POSSIBLE_DATASETS
        assert dataset_type in self.POSSIBLE_DATASET_TYPES
                   
        idx_sequences = []
        for entity in ['lhs', 'rel', 'rhs']:
            idx_sequences.append(self.__get_indexed_sequence(f'{self.data_path}/{dataset}-{dataset_type}-{entity}-fold{fold_num}.pkl'))
        
        return list(zip(*idx_sequences))
        
    def create_folds(self, K: int=10) -> None:
        for dataset in self.POSSIBLE_DATASETS:
            f = open(self.path + '/' + dataset + '.pkl', 'rb')
            dictdata = pickle.load(f, encoding='latin1')
            tensordata = dictdata['tensor']

            # List non-zeros
            lnz = []
            # List zeros
            lz = []
            # List of feature triplets
            if dataset == 'nations':
                lzfeat = []
                lnzfeat = []
            # Fill the lists
            for i in range(tensordata.shape[0]):
                for j in range(tensordata.shape[1]):
                    for k in range(tensordata.shape[2]):
                        # Separates features triplets for nation
                        if dataset == 'nations' and (i >= 14 or j >= 14):
                            if tensordata[i, j, k] == 0:
                                lzfeat += [(i, j, k)]
                            elif tensordata[i, j, k] == 1:
                                lnzfeat += [(i, j, k)]
                        else:
                            if tensordata[i, j, k] == 0:
                                lz += [(i, j, k)]
                            elif tensordata[i, j, k] == 1:
                                lnz += [(i, j, k)]

            # Pad the feature triplets lists (same for all training folds)
            if dataset == 'nation':
                if len(lzfeat) < len(lnzfeat):
                    while len(lzfeat) < len(lnzfeat):
                        lzfeat += lzfeat[:len(lnzfeat) - len(lzfeat)]
                else:
                    while len(lnzfeat) < len(lzfeat):
                        lnzfeat += lnzfeat[:len(lzfeat) - len(lnzfeat)]

            f = open(self.path + '/' + dataset + '_permutations.pkl', 'rb')
            idxnz = pickle.load(f, encoding='latin1')
            idxz = pickle.load(f, encoding='latin1')
            f.close()

            # For each fold
            for k in range(K):
                if k != K - 1:
                    tmpidxnz = (idxnz[:k * len(idxnz) // K] + idxnz[(k + 2) * len(idxnz) // K:])
                    tmpidxz = (idxz[:k * len(idxz) // K] + idxz[(k + 2) * len(idxz) // K:])
                    tmpidxtestnz = idxnz[k * len(idxnz) // K:(k + 1) * len(idxnz) // K]
                    tmpidxtestz = idxz[k * len(idxz) // K:(k + 1) * len(idxz) // K]
                    tmpidxvalnz = idxnz[(k + 1) * len(idxnz) // K:(k + 2) * len(idxnz) // K]
                    tmpidxvalz = idxz[(k + 1) * len(idxz) // K:(k + 2) * len(idxz) // K]
                else:
                    tmpidxnz = idxnz[len(idxnz) // K:k * len(idxnz) // K]
                    tmpidxz = idxz[len(idxz) // K:k * len(idxz) // K]
                    tmpidxtestnz = idxnz[k * len(idxnz) // K:(k + 1) * len(idxnz) // K]
                    tmpidxtestz = idxz[k * len(idxz) // K:(k + 1) * len(idxz) // K]
                    tmpidxvalnz = idxnz[:len(idxnz) // K]
                    tmpidxvalz = idxz[:len(idxz) // K]

                # Test data files
                testl = sp.lil_matrix((tensordata.shape[1] + tensordata.shape[2], len(tmpidxtestnz) + len(tmpidxtestz)))
                testr = sp.lil_matrix((tensordata.shape[1] + tensordata.shape[2], len(tmpidxtestnz) + len(tmpidxtestz)))
                testo = sp.lil_matrix((tensordata.shape[1] + tensordata.shape[2], len(tmpidxtestnz) + len(tmpidxtestz)))
                outtest = []
                ct = 0
                for j in tmpidxtestnz:
                    i = lnz[j]
                    testl[i[0], ct] = 1
                    testr[i[1], ct] = 1
                    testo[i[2] + tensordata.shape[1], ct] = 1
                    outtest += [1]
                    ct += 1
                for j in tmpidxtestz:
                    i = lz[j]
                    testl[i[0], ct] = 1
                    testr[i[1], ct] = 1
                    testo[i[2] + tensordata.shape[1], ct] = 1
                    outtest += [0]
                    ct += 1
                f = open(f'{self.data_path}/%s-test-lhs-fold%s.pkl' % (dataset, k), 'wb')
                g = open(f'{self.data_path}/%s-test-rhs-fold%s.pkl' % (dataset, k), 'wb')
                h = open(f'{self.data_path}/%s-test-rel-fold%s.pkl' % (dataset, k), 'wb')
                l = open(f'{self.data_path}/%s-test-targets-fold%s.pkl' % (dataset, k), 'wb')
                pickle.dump(testl.tocsr(), f)
                pickle.dump(testr.tocsr(), g)
                pickle.dump(testo.tocsr(), h)
                pickle.dump(np.asarray(outtest), l)
                f.close()
                g.close()
                h.close()
                l.close()

                # Valid data files
                validl = sp.lil_matrix((tensordata.shape[1] + tensordata.shape[2], len(tmpidxvalnz) + len(tmpidxvalz)))
                validr = sp.lil_matrix((tensordata.shape[1] + tensordata.shape[2], len(tmpidxvalnz) + len(tmpidxvalz)))
                valido = sp.lil_matrix((tensordata.shape[1] + tensordata.shape[2], len(tmpidxvalnz) + len(tmpidxvalz)))
                outvalid = []
                ct = 0
                for j in tmpidxvalnz:
                    i = lnz[j]
                    validl[i[0], ct] = 1
                    validr[i[1], ct] = 1
                    valido[i[2] + tensordata.shape[1], ct] = 1
                    outvalid += [1]
                    ct += 1
                for j in tmpidxvalz:
                    i = lz[j]
                    validl[i[0], ct] = 1
                    validr[i[1], ct] = 1
                    valido[i[2] + tensordata.shape[1], ct] = 1
                    outvalid += [0]
                    ct += 1
                f = open(f'{self.data_path}/%s-valid-lhs-fold%s.pkl' % (dataset, k), 'wb')
                g = open(f'{self.data_path}/%s-valid-rhs-fold%s.pkl' % (dataset, k), 'wb')
                h = open(f'{self.data_path}/%s-valid-rel-fold%s.pkl' % (dataset, k), 'wb')
                l = open(f'{self.data_path}/%s-valid-targets-fold%s.pkl' % (dataset, k), 'wb')
                pickle.dump(validl.tocsr(), f)
                pickle.dump(validr.tocsr(), g)
                pickle.dump(valido.tocsr(), h)
                pickle.dump(np.asarray(outvalid), l)
                f.close()
                g.close()
                h.close()
                l.close()

                # Train data files
                # Pad the shorter list
                if len(tmpidxz) < len(tmpidxnz):
                    while len(tmpidxz) < len(tmpidxnz):
                        tmpidxz += tmpidxz[:len(tmpidxnz) - len(tmpidxz)]
                else:
                    while len(tmpidxnz) < len(tmpidxz):
                        tmpidxnz += tmpidxnz[:len(tmpidxz) - len(tmpidxnz)]

                ct = len(tmpidxz)
                if dataset == 'nations':
                    ct += len(lzfeat)
                trainposl = sp.lil_matrix((tensordata.shape[1] + tensordata.shape[2], ct))
                trainnegl = sp.lil_matrix((tensordata.shape[1] + tensordata.shape[2], ct))
                trainposr = sp.lil_matrix((tensordata.shape[1] + tensordata.shape[2], ct))
                trainnegr = sp.lil_matrix((tensordata.shape[1] + tensordata.shape[2], ct))
                trainposo = sp.lil_matrix((tensordata.shape[1] + tensordata.shape[2], ct))
                trainnego = sp.lil_matrix((tensordata.shape[1] + tensordata.shape[2], ct))
                ct = 0
                for u, v in zip(tmpidxnz, tmpidxz):
                    ipos = lnz[u]
                    ineg = lz[v]
                    trainposl[ipos[0], ct] = 1
                    trainnegl[ineg[0], ct] = 1
                    trainposr[ipos[1], ct] = 1
                    trainnegr[ineg[1], ct] = 1
                    trainposo[ipos[2] + tensordata.shape[1], ct] = 1
                    trainnego[ineg[2] + tensordata.shape[1], ct] = 1
                    ct += 1
                # Add all the feature triplets to each folds
                if dataset == 'nations':
                    for u, v in zip(lnzfeat, lzfeat):
                        ipos = u
                        ineg = v
                        trainposl[ipos[0], ct] = 1
                        trainnegl[ineg[0], ct] = 1
                        trainposr[ipos[1], ct] = 1
                        trainnegr[ineg[1], ct] = 1
                        trainposo[ipos[2] + tensordata.shape[1], ct] = 1
                        trainnego[ineg[2] + tensordata.shape[1], ct] = 1
                        ct += 1
                f = open(f'{self.data_path}/%s-train-pos-lhs-fold%s.pkl' % (dataset, k), 'wb')
                g = open(f'{self.data_path}/%s-train-pos-rhs-fold%s.pkl' % (dataset, k), 'wb')
                h = open(f'{self.data_path}/%s-train-pos-rel-fold%s.pkl' % (dataset, k), 'wb')
                l = open(f'{self.data_path}/%s-train-neg-lhs-fold%s.pkl' % (dataset, k), 'wb')
                m = open(f'{self.data_path}/%s-train-neg-rhs-fold%s.pkl' % (dataset, k), 'wb')
                n = open(f'{self.data_path}/%s-train-neg-rel-fold%s.pkl' % (dataset, k), 'wb')
                pickle.dump(trainposl.tocsr(), f)
                pickle.dump(trainposr.tocsr(), g)
                pickle.dump(trainposo.tocsr(), h)
                pickle.dump(trainnegl.tocsr(), l)
                pickle.dump(trainnegr.tocsr(), m)
                pickle.dump(trainnego.tocsr(), n)
                f.close()
                g.close()
                h.close()
                l.close()
                m.close()
                n.close()

    def __load_file(self, path):
        return sp.csr_matrix(pickle.load(open(path, 'rb')))

    def __convert2idx(self, spmat):
        rows, cols = spmat.nonzero()
        return rows[np.argsort(cols)]
    
    def __get_indexed_sequence(self, path: str) -> List[int]:
        return self.__convert2idx(self.__load_file(path))