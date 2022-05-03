import torch

torch.autograd.set_detect_anomaly(False)
torch.autograd.profiler.profile(False)
torch.autograd.profiler.emit_nvtx(False);

import argparse

from datasets.wordnet.wordnet import wordnet_preprocessed, get_geographical_data_from_wordnet
from datasets.tensor.tensor import tensor_datasets

from models.sme_linear import SME_lin
from models.sme_bilinear import SME_bil
from models.train_kit import train_kit

from plotter import reduce_embeddings_dimension, plot

# ================================================================

parser_main = argparse.ArgumentParser()
parser_main.add_argument('--energy', type=str, choices=['bilinear', 'linear'])   
parser_main.add_argument('--embedding-shape', nargs=2, type=int)


subparsers = parser_main.add_subparsers()
subparsers.required = True


parser_train = subparsers.add_parser('train-geo')
parser_train.add_argument('-save-path', type=str)
parser_train.add_argument('-epochs', type=int)
parser_train.add_argument('-batch-size', type=int)
parser_train.add_argument('-save-epochs', nargs='*', type=int)
parser_train.set_defaults(which='train-geo')


parser_plot = subparsers.add_parser('plot')
parser_plot.add_argument('-model-path', type=str)
parser_plot.add_argument('-name', type=str)
parser_plot.set_defaults(which='plot')


parser_train = subparsers.add_parser('train-tens')
parser_train.add_argument('-save-path', type=str)
parser_train.add_argument('-epochs', type=int)
parser_train.add_argument('-batch-size', type=int)
parser_train.set_defaults(which='train-tens')


args = parser_main.parse_args()

# ================================================================

if args.which in ['train-geo', 'plot']:
    
    wordnet = wordnet_preprocessed('./datasets/wordnet/.cache/wordnet-mlj12')
    print('> wordnet loaded.')


    region_names = ['Africa', 'Asia', 'Europe', 'North America', 'Central America', 'South America', 'Russian Federation']
    region_synsets = ['09189411', '09207288', '09275473', '09372504', '08735705', '09440400', '09006413']
    region_colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

    dataset, region_synsets, entities, relations, geographical_data = \
                                    get_geographical_data_from_wordnet(wordnet, region_names, region_synsets, region_colors)
    print('> geographical data loaded.')
    
    
    words_count = len(entities) + len(relations)
    
    model = None
    if args.energy == 'linear':
        model = SME_lin(words_count, *args.embedding_shape)
    else:
        model = SME_bil(words_count, *args.embedding_shape)
    
    if args.which == 'train-geo':
    
        optimizer = torch.optim.AdamW(model.parameters())
        trainer = train_kit(args.save_path, set(dataset))
        train_accuracies = trainer.train(model, optimizer, args.epochs, args.batch_size, save_epochs=args.save_epochs, write_log=False)
    
    elif args.which == 'plot':
        
        model.load_state_dict(torch.load(args.model_path))
        model.eval()
        
        x_vals, y_vals = reduce_embeddings_dimension(model.embed.weight.clone())        
        plot(args.name, wordnet, x_vals, y_vals, region_synsets, geographical_data) 
        
elif args.which in ['train-tens']:
    
    tensor_dataset_manager = tensor_datasets('./datasets/tensor/.cache/Tensor')
    print('> tensor datasets loaded.')
    
    dataset = tensor_dataset_manager.get('nations', 'train-pos') + tensor_dataset_manager.get('nations', 'valid')
    
    entities = set()       
    relations = set()

    for l, o, r in dataset:
        entities.add(l)
        relations.add(o)
        entities.add(r)
    
    words_count = len(entities) + len(relations)

    model = None
    if args.energy == 'linear':
        model = SME_lin(words_count, *args.embedding_shape)
    else:
        model = SME_bil(words_count, *args.embedding_shape)
        
    optimizer = torch.optim.AdamW(model.parameters())
    trainer = train_kit(args.save_path, set(dataset))
    train_accuracies = trainer.train(model, optimizer, args.epochs, args.batch_size, save_epochs=[], write_log=False)