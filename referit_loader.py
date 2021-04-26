# -*- coding: utf-8 -*-

"""
ReferIt, UNC, UNC+ and GRef referring image segmentation PyTorch dataset.

Define and group batches of images, segmentations and queries.
Based on:
https://github.com/chenxi116/TF-phrasecut-public/blob/master/build_batches.py
"""

import os
import sys
import cv2
from PIL import Image
import json
import uuid
import tqdm
import torch
import numpy as np
import os.path as osp
import scipy.io as sio
import scipy.sparse as sp
from referit import REFER
import torch.utils.data as data
from referit.refer import mask as cocomask

from nltk.parse.corenlp import CoreNLPDependencyParser
from nltk.parse.dependencygraph import DependencyGraph

import utils
from utils import Corpus

sys.modules['utils'] = utils

cv2.setNumThreads(0)

def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)

class DatasetNotFoundError(Exception):
    pass


class ReferDataset(data.Dataset):
    SUPPORTED_DATASETS = {
        'referit': {'splits': ('train', 'val', 'trainval', 'test')},
        'unc': {
            'splits': ('train', 'val', 'trainval', 'testA', 'testB'),
            'params': {'dataset': 'refcoco', 'split_by': 'unc'}
        },
        'unc+': {
            'splits': ('train', 'val', 'trainval', 'testA', 'testB'),
            'params': {'dataset': 'refcoco+', 'split_by': 'unc'}
        },
        'gref': {
            'splits': ('train', 'val'),
            'params': {'dataset': 'refcocog', 'split_by': 'google'}
        }
    }

    def __init__(self, data_root, split_root='data', dataset='referit',
                 transform=None, annotation_transform=None,
                 split='train', max_query_len=20,
                 parser_url=None):
        self.images = []
        self.data_root = data_root
        self.split_root = split_root
        self.dataset = dataset
        self.query_len = max_query_len
        self.corpus = Corpus()
        self.transform = transform
        self.annotation_transform = annotation_transform
        self.split = split
        if parser_url is not None:
            self.dep_parser = CoreNLPDependencyParser(url=parser_url)

        # Dataset: referit
        self.dataset_root = osp.join(self.data_root, 'referit')
        self.im_dir = osp.join(self.dataset_root, 'images')
        self.mask_dir = osp.join(self.dataset_root, 'mask')
        self.split_dir = osp.join(self.dataset_root, 'splits')
        self.parse_dir = osp.join(self.dataset_root, 'dep_graph');

        # Dataset: unc, unc+, gref
        if self.dataset != 'referit':
            self.dataset_root = osp.join(self.data_root, 'other')
            self.im_dir = osp.join(
                self.dataset_root, 'images', 'mscoco', 'images', 'train2014')
            self.mask_dir = osp.join(self.dataset_root, self.dataset, 'mask')
            self.parse_dir = osp.join(self.dataset_root, self.dataset, 'dep_graph')

        if parser_url is None:
            if not self.exists_dataset():
                raise ValueError("Not processed data found at {} .".format(osp.join(self.split_root, self.dataset)) +
                    "Please run `process_datasets.py` or download the processed data")
        else:
            self.process_dataset()


        dataset_path = osp.join(self.split_root, self.dataset)
        corpus_path = osp.join(dataset_path, 'corpus.pth')
        valid_splits = self.SUPPORTED_DATASETS[self.dataset]['splits']

        if split not in valid_splits:
            raise ValueError(
                'Dataset {0} does not have split {1}'.format(
                    self.dataset, split))

        self.corpus = torch.load(corpus_path)

        # read (image_name, mask_name, phrase)
        splits = [self.split]
        if self.dataset != 'referit':
            splits = ['train', 'val'] if self.split == 'trainval' else [self.split]
        for split in splits:
            imgset_file = '{0}_{1}.pth'.format(self.dataset, split)
            imgset_path = osp.join(dataset_path, imgset_file)
            self.images += torch.load(imgset_path)

    def exists_dataset(self):
        return osp.exists(osp.join(self.split_root, self.dataset))

    def process_dataset(self):
        if self.dataset not in self.SUPPORTED_DATASETS:
            raise DatasetNotFoundError(
                'Dataset {0} is not supported by this loader'.format(
                    self.dataset))

        dataset_folder = osp.join(self.split_root, self.dataset)
        if not osp.exists(dataset_folder):
            os.makedirs(dataset_folder)

        if self.dataset == 'referit':
            data_func = self.process_referit
        else:
            data_func = self.process_coco

        splits = self.SUPPORTED_DATASETS[self.dataset]['splits']

        for split in splits:
            print('Processing {0}: {1} set'.format(self.dataset, split))
            data_func(split, dataset_folder)

    def process_referit(self, setname, dataset_folder):
        split_dataset = []

        query_file = osp.join(
            self.dataset_root, 'referit',
            'referit_query_{0}.json'.format(setname))
        vocab_file = osp.join(self.split_dir, 'vocabulary_referit.txt')

        query_dict = json.load(open(query_file))
        im_list = query_dict.keys()

        if len(self.corpus) == 0:
            print('Saving dataset corpus dictionary...')
            corpus_file = osp.join(self.split_root, self.dataset, 'corpus.pth')
            self.corpus.load_file(vocab_file)
            torch.save(self.corpus, corpus_file)

        if not osp.exists(self.parse_dir):
            os.makedirs(self.parse_dir)

        for name in tqdm.tqdm(im_list):
            im_filename = name.split('_', 1)[0] + '.jpg'
            if im_filename in ['19579.jpg', '17975.jpg', '19575.jpg']:
                continue
            if osp.exists(osp.join(self.im_dir, im_filename)):
                mask_mat_filename = osp.join(self.mask_dir, name + '.mat')
                mask_png_filename = osp.join(self.mask_dir, name + '.png')
                if osp.exists(mask_mat_filename):
                    mask = sio.loadmat(mask_mat_filename)['segimg_t'] == 0
                    #mask = mask.astype(np.float64)
                    #mask = torch.from_numpy(mask)
                    #torch.save(mask, mask_pth_filename)
                    #os.remove(mask_mat_filename)
                    mask = mask.astype(np.uint8) * 255
                    cv2.imwrite(mask_png_filename, mask)
                for query_id, query in enumerate(query_dict[name]):
                    # save dependency parse
                    parse = next(self.dep_parser.raw_parse(query))
                    parse_file = '{}_{}.conll'.format(name, query_id)
                    parse_filename = osp.join(self.parse_dir, parse_file)
                    with open(parse_filename, 'w') as parse_conll:
                        parse_conll.write(parse.to_conll(4))
                    split_dataset.append((im_filename, name + '.png', parse_file, query))

        output_file = '{0}_{1}.pth'.format(self.dataset, setname)
        torch.save(split_dataset, osp.join(dataset_folder, output_file))

    def process_coco(self, setname, dataset_folder):
        split_dataset = []
        vocab_file = osp.join(self.split_dir, 'vocabulary_Gref.txt')

        refer = REFER(
            self.dataset_root, **(
                self.SUPPORTED_DATASETS[self.dataset]['params']))

        refs = [refer.refs[ref_id] for ref_id in refer.refs
                if refer.refs[ref_id]['split'] == setname]

        refs = sorted(refs, key=lambda x: x['file_name'])

        if len(self.corpus) == 0:
            print('Saving dataset corpus dictionary...')
            corpus_file = osp.join(self.split_root, self.dataset, 'corpus.pth')
            self.corpus.load_file(vocab_file)
            torch.save(self.corpus, corpus_file)

        if not osp.exists(self.mask_dir):
            os.makedirs(self.mask_dir)
        if not osp.exists(self.parse_dir):
            os.makedirs(self.parse_dir)

        for ref in tqdm.tqdm(refs):
            img_filename = 'COCO_train2014_{0}.jpg'.format(
                str(ref['image_id']).zfill(12))
            if osp.exists(osp.join(self.im_dir, img_filename)):
                h, w, _ = cv2.imread(osp.join(self.im_dir, img_filename)).shape
                seg = refer.anns[ref['ann_id']]['segmentation']
                rle = cocomask.frPyObjects(seg, h, w)

                # save as png (uint8) to save space
                mask = np.max(cocomask.decode(rle), axis=2).astype(np.uint8) * 255
                #mask = np.max(cocomask.decode(rle), axis=2).astype(np.float32)
                #mask = torch.from_numpy(mask)

                mask_file = str(ref['ann_id']) + '.png'
                mask_filename = osp.join(self.mask_dir, mask_file)
                if not osp.exists(mask_filename):
                    #torch.save(mask, mask_filename)
                    cv2.imwrite(mask_filename, mask)
                for sent_id, sentence in enumerate(ref['sentences']):
                    # save dependency parse
                    parse = next(self.dep_parser.raw_parse(sentence['sent']))
                    parse_file = '{}_{}.conll'.format(str(ref['ann_id']), sent_id)
                    parse_filename = osp.join(self.parse_dir, parse_file)
                    with open(parse_filename, 'w') as parse_conll:
                        parse_conll.write(parse.to_conll(4))
                    split_dataset.append((
                        img_filename, mask_file, parse_file, sentence['sent']))



        output_file = '{0}_{1}.pth'.format(self.dataset, setname)
        torch.save(split_dataset, osp.join(dataset_folder, output_file))

    def read_dep_parse_graph(self, parse_path):
        dep_graph = DependencyGraph.load(parse_path)[0]
        nodes = dep_graph.nodes
        time_steps = len(nodes) - 1 # remove (ROOT)
        words = []
        edges = []
        for i in range(time_steps):
            if nodes[i+1]['address'] is None:
                time_steps -= 1
            else:
                if nodes[i+1]['head'] != 0: # not (ROOT)'s child
                    edges.append([nodes[i+1]['head']-1, nodes[i+1]['address']-1])
                # self loop
                edges.append([nodes[i+1]['address']-1, nodes[i+1]['address']-1])
                words.append(nodes[i+1]['word'])
        edges = np.array(edges)
        # adjacent matrix
        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(time_steps, time_steps), dtype=np.float32)
        adj = np.array(adj.todense()) - np.eye(adj.shape[0]) # delete self loop
        return adj, words

    def pull_item(self, idx):
        img_file, mask_file, parse_file, _ = self.images[idx]
        parse_path = osp.join(self.parse_dir, parse_file)
        adj, phrase = self.read_dep_parse_graph(parse_path)

        img_path = osp.join(self.im_dir, img_file)
        img = Image.open(img_path).convert('RGB')

        mask_path = osp.join(self.mask_dir, mask_file)
        #mask = torch.load(mask_path)
        mask = np.array(Image.open(mask_path).convert('L')).astype(np.float32)
        if mask.max() > 127:
            #mask = mask / 255.0 # convert 255 to 1
            mask = (mask >= 127).astype(np.float32)
        mask = Image.fromarray(mask)
        return img, mask, phrase, adj

    def tokenize_adj(self, adj, words_len):
        adj = torch.FloatTensor(adj)
        if self.query_len > 0:
            if words_len > self.query_len:
                adj = adj[:self.query_len, :self.query_len]
            elif words_len < self.query_len: # pad
                new_adj = torch.zeros(self.query_len, self.query_len).float()
                new_adj[:words_len, :words_len] = adj
                adj = new_adj
        return adj

    def tokenize_phrase(self, phrase):
        return self.corpus.tokenize(phrase, self.query_len)

    def untokenize_word_vector(self, words):
        return self.corpus.dictionary[words]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img, mask, phrase, adj = self.pull_item(idx)
        if self.transform is not None:
            img = self.transform(img)
        if self.annotation_transform is not None:
            mask = self.annotation_transform(mask)
        words_len = len(phrase)
        phrase = self.tokenize_phrase(phrase)
        #_phrase = [_phrase]
        adj = self.tokenize_adj(adj, words_len)
        words_len = words_len if words_len < self.query_len else self.query_len
        return img, mask, phrase, adj, words_len

if __name__ == '__main__':
    refer = ReferDataset(data_root='/data0/yanpengxiang/dataset/refer',
                         dataset='unc+',
                         split_root='/data0/yanpengxiang/dataset/refer/data',
                         split='train')
