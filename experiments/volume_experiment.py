#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Saves features based on volume sensitivity

For usage information, call with --help.

Author: Vinod Subramanian
"""

from __future__ import print_function

import sys
import os
import io
from argparse import ArgumentParser
import json

import numpy as np
import torch
floatX = np.float32

from progress import progress
from simplecache import cached
import audio
from labels import create_aligned_targets
import model
import augment
import config

def change_volume(audio_data,spl=3):
    rms = (np.sqrt(np.mean(audio_data**2)))
    factor = np.power(10,spl/20.)
    return factor*audio_data
def opts_parser():
    descr = ("Computes predictions with a neural network trained for singing "
             "voice detection.")
    parser = ArgumentParser(description=descr)
    parser.add_argument('modelfile', metavar='MODELFILE',
            type=str,
            help='File to load the learned weights from (.npz format)')
    parser.add_argument('outfile', metavar='OUTFILE',
            type=str,
            help='File to save the prediction curves to (.npz format)')
    parser.add_argument('--dataset',
            type=str, default='jamendo',
            help='Name of the dataset to use (default: %(default)s)')
    parser.add_argument('--pitchshift', metavar='PERCENT',
            type=float, default=0.0,
            help='Perform test-time pitch-shifting of given amount and '
                 'direction in percent (e.g., -10 shifts down by 10%%).')
    parser.add_argument('--mem-use',
            type=str, choices=('high', 'mid', 'low'), default='mid',
            help='How much main memory to use. More memory allows a faster '
                 'implementation, applying the network as a fully-'
                 'convolutional net to longer excerpts or the full files. '
                 '(default: %(default)s)')
    parser.add_argument('--cache-spectra', metavar='DIR',
            type=str, default=None,
            help='Store spectra in the given directory (disabled by default)')
    parser.add_argument('--plot',
            action='store_true', default=False,
            help='If given, plot each spectrogram with predictions on screen.')
    parser.add_argument('--vars', metavar='FILE',
            action='append', type=str,
            default=[os.path.join(os.path.dirname(__file__), 'defaults.vars')],
            help='Reads configuration variables from a FILE of KEY=VALUE lines.'
            'Can be given multiple times, settings from later '
            'files overriding earlier ones. Will read defaults.vars, '
            'then files given here.')
    parser.add_argument('--var', metavar='KEY=VALUE',
            action='append', type=str,
            help='Set the configuration variable KEY to VALUE. Overrides '
            'settings from --vars options. Can be given multiple times') 
    return parser

def main():
    print(torch.cuda.is_available())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # parse command line
    parser = opts_parser()
    options = parser.parse_args()
    modelfile = options.modelfile

    cfg = {}
    for fn in options.vars:
        cfg.update(config.parse_config_file(fn))

    cfg.update(config.parse_variable_assignments(options.var))

    outfile = options.outfile
    sample_rate = cfg['sample_rate']
    frame_len = cfg['frame_len']
    fps = cfg['fps']
    mel_bands = cfg['mel_bands']
    mel_min = cfg['mel_min']
    mel_max = cfg['mel_max']
    blocklen = cfg['blocklen']
    batchsize = cfg['batchsize']
    
    bin_nyquist = frame_len // 2 + 1
    bin_mel_max = bin_nyquist * 2 * mel_max // sample_rate


    # prepare dataset
    print("Preparing data reading...")
    datadir = os.path.join(os.path.dirname(__file__),
                           os.path.pardir, 'datasets', options.dataset)

    # - load filelist
    with io.open(os.path.join(datadir, 'filelists', 'valid')) as f:
        filelist = [l.rstrip() for l in f if l.rstrip()]
    with io.open(os.path.join(datadir, 'filelists', 'test')) as f:
        filelist += [l.rstrip() for l in f if l.rstrip()]

    # - create generator for spectra
    spects = (cached(options.cache_spectra and
                     os.path.join(options.cache_spectra, fn + '.npy'),
                     audio.extract_spect,
                     os.path.join(datadir, 'audio', fn),
                     sample_rate, frame_len, fps)
              for fn in progress(filelist, 'File'))
    
    
    # - load and convert corresponding labels
    #print("Loading labels...")
    #labels = []
    #for fn, spect in zip(filelist, spects):
    #    fn = os.path.join(datadir, 'labels', fn.rsplit('.', 1)[0] + '.lab')
    #    with io.open(fn) as f:
    #        segments = [l.rstrip().split() for l in f if l.rstrip()]
    #    segments = [(float(start), float(end), label == 'sing')
    #                for start, end, label in segments]
    #    timestamps = np.arange(len(spect)) / float(fps)
    #    labels.append(create_aligned_targets(segments, timestamps, np.bool))


    
    #spects = list(spects) 
    filterbank = audio.create_mel_filterbank(sample_rate, frame_len, mel_bands,
                                             mel_min, mel_max)
    filterbank = filterbank[:bin_mel_max].astype(floatX)

    # - load mean/std
    meanstd_file = os.path.join(os.path.dirname(__file__),
                                '%s_meanstd.npz' % options.dataset)


    with np.load(meanstd_file) as f:
        mean = f['mean']
        std = f['std']
    mean = mean.astype(floatX)
    istd = np.reciprocal(std).astype(floatX)


     # - define generator for silence-padding
    pad = np.tile((np.log(1e-7)), (blocklen // 2, frame_len // 2 + 1))
    spects = (np.concatenate((pad, spect, pad), axis=0) for spect in spects)


    spects = augment.generate_in_background([spects], num_cached=1)
    
    

    mdl = model.CNNModel()
    mdl.load_state_dict(torch.load(modelfile))
    mdl.to(device)
    mdl.eval()
    mdl_test = torch.nn.ModuleList(mdl.children())
    spl_values = np.array([-9,-6,-3,3,6,9])
    count = 0
    metadata = []
    for fn,spect in zip(filelist,spects):
        fl = fn
        num_excerpts = len(spect) - blocklen + 1
        fn = os.path.join(datadir, 'labels', fn.rsplit('.', 1)[0] + '.lab')
        with io.open(fn) as f:
            segments = [l.rstrip().split() for l in f if l.rstrip()]
        segments = [(float(start), float(end), label == 'sing')
                    for start, end, label in segments]
        timestamps = np.arange(num_excerpts) / float(fps)
        label = (create_aligned_targets(segments, timestamps, np.bool))

        excerpts = np.lib.stride_tricks.as_strided(
                spect, shape=(num_excerpts, blocklen, spect.shape[1]),
                strides=(spect.strides[0], spect.strides[0], spect.strides[1]))
        
        num_iter = 0 
        while True:
            if(num_iter==15):
                print(fl)
                break
                
            # - Initialize dictionary and list to store the layer data 
            layer_data = {}
            out_list = []

            pos = np.random.choice(num_excerpts,replace=False)
           
            spect = np.squeeze(excerpts[pos:pos + 1,:,:])
            data = np.log(np.maximum(np.dot(spect[:, :bin_mel_max], filterbank),
                                1e-7))
            data = (data - mean) * istd
            data = data[np.newaxis,np.newaxis,:,:].astype(np.float32) 
            data = torch.from_numpy(data)

            for i, l in enumerate(mdl_test):
                for layer in l:
                    data = layer(data.to(device))
                    out_list.append(data.cpu().detach().numpy())
            layer_data['orig']= out_list

            pred_orig = data.cpu().detach().numpy()


            # - Compute labels adjascent to current excerpt
            if(pos>57 and pos<num_excerpts-57):
                (unique, counts) = np.unique(label[pos-57:pos+58], return_counts=True)
                frequencies = np.asarray((unique, counts)).T
            elif(pos<57):
                (unique, counts) = np.unique(label[:pos+58], return_counts=True)
                frequencies = np.asarray((unique, counts)).T
            else:
                (unique, counts) = np.unique(label[pos-57:], return_counts=True)
                frequencies = np.asarray((unique, counts)).T
            
            cond = False
            # - Iterate through different volume transformations
            for spl in spl_values:
                spect_change = change_volume(spect,spl=spl)
                data = np.log(np.maximum(np.dot(spect_change[:, :bin_mel_max],
                            filterbank),1e-7))
                data = (data - mean) * istd
                data = data[np.newaxis,np.newaxis,:,:].astype(np.float32)
            
                data = torch.from_numpy(data)
                out_list = []
                for i, l in enumerate(mdl_test):
                    for layer in l:
                        data = layer(data.to(device))
                        out_list.append(data.cpu().detach().numpy())
                    
                pred_volume = data.cpu().detach().numpy()
                cond += (np.abs(pred_volume-pred_orig)>0.3)
                layer_data[str(spl)] = out_list
            if(cond):
                num_iter += 1
                metadata.append((fl,pos,int(label[pos]),frequencies.tolist()))
                np.savez(os.path.join(outfile,fl+'-'+str(pos)+'.npz'),layer_data)  
    with open(os.path.join(outfile,'data.json'), 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=4)
    return
if __name__=='__main__':
    main()
