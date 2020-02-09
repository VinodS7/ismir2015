#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Trains a neural network for singing voice detection.

For usage information, call with --help.

Author: Jan Schlüter
"""

from __future__ import print_function

import sys
import os
import io
from argparse import ArgumentParser

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
floatX = np.float32

from progress import progress
from simplecache import cached
import audio
import znorm
from labels import create_aligned_targets
import model
import augment
import config

def opts_parser():
    descr = "Trains a neural network for singing voice detection."
    parser = ArgumentParser(description=descr)
    parser.add_argument('modelfile', metavar='MODELFILE',
            type=str,
            help='File to save the learned weights to (.npz format)')
    parser.add_argument('--dataset',
            type=str, default='jamendo',
            help='Name of the dataset to use (default: %(default)s)')
    parser.add_argument('--augment',
            action='store_true', default=True,
            help='Perform train-time data augmentation (enabled by default)')
    parser.add_argument('--no-augment',
            action='store_false', dest='augment',
            help='Disable train-time data augmentation')
    parser.add_argument('--cache-spectra', metavar='DIR',
            type=str, default=None,
            help='Store spectra in the given directory (disabled by default)')
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
    parser.add_argument('--validate',
            action='store_true', default=False,
            help='Monitor validation loss')
    parser.add_argument('--no-validate',
            action='store_false', dest='validate',
            help='Disable monitoring validation loss')
    return parser

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # parse command line
    parser = opts_parser()
    options = parser.parse_args()
    modelfile = options.modelfile

    cfg = {}
    for fn in options.vars:
        cfg.update(config.parse_config_file(fn))

    cfg.update(config.parse_variable_assignments(options.var))
    
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
    datadir = os.path.join(os.path.dirname(__file__),
                           os.path.pardir, 'datasets', options.dataset)

    # - load filelist
    with io.open(os.path.join(datadir, 'filelists', 'train')) as f:
        filelist = [l.rstrip() for l in f if l.rstrip()]
    if options.validate:
        with io.open(os.path.join(datadir, 'filelists', 'valid')) as f:
            filelist_val = [l.strip() for l in f if l.strip()]
        filelist.extend(filelist_val)
    else:
        filelist_val = []
    # - compute spectra
    print("Computing%s spectra..." %
          (" or loading" if options.cache_spectra else ""))
    spects = []
    for fn in progress(filelist, 'File '):
        cache_fn = (options.cache_spectra and
                    os.path.join(options.cache_spectra, fn + '.npy'))
        spects.append(cached(cache_fn,
                             audio.extract_spect,
                             os.path.join(datadir, 'audio', fn),
                             sample_rate, frame_len, fps))

    # - load and convert corresponding labels
    print("Loading labels...")
    labels = []
    for fn, spect in zip(filelist, spects):
        fn = os.path.join(datadir, 'labels', fn.rsplit('.', 1)[0] + '.lab')
        with io.open(fn) as f:
            segments = [l.rstrip().split() for l in f if l.rstrip()]
        segments = [(float(start), float(end), label == 'sing')
                    for start, end, label in segments]
        timestamps = np.arange(len(spect)) / float(fps)
        labels.append(create_aligned_targets(segments, timestamps, np.bool))

    # - prepare mel filterbank
    filterbank = audio.create_mel_filterbank(sample_rate, frame_len, mel_bands,
                                             mel_min, mel_max)
    filterbank = filterbank[:bin_mel_max].astype(floatX)

    if options.validate:
        spects_val = spects[-len(filelist_val):]
        spects = spects[:-len(filelist_val)]
        labels_val = labels[-len(filelist_val):]
        labels = labels[:-len(filelist_val)]


    # - precompute mel spectra, if needed, otherwise just define a generator
    mel_spects = (np.log(np.maximum(np.dot(spect[:, :bin_mel_max], filterbank),
                                    1e-7))
                  for spect in spects)
   

    if not options.augment:
        mel_spects = list(mel_spects)
        del spects

    # - load mean/std or compute it, if not computed yet
    meanstd_file = os.path.join(os.path.dirname(__file__),
                                '%s_meanstd.npz' % options.dataset)
    try:
        with np.load(meanstd_file) as f:
            mean = f['mean']
            std = f['std']
    except (IOError, KeyError):
        print("Computing mean and standard deviation...")
        mean, std = znorm.compute_mean_std(mel_spects)
        np.savez(meanstd_file, mean=mean, std=std)
    mean = mean.astype(floatX)
    istd = np.reciprocal(std).astype(floatX)

    # - prepare training data generator
    print("Preparing training data feed...")
    if not options.augment:
        # Without augmentation, we just precompute the normalized mel spectra
        # and create a generator that returns mini-batches of random excerpts
        mel_spects = [(spect - mean) * istd for spect in mel_spects]
        batches = augment.grab_random_excerpts(
            mel_spects, labels, batchsize, blocklen)
    else:
        # For time stretching and pitch shifting, it pays off to preapply the
        # spline filter to each input spectrogram, so it does not need to be
        # applied to each mini-batch later.
        spline_order = cfg['spline_order']
        if spline_order > 1:
            from scipy.ndimage import spline_filter
            spects = [spline_filter(spect, spline_order).astype(floatX)
                      for spect in spects]

        # We define a function to create the mini-batch generator. This allows
        # us to easily create multiple generators for multithreading if needed.
        def create_datafeed(spects, labels):
            # With augmentation, as we want to apply random time-stretching,
            # we request longer excerpts than we finally need to return.
            max_stretch = cfg['max_stretch']
            batches = augment.grab_random_excerpts(
                    spects, labels, batchsize=batchsize,
                    frames=int(blocklen / (1 - max_stretch)))

            # We wrap the generator in another one that applies random time
            # stretching and pitch shifting, keeping a given number of frames
            # and bins only.
            max_shift = cfg['max_shift']
            batches = augment.apply_random_stretch_shift(
                    batches, max_stretch, max_shift,
                    keep_frames=blocklen, keep_bins=bin_mel_max,
                    order=spline_order, prefiltered=True)

            # We transform the excerpts to mel frequency and log magnitude.
            batches = augment.apply_filterbank(batches, filterbank)
            batches = augment.apply_logarithm(batches)

            # We apply random frequency filters
            max_db = cfg['max_db']
            batches = augment.apply_random_filters(batches, filterbank,
                                                   mel_max, max_db=max_db)

            # We apply normalization
            batches = augment.apply_znorm(batches, mean, istd)

            return batches

        # We start the mini-batch generator and augmenter in one or more
        # background threads or processes (unless disabled).
        bg_threads = cfg['bg_threads']
        bg_processes = cfg['bg_processes']
        if not bg_threads and not bg_processes:
            # no background processing: just create a single generator
            batches = create_datafeed(spects, labels)
        elif bg_threads:
            # multithreading: create a separate generator per thread
            batches = augment.generate_in_background(
                    [create_datafeed(spects, labels)
                     for _ in range(bg_threads)],
                    num_cached=bg_threads * 5)
        elif bg_processes:
            # multiprocessing: single generator is forked along with processes
            batches = augment.generate_in_background(
                    [create_datafeed(spects, labels)] * bg_processes,
                    num_cached=bg_processes * 25,
                    in_processes=True)

    ###########################################################################
    #-----------Main changes to code to make it work with pytorch-------------#
    ###########################################################################
    
    print("preparing training function...")
    mdl = model.CNNModel()
    print(mdl)
    return
    mdl = mdl.to(device)
    
    initial_eta = cfg['initial_eta']
    eta_decay = cfg['eta_decay']
    momentum = cfg['momentum']
    eta_decay_every = cfg.get('eta_decay_every', 1)
    eta = initial_eta
    print(cfg) 
    #set up loss
    criterion = torch.nn.BCELoss()

    #set up optimizer
    optimizer = torch.optim.SGD(mdl.parameters(),lr=eta,momentum=momentum,nesterov=True)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=eta_decay_every,gamma=eta_decay)

    #set up optimizer 
    writer = SummaryWriter()

    epochs = cfg['epochs']
    epochsize = cfg['epochsize']
    batches = iter(batches)

    for epoch in range(epochs):
        err = 0
        n_iter = 0 
        total_norm = 0
        loss_accum = 0
        for p in mdl.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        
        writer.add_scalar('Gradient norm',total_norm,epoch)
        for batch in progress(range(epochsize), min_delay=0.5,desc='Epoch %d/%d: Batch ' % (epoch+1, epochs)):
            data = next(batches)
            input_data = np.transpose(data[0][:,:,:,np.newaxis],(0,3,1,2))
            labels = data[1][:,np.newaxis].astype(np.float32)
            labels = (0.02 + 0.96*labels)
            optimizer.zero_grad()
            
            outputs = mdl(torch.from_numpy(input_data).to(device))
            loss = criterion(outputs, torch.from_numpy(labels).to(device))
            loss.backward()
            optimizer.step()
            writer.add_scalar('training loss per batch',loss.item())
            loss_accum += loss.item()
            n_iter+=1
        if options.validate:
            
            from eval import evaluate
            val_err = 0
            preds = []
            labs = []
            max_len = fps
            # - precompute mel spectra, if needed, otherwise just define a generator
            mel_spects_val = (np.log(np.maximum(np.dot(spect[:, :bin_mel_max], filterbank),
                                    1e-7))
                  for spect in spects_val)
            mel_spects_val = [(spect - mean) * istd for spect in mel_spects_val]

            num_iter = 0 

            for spect, label in zip(mel_spects_val, labels_val):
                num_excerpts = len(spect) - blocklen + 1
                excerpts = np.lib.stride_tricks.as_strided(
                    spect, shape=(num_excerpts, blocklen, spect.shape[1]),
                    strides=(spect.strides[0], spect.strides[0], spect.strides[1]))
                # - pass mini-batches through the network and concatenate results
                for pos in range(0, num_excerpts, batchsize):
                    input_data = np.transpose(excerpts[pos:pos + batchsize,:,:,np.newaxis],(0,3,1,2))
                    if (pos+batchsize>num_excerpts):
                        label_batch = label[blocklen//2+pos:blocklen//2+num_excerpts,np.newaxis].astype(np.float32)
                    else:
                        label_batch = label[blocklen//2+pos:blocklen//2+pos+batchsize,np.newaxis].astype(np.float32)
                    
                    pred = mdl(torch.from_numpy(input_data).to(device))
                    e = criterion(pred,torch.from_numpy(label_batch).to(device))
                    preds = np.append(preds,pred[:,0].cpu().detach().numpy())
                    labs = np.append(labs,label_batch)
                    val_err +=e.item()
                    num_iter+=1
           
            print("Validation loss: %.3f" % (val_err / num_iter))
            _, results = evaluate(preds,labs)
            print("Validation error: %.3f" % (1 - results['accuracy']))
            #if options.save_errors:
            #    errors.append(val_err / len(filelist_val))
            #    errors.append(1 - results['accuracy'])
        print('Training Loss per epoch', loss_accum/epochsize) 
        scheduler.step()
        #torch.save(mdl.state_dict(), os.path.join('test',modelfile))
        writer.add_scalar('avg loss per epoch',loss_accum/epochsize,epoch)
        writer.add_scalar('Validation loss', val_err/num_iter,epoch) 
        
    
if __name__=="__main__":
    main()

