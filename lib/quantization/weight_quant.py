# From https://github.com/IST-DASLab/gptq/blob/main/llama.py
# Disable cpu offloading because of conflicts with transformers version (FIXME)

import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import transformers

from lib.gptq.gptq import *
from lib.gptq.modelutils import *
from lib.quantization.quantizer import *

import logging

@torch.no_grad()
def opt_sequential(model, dataloader, dev, args=None):
    logging.info('Starting GPTQ ...')

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.decoder.layers

    #model.model.embed_tokens = model.model.embed_tokens.to(dev)
    #model.model.norm = model.model.norm.to(dev)
    #layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.gptq_nsamples, args.gptq_seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    #layers[0] = layers[0].cpu()
    #model.model.embed_tokens = model.model.embed_tokens.cpu()
    #model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']

    quantizers = {}
    for i in range(len(layers)):
        #layer = layers[i].to(dev)
        layer = layers[i]
        full = find_layers(layer)

        if args.gptq_true_sequential:
            sequential = [
                ['self_attn.k_proj', 'self_attn.v_proj', 'self_attn.q_proj'],
                ['self_attn.out_proj'],
                ['mlp.fc1'],
                ['mlp.fc2']
            ]
        else:
            sequential = [list(full.keys())]
       
        for names in sequential:
            subset = {n: full[n] for n in names}

            gptq = {}
            for name in subset:
                gptq[name] = GPTQ(subset[name])
                gptq[name].quantizer = Quantizer()
                gptq[name].quantizer.configure(
                    args.bits_w, perchannel=True, sym=args.sym_w, mse=False
                )

            def add_batch(name):
                def tmp(_, inp, out):
                    gptq[name].add_batch(inp[0].data, out.data)
                return tmp
            handles = []
            for name in subset:
                handles.append(subset[name].register_forward_hook(add_batch(name)))
            for j in range(args.gptq_nsamples):
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
            for h in handles:
                h.remove()

            for name in subset:
                logging.info(f'Quantizing layer {i}: {name}')
                gptq[name].fasterquant(
                    percdamp=args.gptq_percdamp,
                    groupsize=args.groupsize_w,
                    actorder=args.gptq_act_order,
                    static_groups=args.gptq_static_groups,
                    bitflip_prob=args.w_bitflip_prob,
                    bitflip_bits=args.bits_w,
                )
                quantizers['model.decoder.layers.%d.%s' % (i, name)] = gptq[name].quantizer
                gptq[name].free()

        for j in range(args.gptq_nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]

        #layers[i] = layer.cpu()
        del layer
        del gptq 
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    
    return quantizers

@torch.no_grad()
def llama_sequential(model, dataloader, dev, args=None):
    logging.info('Starting GPTQ ...')

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    #model.model.embed_tokens = model.model.embed_tokens.to(dev)
    #model.model.norm = model.model.norm.to(dev)
    #layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.gptq_nsamples, args.gptq_seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    #layers[0] = layers[0].cpu()
    #model.model.embed_tokens = model.model.embed_tokens.cpu()
    #model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

    quantizers = {}
    for i in range(len(layers)):
        #layer = layers[i].to(dev)
        layer = layers[i]
        full = find_layers(layer)

        if args.gptq_true_sequential:
            sequential = [
                ['self_attn.k_proj', 'self_attn.v_proj', 'self_attn.q_proj'],
                ['self_attn.o_proj'],
                ['mlp.up_proj', 'mlp.gate_proj'],
                ['mlp.down_proj']
            ]
        else:
            sequential = [list(full.keys())]
       
        for names in sequential:
            subset = {n: full[n] for n in names}

            gptq = {}
            for name in subset:
                gptq[name] = GPTQ(subset[name])
                gptq[name].quantizer = Quantizer()
                gptq[name].quantizer.configure(
                    args.bits_w, perchannel=True, sym=args.sym_w, mse=False
                )

            def add_batch(name):
                def tmp(_, inp, out):
                    gptq[name].add_batch(inp[0].data, out.data)
                return tmp
            handles = []
            for name in subset:
                handles.append(subset[name].register_forward_hook(add_batch(name)))
            for j in range(args.gptq_nsamples):
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
            for h in handles:
                h.remove()

            for name in subset:
                logging.info(f'Quantizing layer {i}: {name}')
                gptq[name].fasterquant(
                    percdamp=args.gptq_percdamp,
                    groupsize=args.groupsize_w,
                    actorder=args.gptq_act_order,
                    static_groups=args.gptq_static_groups,
                    bitflip_prob=args.w_bitflip_prob,
                    bitflip_bits=args.bits_w,
                )
                quantizers['model.layers.%d.%s' % (i, name)] = gptq[name].quantizer
                gptq[name].free()

        for j in range(args.gptq_nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

        #layers[i] = layer.cpu()
        del layer
        del gptq 
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    
    return quantizers

def quantize_gptq(model, args, dev):
    from utils.data_utils import get_loaders
    dataloader = get_loaders(
        args.gptq_dataset, nsamples=args.gptq_nsamples,
        seed=args.seed, model=args.model_path,
        seqlen=args.gptq_seqlen, cache_dir=args.cache_dir,
    )
    if 'llama' in args.model_path:
        quantizers = llama_sequential(model, dataloader, dev, args)
    elif 'opt' in args.model_path:
        quantizers = opt_sequential(model, dataloader, dev, args)
    else:
        raise NotImplementedError


class HessianDiagCollector:

    def __init__(self, layer):
        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.columns = W.shape[1]
        self.hdiag = torch.zeros((self.columns,), device=self.dev)
        self.nsamples = 0

    def add_batch(self, inp):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear) or isinstance(self.layer, transformers.Conv1D):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()
        if isinstance(self.layer, nn.Conv2d):
            unfold = nn.Unfold(
                self.layer.kernel_size,
                dilation=self.layer.dilation,
                padding=self.layer.padding,
                stride=self.layer.stride
            )
            inp = unfold(inp)
            inp = inp.permute([1, 0, 2])
            inp = inp.flatten(1)

        self.hdiag *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        self.hdiag += torch.sum(inp * inp, dim=1)


@torch.no_grad()
def collect_rtn_hessian_diagonal(model, args, dev):
    from utils.data_utils import get_loaders

    dataloader = get_loaders(
        args.gptq_dataset, nsamples=args.gptq_nsamples,
        seed=args.seed, model=args.model_path,
        seqlen=args.gptq_seqlen, cache_dir=args.cache_dir,
    )
    logging.info(
        'Collecting RTN Hessian diagonal '
        f'(dataset={args.gptq_dataset}, nsamples={args.gptq_nsamples}, seqlen={args.gptq_seqlen})'
    )

    collectors = {
        layer_name: HessianDiagCollector(linear)
        for layer_name, linear in iter_quantized_linears(model, args)
    }

    def add_batch(layer_name):
        def tmp(_, inp, out):
            collectors[layer_name].add_batch(inp[0].data)
        return tmp

    handles = []
    for layer_name, linear in iter_quantized_linears(model, args):
        handles.append(linear.register_forward_hook(add_batch(layer_name)))

    use_cache = model.config.use_cache
    model.config.use_cache = False
    try:
        for batch in dataloader:
            model(batch[0].to(dev))
    finally:
        for h in handles:
            h.remove()
        model.config.use_cache = use_cache

    hessian_diagonal = {}
    missing = 0
    for layer_name, collector in collectors.items():
        if collector.nsamples == 0:
            missing += 1
            continue
        hessian_diagonal[layer_name] = collector.hdiag.detach().cpu()

    logging.info(
        f'Collected RTN Hessian diagonal for {len(hessian_diagonal)} layers '
        f'(missing={missing}).'
    )
    torch.cuda.empty_cache()
    return hessian_diagonal

def iter_quantized_linears(model, args):
    if 'llama' in args.model_path:
        layers = model.model.layers
        prefix = 'model.layers'
    elif 'opt' in args.model_path:
        layers = model.model.decoder.layers
        prefix = 'model.decoder.layers'
    else:
        raise NotImplementedError

    for i, layer in enumerate(layers):
        subset = find_layers(layer)
        for name, linear in subset.items():
            yield f'{prefix}.{i}.{name}', linear

@torch.no_grad()
def apply_weight_bitflip_with_pattern(model, args, pattern_path=None):
    if args.w_bitflip_prob <= 0:
        return

    cached_pattern = None
    if pattern_path is not None and Path(pattern_path).exists():
        pattern_blob = torch.load(pattern_path, map_location='cpu')
        if isinstance(pattern_blob, dict) and 'pattern' in pattern_blob:
            cached_pattern = pattern_blob['pattern']
        else:
            cached_pattern = pattern_blob
        logging.info(f'Loading bit-flip pattern from {pattern_path}')

    generated_pattern = {} if (pattern_path is not None and cached_pattern is None) else None
    total_flips = 0

    for layer_name, linear in iter_quantized_linears(model, args):
        W = linear.weight.data
        shape_ = W.shape
        quant_input = W
        if args.groupsize_w > 0:
            quant_input = quant_input.reshape(-1, args.groupsize_w)

        quantizer = Quantizer()
        quantizer.configure(
            args.bits_w, perchannel=True, sym=args.sym_w, mse=False
        )
        quantizer.find_params(quant_input, weight=True)
        qint = quantize_to_int(quant_input, quantizer.scale, quantizer.zero, quantizer.maxq)

        if cached_pattern is not None:
            layer_pattern = cached_pattern.get(layer_name, None)
            if layer_pattern is None:
                target_idx = torch.empty((0,), dtype=torch.long, device=qint.device)
                target_mask = torch.empty((0,), dtype=torch.int32, device=qint.device)
            else:
                target_idx = layer_pattern['target_idx'].to(qint.device, dtype=torch.long)
                if 'target_mask' in layer_pattern:
                    target_mask = layer_pattern['target_mask'].to(qint.device, dtype=torch.int32)
                elif 'bit_idx' in layer_pattern:  # backward compatibility
                    bit_idx = layer_pattern['bit_idx'].to(qint.device, dtype=torch.int32)
                    target_mask = torch.bitwise_left_shift(
                        torch.ones_like(bit_idx, dtype=torch.int32), bit_idx
                    )
                else:
                    target_mask = torch.empty((0,), dtype=torch.int32, device=qint.device)
        else:
            target_idx, target_mask = sample_bitflip_pattern(
                qint.numel(), args.bits_w, args.w_bitflip_prob, qint.device
            )
            if generated_pattern is not None:
                generated_pattern[layer_name] = {
                    'target_idx': target_idx.cpu().to(torch.int32),
                    'target_mask': target_mask.cpu().to(torch.int32),
                }

        total_flips += int(target_idx.numel())
        qint = apply_bitflip_pattern_int(qint, target_idx, target_mask, quantizer.maxq)
        qW = (quantizer.scale * (qint.to(quantizer.scale.dtype) - quantizer.zero))
        linear.weight.data = qW.reshape(shape_).to(linear.weight.data.dtype)

    logging.info(f'Applied weight bit-flips: total={total_flips}')

    if generated_pattern is not None:
        pattern_path = Path(pattern_path)
        pattern_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                '__meta__': {
                    'bitflip_prob': args.w_bitflip_prob,
                    'bits_w': args.bits_w,
                    'groupsize_w': args.groupsize_w,
                    'sym_w': bool(args.sym_w),
                    'seed': args.seed,
                },
                'pattern': generated_pattern,
            },
            pattern_path,
        )
        logging.info(f'Saved bit-flip pattern to {pattern_path}')

def quantize_nearest(model, args, dev, hessian_diagonal=None):
    if 'llama' in args.model_path:
        layers = model.model.layers
        prefix = 'model.layers'
    elif 'opt' in args.model_path:
        layers = model.model.decoder.layers
        prefix = 'model.decoder.layers'
    else:
        raise NotImplementedError

    use_act_order = bool(args.gptq_act_order and hessian_diagonal is not None)
    if args.gptq_act_order and hessian_diagonal is None:
        logging.warning(
            'gptq_act_order=True but Hessian diagonal is missing; RTN will run without act_order.'
        )

    quantizers = {}
    for i in range(len(layers)):
        logging.info(f'Quantizing layer {i}')
        #layer = layers[i].to(dev)
        layer = layers[i]

        subset = find_layers(layer)
        for name in subset:
            quantizer = Quantizer()
            quantizer.configure(
                args.bits_w, perchannel=True, sym=args.sym_w, mse=False
            )
            W = subset[name].weight.data
            shape_ = W.shape
            quant_input = W

            layer_name = f'{prefix}.{i}.{name}'
            perm = None
            invperm = None
            if use_act_order:
                hdiag = hessian_diagonal.get(layer_name, None)
                if hdiag is None:
                    logging.warning(f'No Hessian diagonal for {layer_name}; skipping act_order.')
                elif W.dim() != 2:
                    logging.warning(
                        f'RTN act_order currently supports 2D weights only: {layer_name} has shape {tuple(W.shape)}.'
                    )
                elif hdiag.numel() != W.shape[1]:
                    logging.warning(
                        f'Hessian diagonal mismatch for {layer_name}: '
                        f'expected {W.shape[1]}, got {hdiag.numel()}.'
                    )
                else:
                    perm = torch.argsort(hdiag.to(W.device), descending=True)
                    invperm = torch.argsort(perm)
                    quant_input = quant_input[:, perm]

            if args.groupsize_w > 0:
                quant_input = quant_input.reshape(-1, args.groupsize_w)

            quantizer.find_params(quant_input, weight=True)
            qW = quantize_with_int_bitflip(
                quant_input,
                quantizer.scale,
                quantizer.zero,
                quantizer.maxq,
                bitflip_prob=args.w_bitflip_prob,
                bitflip_bits=args.bits_w,
            ).to(next(iter(layer.parameters())).dtype)
            qW = qW.reshape(shape_)
            if invperm is not None:
                qW = qW[:, invperm]
            subset[name].weight.data = qW
            quantizers[f'{prefix}.{i}.{name}'] = {
                'maxq': quantizer.maxq.detach().cpu().clone(),
                'scale': quantizer.scale.detach().cpu().clone(),
                'zero': quantizer.zero.detach().cpu().clone(),
                'perm': perm.detach().cpu().clone() if use_act_order and hdiag is not None and invperm is not None else None,
                'invperm': invperm.detach().cpu().clone() if invperm is not None else None,
            }

    return quantizers
