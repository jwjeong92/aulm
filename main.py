import logging
import re
import contextvars
from contextlib import contextmanager
from pathlib import Path
import torch
import transformers

LOG_TAG = contextvars.ContextVar('log_tag', default='MAIN')

class TagFilter(logging.Filter):
    def filter(self, record):
        record.tag = LOG_TAG.get()
        return True

@contextmanager
def logging_tag(tag):
    token = LOG_TAG.set(tag)
    try:
        yield
    finally:
        LOG_TAG.reset(token)

def add_tag_filter(handler):
    handler.addFilter(TagFilter())
    return handler

def ensure_logfile_path(logfile):
    logfile_path = Path(logfile)
    logfile_path.parent.mkdir(parents=True, exist_ok=True)
    logfile_path.touch(exist_ok=True)
    return logfile_path

def setup_logging(logfile):
    handlers = [add_tag_filter(logging.StreamHandler())]
    if logfile != 'none':
        logfile_path = ensure_logfile_path(logfile)
        handlers.append(add_tag_filter(logging.FileHandler(logfile_path, mode='a')))
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(tag)s::%(levelname)s %(message)s',
        handlers=handlers,
        force=True,
    )

def get_gptq_checkpoint_path(args):
    model_tag = Path(args.model_path).name if args.model_path else 'model'
    raw_name = (
        f"{model_tag}_w{args.bits_w}_sym{int(args.sym_w)}_g{args.groupsize_w}_"
        f"ds{args.gptq_dataset}_n{args.gptq_nsamples}_sl{args.gptq_seqlen}_"
        f"ts{int(args.gptq_true_sequential)}_pd{args.gptq_percdamp}_"
        f"ao{int(args.gptq_act_order)}_sg{int(args.gptq_static_groups)}.pt"
    )
    safe_name = re.sub(r'[^A-Za-z0-9_.-]+', '_', raw_name)
    ckpt_dir = Path(args.gptq_ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    return ckpt_dir / safe_name

def try_load_gptq_checkpoint(model, args):
    if not args.gptq_ckpt:
        return False

    ckpt_path = get_gptq_checkpoint_path(args)
    if not ckpt_path.exists():
        return False

    logging.info(f'Loading GPTQ checkpoint from {ckpt_path}')
    state_dict = torch.load(ckpt_path, map_location='cpu')
    incompatible = model.load_state_dict(state_dict, strict=False)
    if incompatible.missing_keys:
        logging.warning(f'Missing keys while loading GPTQ checkpoint: {len(incompatible.missing_keys)}')
    if incompatible.unexpected_keys:
        logging.warning(f'Unexpected keys while loading GPTQ checkpoint: {len(incompatible.unexpected_keys)}')
    logging.info('Applied GPTQ checkpoint.')
    return True

def setup_gptq_log_handler(ckpt_path):
    gptq_log_path = ckpt_path.with_suffix('.log')
    formatter = logging.Formatter('%(asctime)s %(tag)s::%(levelname)s %(message)s')
    gptq_log_handler = add_tag_filter(logging.FileHandler(gptq_log_path, mode='w'))
    gptq_log_handler.setLevel(logging.INFO)
    gptq_log_handler.setFormatter(formatter)
    logging.getLogger().addHandler(gptq_log_handler)
    return gptq_log_handler, gptq_log_path

def main(args):
    
    # Loading Huggingface Model
    from utils.import_model import model_from_hf_path
    model = model_from_hf_path(
            args.model_path,
            args.use_cuda_graph,
            device_map='auto',
    ).eval()
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model_path
    )

    
    # Smooth Model
    if args.smoothquant:
        from lib.smoothquant.get_smooth_model import get_smoothquant_model
        get_smoothquant_model(model, tokenizer, args)

    fp_state_dict = dict()
    if args.analyze_stats or args.get_layerwise_distance: # Dump reference weight
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                fp_state_dict[name] = module.weight.data.cpu()

    # Quantization
    if args.bits_w < 16:
        from lib.quantization.weight_quant import (
            quantize_gptq,
            quantize_nearest,
        )
        if args.gptq:
            # Case: gptq-only
            with logging_tag('GPTQ'):
                loaded = try_load_gptq_checkpoint(model, args)
                if not loaded:
                    ckpt_path = None
                    gptq_log_handler = None
                    gptq_log_path = None
                    if args.gptq_ckpt:
                        ckpt_path = get_gptq_checkpoint_path(args)
                        gptq_log_handler, gptq_log_path = setup_gptq_log_handler(ckpt_path)
                        logging.info(f'Writing GPTQ log to {gptq_log_path}')
                    try:
                        quantize_gptq(model, args, dev='cuda')
                        logging.info("Applied GPTQ quantization.")
                        if args.gptq_ckpt:
                            torch.save(model.state_dict(), ckpt_path)
                            logging.info(f'Saved GPTQ checkpoint to {ckpt_path}')
                            logging.info(f'Saved GPTQ log to {gptq_log_path}')
                    finally:
                        if gptq_log_handler is not None:
                            logging.getLogger().removeHandler(gptq_log_handler)
                            gptq_log_handler.close()
        else:
            # Case: nearest-only
            quantize_nearest(model, args, dev='cuda')
            print("Applied nearest quantization.")

    # Activation Quantization
    if args.bits_a < 16 or args.analyze_stats: # Using custom Linear
        from lib.quantization.act_quant import add_act_quant
        add_act_quant(model, args)

    # Analysis Tool
    if args.analyze_stats:
        from utils.statistics import summarize_stats
        stats = summarize_stats(model, tokenizer, fp_state_dict, args)
        return
    if args.get_layerwise_distance:
        from utils.statistics import get_layerwise_distance
        stats = get_layerwise_distance(model, tokenizer, fp_state_dict, args)
        return

    # Inference (Chatbot, NIAH, Perplexity, LM-Eval)
    ppls = dict()
    results = dict()
    if args.chat:
        from utils.chatbot import chatbot_play
        chatbot_play(model, tokenizer, max_new_tokens=128, device='cuda')
    if args.niah:
        from utils.needle_in_a_haystack.needle_in_a_haystack_example import niah_example
        niah_example(model, tokenizer)
    if args.eval_ppl:
        from utils.perplexity import eval_ppl
        ppls = eval_ppl(model if args.llm_int8 else model.cuda(), tokenizer, args)
    if len(args.tasks) > 0:
        import lm_eval
        lm = lm_eval.models.huggingface.HFLM(
                pretrained=model,
                tokenizer=tokenizer,
                backend='causal',
                trust_remote_code=True,
            )
        results = lm_eval.evaluator.simple_evaluate(
            model=lm,
            tasks=args.tasks,
            num_fewshot=args.num_fewshot,
            limit=args.limit,
        )['results']
        logging.info(results)

    if args.logfile != 'none':
        # codex: logfile이 있는지 확인하고 없으면 새로 경로+파일까지 생성 후 append
        import json
        logfile_path = ensure_logfile_path(args.logfile)

        with open(logfile_path, 'a') as file:
            file.write(json.dumps(vars(args), indent=4) + '\n')
            file.write(json.dumps(ppls, indent=4) + '\n')
            file.write(json.dumps(results, indent=4) + '\n')
            file.write('\n')
    return

if __name__ == '__main__':
    import argparse
    from utils.common import *
    parser = argparse.ArgumentParser()
    
    # Model and Tasks
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--cache_dir', type=str, default='./cache')
    parser.add_argument('--tasks', type=str2list, default=[])
    parser.add_argument('--num_fewshot', type=str2int, default='none')
    parser.add_argument('--limit',type=str2int, default='none')
    parser.add_argument('--eval_ppl', type=str2bool, default=False)
    parser.add_argument('--eval_ppl_seqlen', type=int, default=2048)
    parser.add_argument('--use_cuda_graph', type=str2bool, default=False)
    parser.add_argument('--seed',type=int, default=0)

    # Quantization Configs
    parser.add_argument('--bits_a', type=int, default=16)
    parser.add_argument('--sym_a', type=str2bool, default=False)
    parser.add_argument('--groupsize_a', type=int, default=-1)
    parser.add_argument('--bits_w', type=int, default=4)
    parser.add_argument('--sym_w', type=str2bool, default=False)
    parser.add_argument('--groupsize_w', type=int, default=-1)
    # SmoothQuant Configs
    parser.add_argument('--llm_int8', type=str2bool, default=False)
    parser.add_argument('--smoothquant', type=str2bool, default=False)
    parser.add_argument('--smoothquant_alpha', type=float, default=0.5)
    parser.add_argument('--smoothquant_dataset', type=str, default='pile')
    parser.add_argument('--smoothquant_nsamples', type=int, default=512)
    parser.add_argument('--smoothquant_seqlen', type=int, default=512)
    # GPTQ Configs
    parser.add_argument('--gptq', type=str2bool, default=False)
    parser.add_argument('--gptq_dataset', type=str, default='c4')
    parser.add_argument('--gptq_nsamples', type=int, default=128)
    parser.add_argument('--gptq_seqlen', type=int, default=2048)
    parser.add_argument('--gptq_true_sequential', type=str2bool, default=False)
    parser.add_argument('--gptq_percdamp', type=float, default=.01)
    parser.add_argument('--gptq_act_order', type=str2bool, default=False)
    parser.add_argument('--gptq_static_groups', type=str2bool, default=False)
    parser.add_argument('--gptq_ckpt', type=str2bool, default=True)
    parser.add_argument('--gptq_ckpt_dir', type=str, default='./cache/gptq_model')

    # Others
    parser.add_argument('--chat', type=str2bool, default=False)
    parser.add_argument('--niah', type=str2bool, default=False)
    parser.add_argument('--logfile', type=str, default='./logs/dummy')
    # Analysis Tools
    parser.add_argument('--analyze_stats', type=str2bool, default=False)
    parser.add_argument('--stats_csv_path', type=str, default='./cache/stats.csv')
    parser.add_argument('--get_layerwise_distance', type=str2bool, default=False)

    args = parser.parse_args()
    setup_logging(args.logfile)
    set_seed(args.seed)
    logging.info(args)
    main(args)
