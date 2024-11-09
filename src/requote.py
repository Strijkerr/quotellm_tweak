import argparse
import sys

import torch.cuda
from transformers import LogitsProcessorList, AutoModelForCausalLM, AutoTokenizer
from quote_utils import LogitsProcessorForQuotes, find_spans_for_multiquote
import csv
import logging
import functools
import json
import itertools
import numpy
import gc

MAX_TOKENS = 500
DEFAULT_PROMPT_INFO = {
    'system_prompt': "We're going to find literal quotations that support a specific, extracted meaning component.",
    'prompt_template': """## Example {n}.
Original text: "{original}"
Extracted meaning component, rephrased: "{rephrased}"
The meaning component is conveyed by (literal quoted spans): {response}""",
    'examples': [], # TODO add default examples
}


def main():

    argparser = argparse.ArgumentParser(description='CLI for matching paraphrases of components of a text, to literal quotes in that text.')
    argparser.add_argument('file', nargs='?', type=argparse.FileType('r'), default=sys.stdin, help='Input file with pairs original,rephrased per line (csv); when omitted read from stdin.')

    # Prompting style params:
    argparser.add_argument('--prompt', required=False, type=argparse.FileType('r'), default=None, help='.jsonl file with system prompt, prompt template, and examples (keys original, rephrased (list), response)')
    argparser.add_argument('--json', action='store_true', help='To force model to output json list of strings; otherwise strings separated by --sep')
    argparser.add_argument('--sep', required=False, type=str, default='...', help='If not --json, this is the separator string to be used.')

    # Transformer params:
    argparser.add_argument('--model', required=False, default="unsloth/llama-3-70b-bnb-4bit", type=str)  # test: xiaodongguaAIGC/llama-3-debug
    argparser.add_argument('--temp', required=False, type=float, help='Temperature', default=None)
    argparser.add_argument('--topp', required=False, type=float, help='Sample only from top p portion of probability distribution', default=None)
    argparser.add_argument('--topk', required=False, type=int, help='Sample only from top k tokens with max probability', default=None)
    argparser.add_argument('--beams', required=False, type=int, help='number of beams to search (does not work well with constrained generation)', default=1)
    argparser.add_argument('--quote-verbosity', required=False, type=float, help='Only used if --beams > 1. Positive to favor longer quotes, negative to favor shorter ones.', default=0.0)

    argparser.add_argument('-v', '--verbose', action='store_true', help='To show debug messages.')
    args = argparser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG, format='')

    logging.info(json.dumps({k: v for k, v in args.__dict__.items() if k not in ['file', 'prompt']}, indent='  '))

    if args.model != 'unsloth/llama-3-70b-bnb-4bit':
        logging.warning('Not sure if it works with this model, but let\'s try! If it has a similar vocabulary, and big enough context window, it should be fine...')

    if not args.prompt:
        logging.warning('Are you sure you don\'t want to specify a custom prompt .json file (--prompt), perhaps containing few-shot examples?')

    prompt_template = create_prompt_template(**(json.load(args.prompt) if args.prompt else DEFAULT_PROMPT_INFO), json_format=args.json, sep=args.sep)

    logging.info(f'Prompt template: {prompt_template}')

    tokenizer = AutoTokenizer.from_pretrained(args.model, clean_up_tokenization_spaces=False)  # clean up changes e.g. " ." to "."
    model = AutoModelForCausalLM.from_pretrained(args.model)    # no explicit to(cuda) for a quantized model
    generator = functools.partial(model.generate, max_new_tokens=MAX_TOKENS, do_sample=args.temp is not None,
                                  num_beams=args.beams, temperature=args.temp, top_k=args.topk, top_p=args.topp,
                                  length_penalty=args.quote_verbosity)
    do_prompt = functools.partial(prompt_for_quote,
                                  generator=generator,
                                  tokenizer=tokenizer,
                                  prompt_template=prompt_template,
                                  force_json=args.json,
                                  sep=args.sep)

    stats_keeper = []

    for n, (original_text, rephrased) in enumerate(csv.reader(args.file)):
        logging.debug(f'----- {n} -----')
        logging.debug(f'Original:  {original_text}')
        logging.debug(f'Rephrased: {rephrased}')

        # check if we have a good match already, no need for LLM:
        try:
            result_with_spans = find_spans_for_multiquote(
                original_text.lower(), [rephrased.lower()], must_exist=True, must_unique=False
            )
        except ValueError:
            pass
        else:
            stats_keeper.append(stats_to_record(original_text, rephrased, result_with_spans, shortcut_used=True))
            logging.debug(f'Shortcut used!')
            print(json.dumps(result_with_spans))
            continue

        # Otherwise, use LLM to figure it out:
        result_with_spans = do_prompt(original_text=original_text, rephrased=rephrased)
        stats_keeper.append(stats_to_record(original_text, rephrased, result_with_spans))
        print(json.dumps(result_with_spans))

    log_stats_summary(stats_keeper)


def prompt_for_quote(generator, tokenizer, prompt_template, original_text, rephrased, force_json=False, sep='...') -> list[dict]:
    """
    Kinda large, but convenient wrapper function.

    Fills the prompt template with original text and rephrased, applies the tokenizer to it, feeds it into
    the generator, with constrained generation using either json or sep.

    Returns a list of spans (dictionaries) with keys start, end, and text.
    """

    prompt = prompt_template.format(original=original_text, rephrased=rephrased)
    original_text = original_text.replace('"', '\"')  # to avoid JSON problems

    inputs = tokenizer.encode(prompt, return_tensors="pt").to('cuda' if torch.cuda.is_available() else 'cpu')
    lp = LogitsProcessorForQuotes(original_text, tokenizer, prompt_length=inputs.shape[-1],
                                  force_json_response=force_json, sep=sep)

    try:
        response = generator(inputs, logits_processor=LogitsProcessorList([lp]))
    except torch.cuda.OutOfMemoryError:
        logging.warning('CUDA out of memory: Retry cuda.empty_cache() and use_cache=False.')
        gc.collect()
        torch.cuda.empty_cache()
        # try:
        #     response = generator(inputs, logits_processor=LogitsProcessorList([lp]), cache_implementation="offloaded")
        # except torch.cuda.OutOfMemoryError:
        #     logging.warning('CUDA out of memory again: Retrying with cache disabled.')
        response = generator(inputs, logits_processor=LogitsProcessorList([lp]), use_cache=False)


    result_str = tokenizer.decode(response[0, inputs.shape[-1]:], skip_special_tokens=True)

    # Some edge cases:
    if len(response) >= MAX_TOKENS:
        logging.warning('Response at MAX_TOKENS, a currently hard-coded limit (but feel free to edit the source); '
                        'this should not really happen and means the response is probably truncated/bad.')
    if force_json and not result_str.endswith('"]'):
        logging.warning('Truncated JSON response string; appending "], but the response is probably bad.')
        if not result_str.endswith('"'):
            result_str += '"'
        result_str += ']'

    logging.debug(f'Response: {result_str}')

    result_list = json.loads(result_str) if force_json else [s.strip() for s in result_str.split(sep)]
    result_with_spans = find_spans_for_multiquote(original_text, result_list, must_exist=True, must_unique=False)
    return result_with_spans


def create_prompt_template(system_prompt: str, prompt_template: str, examples: list[dict], json_format=False, sep='...') -> str:
    """
    Creates the prompt template, consisting of system prompt, few-shot examples (if any), and the actual prompt.
    The final prompt will have remaining slots {original} and {rephrased}, to be filled by format.

    :param json_format: to use json format: `["the quick", "fox jumped"]`
    :param sep: if not json format, uses sep (default ...) to separate quotes: `the quick... fox jumped`
    :return: A string with slots {original} and {rephrased}.
    """
    prompt_lines = [system_prompt]
    n_example = 0
    for n_example, example in enumerate(examples, start=1):
        response_str = json.dumps(example['response']) if json_format else f'{sep} '.join(example['response'])
        example_prompt = prompt_template.format(
            n=n_example, original=example['original'],
            rephrased=example['rephrased'], response=response_str
        ).replace('{', '{{').replace('}', '}}')  # to make them survive format string
        prompt_lines.append(example_prompt)
    prompt_lines.append(
        prompt_template.format(n=n_example+1, original='{original}', rephrased='{rephrased}', response='')
    )

    full_prompt_template = '\n\n'.join(prompt_lines)
    return full_prompt_template


def stats_to_record(original_text, rephrased, spans, shortcut_used=False):
    span_length = len(' '.join(span['text'] for span in spans))
    return {
        'n_spans': len(spans),
        'span_length_abs': span_length,
        'span_length_rel': span_length / len(original_text),
        'span_length_rel_rephrased': span_length / len(rephrased),
        'shortcut': shortcut_used,
    }


def log_stats_summary(stats_keeper: list[dict]) -> None:
    stats_keeper_noshortcut = [s for s in stats_keeper if not s['shortcut']]
    logging.info('========================')
    logging.info('subset,stat,mean,std')

    for stats_keeper, label in [(stats_keeper, 'all'), (stats_keeper_noshortcut, 'llm')]:
        stats_lists: dict[str, list] = {k: [dic[k] for dic in stats_keeper] for k in stats_keeper[0]}

        for key, stats_list in stats_lists.items():
            logging.info(f'{label},{key},{numpy.mean(stats_list):.2f},{numpy.std(stats_list):.2f}')
    logging.info('========================')


if __name__ == '__main__':

    main()

