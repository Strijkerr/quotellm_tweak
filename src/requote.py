import argparse
import sys
from transformers import LogitsProcessorList, AutoModelForCausalLM, AutoTokenizer
from quote_utils import ExtractiveGeneration
import csv
import logging
import functools
import json


default_system_prompt = "We're going to find literal quotations that support a given paraphrase."
default_prompt_template = """## Example {n}. 

> {original}

Part of this text conveys the following: "{rephrased}"
This meaning is conveyed exclusively by certain parts of the original text:
{response}

"""


def main():

    argparser = argparse.ArgumentParser(description='Qsep')
    argparser.add_argument('file', nargs='?', type=argparse.FileType('r'), default=sys.stdin, help='Input file with pairs original,rephrased per line (csv); when omitted read from stdin.')
    argparser.add_argument('--prompt', required=False, type=argparse.FileType('r'), default=None, help='.jsonl file with system prompt, prompt template, and examples (keys original, rephrased (list), response)')
    argparser.add_argument('--model', required=False, default="unsloth/llama-3-70b-bnb-4bit", type=str)
    argparser.add_argument('--json', action='store_true', help='Whether to give json output; otherwise each question on a new line, with empty line per input.')
    argparser.add_argument('--temp', required=False, type=float, help='Temperature', default=None)
    argparser.add_argument('--topp', required=False, type=float, help='Sample only from top probability', default=None)
    argparser.add_argument('--topk', required=False, type=int, help='Top k top probability', default=None)
    argparser.add_argument('--beams', required=False, type=int, help='num_beams to search', default=None)
    argparser.add_argument('-v', '--verbose', action='store_true', help='To show debug messages.')
    args = argparser.parse_args()

    if args.beams:
        logging.warning('Beams may not work very well with constrained generation.')

    if not args.prompt:
        logging.warning('Are you sure you don\'t want to specify a custom --prompt, perhaps with few-shot examples?')

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    prompt_template = create_prompt_template(json.load(args.prompt) if args.prompt else None)

    logging.info(f'Prompt template: {prompt_template}')

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model)
    generate = functools.partial(model.generate, max_new_tokens=200, do_sample=args.temp is not None,
                                 num_beams=args.beams, temperature=args.temp, top_k=args.topk, top_p=args.topp)

    for n, (original_text, rephrased) in enumerate(csv.reader(args.file)):

        prompt = prompt_template.format(original=original_text, rephrased=rephrased)
        original_text = original_text.replace('"', '\"')  # to avoid JSON problems

        logging.debug(f'Input: {original_text} | {rephrased}')

        inputs = tokenizer.encode(prompt, return_tensors="pt").to('cuda')
        lp = ExtractiveGeneration(original_text, tokenizer, prompt_length=inputs.shape[-1])
        response = generate(inputs, logits_processor=LogitsProcessorList([lp]))
        result = tokenizer.decode(response[0, inputs.shape[-1]:], skip_special_tokens=True)

        try:    # just to make sure
            json.loads(result)
        except json.JSONDecodeError as e:
            logging.error(f'Response {n} is not valid JSON: {result} ({e})')

        print(result)

        # TODO: Maybe now retrieve the span start:end of the exact quotation parts?
        # try:
        #     result = retry_until_parse(pipe, chat_start, parser=parser, n_retries=args.retries, retry_hotter=args.retry_hotter)
        # except ValueError as e:
        #     logging.warning(f'Failed parsing response for input {n}; {e}')
        #     print()
        #     continue
        #
        # if args.json:
        #     print(json.dumps(result))
        # else:
        #     for res in result:
        #         print(res)
        # print()


def create_prompt_template(prompt_info: dict = None) -> str:
    if not prompt_info:
        prompt_lines = [default_system_prompt,
                        default_prompt_template.format(n='1', original='{original}', rephrased='{rephrased}', response='')]
    else:
        prompt_lines = [prompt_info['system_prompt']]
        n_example = 0
        for n_example, example in enumerate(prompt_info['examples'], start=1):
            example_prompt = prompt_info['prompt_template'].format(n=n_example, original=example['original'], rephrased=example['rephrased'], response=json.dumps(example['response']))
            prompt_lines.append(example_prompt)
        prompt_lines.append(prompt_info['prompt_template'].format(n=n_example+1, original='{original}', rephrased='{rephrased}', response=''))

    prompt_template = '\n'.join(prompt_lines)
    return prompt_template


if __name__ == '__main__':

    main()

