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
The meaning is conveyed exclusively by certain parts of the original text:
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

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    if not args.prompt:
        prompt_lines = [default_system_prompt,
                        default_prompt_template.format(n='{n}', original='{original}', rephrased='{rephrased}', response='')]
    else:
        prompt_info = json.load(args.prompt)
        prompt_lines = [prompt_info['system_prompt']]
        for n_example, example in enumerate(prompt_info['examples'], start=1):
            example_prompt = prompt_info['prompt_template'].format(n=n_example, original=example['original'], rephrased=example['rephrased'], response=json.dumps(example['response']))
            prompt_lines.append(example_prompt)
        prompt_lines.append(prompt_info['prompt_template'].format(n='{n}', original='{original}', rephrased='{rephrased}', response=''))

    prompt_template = '\n'.join(prompt_lines)

    logging.debug(f'Prompt template: {prompt_template}')

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    json_part_ids = get_json_part_ids(tokenizer)

    model = AutoModelForCausalLM.from_pretrained(args.model)
    generate = functools.partial(model.generate, max_new_tokens=200, do_sample=args.temp is not None,
                                 num_beams=args.beams, temperature=args.temp, top_k=args.topk, top_p=args.topp)


    for n, (original, rephrased) in enumerate(csv.reader(args.file)):

        prompt = prompt_template.format(n=n_example + 1, original=original, rephrased=rephrased)

        original = original.replace('"', '\"')  # to avoid JSON problems

        logging.debug(f'Prompt: {original} | {rephrased}')

        inputs = tokenizer.encode(prompt, return_tensors="pt").to('cuda')
        original_ids = tokenizer(original)["input_ids"]
        nospace_mapping = {id: tokenizer.encode(tokenizer.decode(id).lstrip())[1:] for id in original_ids}

        lp = ExtractiveGeneration(inputs.shape[-1], original_ids, tokenizer.eos_token_id, nospace_mapping=nospace_mapping, json_parts=json_part_ids)

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


def get_json_part_ids(tokenizer):
    """
    For Llama3:
    start   '["'    # 1204
    end     '"]'    # 1365
    comma   '",'    # 498
    next    ' "'    # 330
    """

    json_parts = {'start': '["', 'end': '"]', 'comma': '",', 'next': ' "'}
    json_part_ids = {}
    for key, json_part in json_parts.items():
        json_part_encoded = tokenizer.encode(json_part)
        if len(json_part_encoded) > 2:
            raise ValueError(f'Json part {json_part} is not a single token in this LLM.')   # TODO support this
        json_part_ids[key] = json_part_encoded[-1]
    return json_part_ids


if __name__ == '__main__':

    main()

