import argparse
import sys
from transformers import LogitsProcessorList, AutoModelForCausalLM, AutoTokenizer
from quote_utils import LogitsProcessorForMultiQuote, find_spans_for_quote
import csv
import logging
import functools
import json


"""

CLI for matching paraphrases of parts of a text, to literal quotes in that text.

## Example:

$ cat example.csv | python requote.py -v --prompt prompt.json 

It will output a JSON list of strings per input line. This could work for instance based on the following files:

example.csv:

```
"Waarom en door wie zijn de horeca gesloten? Was u daarvan op de hoogte?","Waarom zijn de horeca gesloten?"
"Waarom en door wie zijn de horeca gesloten? Was u daarvan op de hoogte?","Door wie zijn de horeca gesloten?"
"Hoe komt het dat fotonen niets wegen en heel snel bewegen?","Hoe komt het dat fotonen niets wegen?"
"Hoe komt het dat fotonen niets wegen en heel snel bewegen?","Hoe komt het dat fotonen heel snel bewegen?"
"Waarom wandelen mensen in Japan graag, maar in Frankrijk niet?","Waarom wandelen mensen in Frankrijk niet graag?"
"Waarom wandelen mensen in Japan graag, en fietsen mensen in Frankrijk liever?","Waarom fietsen mensen in Frankrijk liever?"
"Bent u op de hoogte van het nieuwsbericht over hooligans? Zoja, wat is daarover uw mening?","Wat is uw mening daarover?"
```

prompt.json

```
{"system_prompt":  "We're going to find literal quotations that support a given paraphrase, for the Dutch language.",
  "prompt_template":  "## Example {n}. \n> {original}\n\nThis text asks the question: \"{rephrased}\"\nThe question is conveyed exclusively by certain parts of the original text:\n{response}\n",
  "examples": [
    {"original": "Sinds wanneer geldt deze maatregel en wat was destijds de motivatie (is deze openbaar)?", "rephrased": "Wat was destijds de motivatie voor deze maatregel?", "response": ["wat was destijds de motivatie"]},
    {"original": "Heeft u de brief van de Indonesische overheid gelezen, en zoja, wat is uw reactie?", "rephrased": "Wat is uw reactie op de brief van de Indonesische overheid?", "response": ["wat is uw reactie?"]},
    {"original": "Bent u het met mij eens dat dierenrecht en milieubescherming een prominentere plek moeten innemen in de samenleving?", "rephrased": "Vindt u ook dat milieubescherming een prominentere plek in de samenleving moet innemen?", "response": ["Bent u het met mij eens dat", "milieubescherming een prominentere plek moeten innemen in de samenleving?"]},
  ]
}
```

"""


default_system_prompt = "We're going to find literal quotations that support a given paraphrase."
default_prompt_template = """## Example {n}. 

> {original}

Part of this text conveys the following: "{rephrased}"
This meaning is conveyed exclusively by certain parts of the original text:
{response}

"""


def main():

    argparser = argparse.ArgumentParser(description='CLI for matching paraphrases of components of a text, to literal quotes in that text.')
    argparser.add_argument('file', nargs='?', type=argparse.FileType('r'), default=sys.stdin, help='Input file with pairs original,rephrased per line (csv); when omitted read from stdin.')
    argparser.add_argument('--prompt', required=False, type=argparse.FileType('r'), default=None, help='.jsonl file with system prompt, prompt template, and examples (keys original, rephrased (list), response)')
    argparser.add_argument('--noshortcut', action='store_true', help='To *not* bypass the LLM, even if the rephrase is already a literal quote.')
    argparser.add_argument('--model', required=False, default="unsloth/llama-3-70b-bnb-4bit", type=str)
    argparser.add_argument('--temp', required=False, type=float, help='Temperature', default=None)
    argparser.add_argument('--topp', required=False, type=float, help='Sample only from top p portion of probability distribution', default=None)
    argparser.add_argument('--topk', required=False, type=int, help='Sample only from top k tokens with max probability', default=None)
    argparser.add_argument('--beams', required=False, type=int, help='number of beams to search (does not work well with constrained generation)', default=1)
    argparser.add_argument('-v', '--verbose', action='store_true', help='To show debug messages.')
    args = argparser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    if args.model != 'unsloth/llama-3-70b-bnb-4bit':
        logging.warning('Not sure if it works with this model, but let\'s try! If it has a similar vocabulary, and big enough context window, it should be fine...')

    if args.beams > 1:
        logging.warning('Beams may not work very well with constrained generation.')

    if not args.prompt:
        logging.warning('Are you sure you don\'t want to specify a custom prompt .json file (--prompt), perhaps containing few-shot examples?')

    prompt_template = create_prompt_template(json.load(args.prompt) if args.prompt else None)

    logging.info(f'Prompt template: {prompt_template}')

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model)
    generate = functools.partial(model.generate, max_new_tokens=200, do_sample=args.temp is not None,
                                 num_beams=args.beams, temperature=args.temp, top_k=args.topk, top_p=args.topp)

    n_shortcuts_used = 0

    for n, (original_text, rephrased) in enumerate(csv.reader(args.file)):

        logging.debug(f'Original:  {original_text}')
        logging.debug(f'Rephrased: {rephrased}')

        if not args.noshortcut:
            try:
                result_with_spans = find_spans_for_quote(original_text.lower(), [rephrased.lower()], must_exist=True, must_unique=False)
            except ValueError:
                pass
            else:
                n_shortcuts_used += 1
                logging.info(f'Shortcut used!')
                print(json.dumps(result_with_spans))
                continue

        prompt = prompt_template.format(original=original_text, rephrased=rephrased)
        original_text = original_text.replace('"', '\"')  # to avoid JSON problems

        inputs = tokenizer.encode(prompt, return_tensors="pt").to('cuda')
        lp = LogitsProcessorForMultiQuote(original_text, tokenizer, prompt_length=inputs.shape[-1])
        response = generate(inputs, logits_processor=LogitsProcessorList([lp]))
        result_str = tokenizer.decode(response[0, inputs.shape[-1]:], skip_special_tokens=True)

        result_list = json.loads(result_str)
        result_with_spans = find_spans_for_quote(original_text, result_list, must_exist=True, must_unique=False)

        print(json.dumps(result_with_spans))

    if not args.noshortcut:
        logging.info(f'Shortcut used: {n_shortcuts_used}/{n}')


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

