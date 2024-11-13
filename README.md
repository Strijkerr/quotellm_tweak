# QuoteLLM: Constrained generation allowing only literal quotes from some source text

## Example

This Python module and command-line tool lets you prompt a Large Language Model (LLM) to, given a source text and some extracted entity or piece of information, find the exact quote that supports that extraction.

For instance, the inputs could be (given as a pair in .csv format):

- **Original text:** Text generation is essential to many NLP tasks, such as open-ended text generation, summarization, translation, and more. Some of the models that can generate text include GPT2, XLNet, OpenAI GPT, CTRL, TransformerXL, XLM, Bart, T5, GIT, Whisper.
- **Extracted info:** Bart is a model that can generate text.

Output (JSON list of supporting quotes):

```json
[{"start": 122, "end": 170, "text": "Some of the models that can generate text include"}, {"start": 222, "end": 226, "text": "Bart"}]
```


## Motivation

Large Language Models aren't great at providing literal quotes. They can mess up word order and punctuation, and they cannot reliably count words or characters. Retrying the generation until a proper quote is generated is costly and not guaranteed, while employing 'fuzzy matching' find the corresponding exact quotes can lead to false positives.

A better approach is _constrained generation_: forcing the LLM to stick to spans from the original text. This turned out to be non-trivial with existing libraries (`transformers`, `outlines`). The current module, `QuoteLLM`, implements a basic solution, also supporting discontinuous spans (like the above example).

_Caveat 1: Currently tested only with `unsloth/llama-3-70b-bnb-4bit`, though any model should work whose tokenizer handles spaces in the same way (namely, as part of word-piece tokens)._

_Caveat 2: While this method guarantees that the LLM responds with exact substrings of the source text, they are currently not guaranteed to be 'unique' substrings._


## Install and basic use

```bash
pip install git+https://github.com/mwestera/quotellm
```

This makes available the `quotellm` module, from which you can import the class `LogitsProcessorForQuotes`. This class can be instantiated (for a source text to quote from, and a tokenizer), and fed into the `transformers` library's `model.generate` function via its `logits_processor` param. See `requote.py` for an example.

It also makes available the `requote` command-line interface, that takes pairs of source texts and extracted info as input, and outputs the exact supporting spans.


## Example of command-line interface `requote`:

Assuming a `.csv` file containing `original,extracted` pairs, and (optionally) a file `prompt.json` with keys `system_prompt`, `prompt_template` and `examples`, you can do:

```bash
$ requote example.csv --prompt prompt.json
```

This will use various reasonable default settings (see `requote --help`), and, for each input line, it will output a JSON list of extracted quotes, each quote a dictionary with `start`, `end` and `text`. 

The above example presupposes files like the following:

### `example.csv`

This file contains original texts, paired with some component extracted from it (in this specific example, a subquestion)

```csv
"Waarom en door wie zijn de horeca gesloten? Was u daarvan op de hoogte?","Waarom zijn de horeca gesloten?"
"Waarom en door wie zijn de horeca gesloten? Was u daarvan op de hoogte?","Door wie zijn de horeca gesloten?"
"Hoe komt het dat fotonen niets wegen en heel snel bewegen?","Hoe komt het dat fotonen niets wegen?"
"Hoe komt het dat fotonen niets wegen en heel snel bewegen?","Hoe komt het dat fotonen heel snel bewegen?"
```

### `prompt.json`

This JSON file specifies the system prompt, main prompt template (with keys `original`, `extracted` and `quotes`), and few-shot examples to be used.

```json
{"system_prompt":  "We're going to find literal quotations that support a given paraphrase, for the Dutch language.",
  "prompt_template":  "## Example {n}. \n> {original}\n\nThis text asks the question: \"{extracted}\"\nThe question is conveyed exclusively by certain parts of the original text:\n{quotes}\n",
  "examples": [
    {"original": "Sinds wanneer geldt deze maatregel en wat was destijds de motivatie (is deze openbaar)?", "extracted": "Wat was destijds de motivatie voor deze maatregel?", "quotes": ["wat was destijds de motivatie"]},
    {"original": "Heeft u de brief van de Indonesische overheid gelezen, en zoja, wat is uw reactie?", "extracted": "Wat is uw reactie op de brief van de Indonesische overheid?", "quotes": ["wat is uw reactie?"]},
    {"original": "Bent u het met mij eens dat dierenrecht en milieubescherming een prominentere plek moeten innemen in de samenleving?", "extracted": "Vindt u ook dat milieubescherming een prominentere plek in de samenleving moet innemen?", "quotes": ["Bent u het met mij eens dat", "milieubescherming een prominentere plek moeten innemen in de samenleving?"]},
  ]
}
```
