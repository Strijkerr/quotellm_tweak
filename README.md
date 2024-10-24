# QuoteLLM: forcing LLMs to answer your prompt only with literal quotes from some source text

## Install and basic use

```bash
pip install git+https://github.com/mwestera/quotellm
```

This makes available the `quotellm` module, from which you can import the class `LogitsProcessorForMultiQuote`. This class can be instantiated (for a source text and a tokenizer), and fed into the `transformers` library's `model.generate` function via its `logits_processor` param. See `requote.py` for an example.

It also makes available the `requote` command-line interface.

_Caveat: Currently built for use with `unsloth/llama-3-70b-bnb-4bit`, though any model with the same tokenization strategy should work. Other models might break, due to the rather crude way in which JSON output is currently guaranteed._

## Example of command-line interface `requote`:

Assuming a `.csv` file containing `original,rephrased` pairs, and a file `prompt.json` with keys `system_prompt`, `prompt_template` and `examples`, you can do:

```bash
$ requote example.csv --prompt prompt.json
```

This will output, for each input line, a JSON list of extracted quotes. 

The above example presupposes files like the following:

### `example.csv`

```csv
"Waarom en door wie zijn de horeca gesloten? Was u daarvan op de hoogte?","Waarom zijn de horeca gesloten?"
"Waarom en door wie zijn de horeca gesloten? Was u daarvan op de hoogte?","Door wie zijn de horeca gesloten?"
"Hoe komt het dat fotonen niets wegen en heel snel bewegen?","Hoe komt het dat fotonen niets wegen?"
"Hoe komt het dat fotonen niets wegen en heel snel bewegen?","Hoe komt het dat fotonen heel snel bewegen?"
"Waarom wandelen mensen in Japan graag, maar in Frankrijk niet?","Waarom wandelen mensen in Frankrijk niet graag?"
"Waarom wandelen mensen in Japan graag, en fietsen mensen in Frankrijk liever?","Waarom fietsen mensen in Frankrijk liever?"
"Bent u op de hoogte van het nieuwsbericht over hooligans? Zoja, wat is daarover uw mening?","Wat is uw mening daarover?"
```

### `prompt.json`

```json
{"system_prompt":  "We're going to find literal quotations that support a given paraphrase, for the Dutch language.",
  "prompt_template":  "## Example {n}. \n> {original}\n\nThis text asks the question: \"{rephrased}\"\nThe question is conveyed exclusively by certain parts of the original text:\n{response}\n",
  "examples": [
    {"original": "Sinds wanneer geldt deze maatregel en wat was destijds de motivatie (is deze openbaar)?", "rephrased": "Wat was destijds de motivatie voor deze maatregel?", "response": ["wat was destijds de motivatie"]},
    {"original": "Heeft u de brief van de Indonesische overheid gelezen, en zoja, wat is uw reactie?", "rephrased": "Wat is uw reactie op de brief van de Indonesische overheid?", "response": ["wat is uw reactie?"]},
    {"original": "Bent u het met mij eens dat dierenrecht en milieubescherming een prominentere plek moeten innemen in de samenleving?", "rephrased": "Vindt u ook dat milieubescherming een prominentere plek in de samenleving moet innemen?", "response": ["Bent u het met mij eens dat", "milieubescherming een prominentere plek moeten innemen in de samenleving?"]},
  ]
}
```