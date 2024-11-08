

import logging
import functools
import re
import typing

from transformers import LogitsProcessor
import torch
import math
import regex
import itertools

import string

class OptionGiverForMultiQuote:

    def __init__(self, original_ids, map_to_unspaced, start_ids, sep_ids, end_ids, empty_ids, start_quote_nospace):

        self.original = original_ids
        self.sep_ids = sep_ids[::-1]        # as stacks, to pop from the end
        self.start_ids = start_ids[::-1]
        self.end_ids = end_ids[::-1]
        self.start_quote_nospace = start_quote_nospace
        self.empty_ids = empty_ids[::-1] if empty_ids is not None else None
        self.allow_empty = self.empty_ids is not None

        self.map_to_unspaced = map_to_unspaced

        self.id_to_pos = {pos: id for pos, id in enumerate(self.original)}

        self.current_pos = None
        self.stack = None


    def next_iter(self, prefix):
        options = self.next(None)
        for i in prefix:
            options = self.next(i)
        return options


    def next(self, i=None):

        # now we process the token id i:
        if i is None:   # first function call, start of list
            # Only start from whole words
            self.current_pos = []
            # ensure spaceless start, for both json and non-json mode:
            self.stack = [([*self.map_to_unspaced.get(self.original[p], self.original[p])[::-1], *self.start_ids], p + 1) for p in range(len(self.original))]
            if self.allow_empty:
                self.stack.append(([*self.empty_ids], 0))
            self.current_pos = []

        elif i == self.sep_ids[-1]:    # end of quote
            # Only continue from whole words, starting two tokens to the right
            min_pos = min(self.current_pos) + 2
            self.current_pos = []
            if self.start_quote_nospace:   # ensure spaceless start, only for json mode
                self.stack = [([*self.map_to_unspaced.get(self.original[p], self.original[p])[::-1], *self.sep_ids[:-1]], p + 1) for p in range(min_pos, len(self.original))]  # expect spaceless subtokens instead
            else:
                self.stack = [(self.sep_ids[:-1], p) for p in range(min_pos, len(self.original))]

        elif i == self.end_ids[-1]:  # end of list
            self.current_pos = []
            self.stack = [(self.end_ids[:-1], None)]

        else:  # consume stacked or original token
            new_pos = []
            for p in self.current_pos:
                if self.original[p][0] == i:
                    self.stack.append(([*self.original[p][::-1]], p + 1))
            new_stack = []
            for s, p in self.stack:
                if s.pop() == i:
                    new_stack.append((s, p))
            self.current_pos = new_pos
            self.stack = new_stack

        # logging.debug(f'Parsed: {i}\n  stack: {self.stack}\n  pos: {self.current_pos}')

        # clean up empty stacks, adding their resultant positions to current_pos:
        if self.stack:
            self.current_pos.extend([p for s, p in self.stack if s == [] and p is not None])
            self.stack = [(s, p) for s, p in self.stack if s]

        # Now to list the options and some special symbols:
        options = [s[-1] for s, p in self.stack if s] + [self.original[p][0] for p in self.current_pos if p is not None and p < len(self.original)]

        if self.current_pos or not options and not i == self.sep_ids[0]:
            options.append(self.end_ids[-1])
        if self.current_pos and any(p + 2 < len(self.original) for p in self.current_pos) and i != self.sep_ids[0]:
            options.append(self.sep_ids[-1])

        return list(set(options))


class LogitsProcessorForMultiQuote(LogitsProcessor):

    def __init__(self, original: str, tokenizer, prompt_length: int, allow_empty: bool = False, json=False, sep='...') -> None:
        self.tokenizer = tokenizer
        self.use_json_mode = json
        self.prompt_length = prompt_length

        json_parts = get_json_part_ids(self.tokenizer)

        self.sep_ids = [*json_parts['comma'], *json_parts['next']] if json else self.tokenizer.encode(sep, add_special_tokens=False)
        self.start_ids = [*json_parts['start']] if json else []
        self.empty_ids = ([*json_parts['start_empty'], *json_parts['end_empty']] if json else []) if allow_empty else None
        self.end_ids = [*json_parts['end'], tokenizer.eos_token_id] if json else [tokenizer.eos_token_id]

        original_token_ids = tokenizer.encode(original, add_special_tokens=False)
        original_tokens = [tokenizer.decode(i, skip_special_tokens=True) for i in original_token_ids]

        self.original_token_ids_grouped = [[]]
        for i, t in zip(original_token_ids, original_tokens):
            if t.startswith(' ') or t in string.punctuation:
                self.original_token_ids_grouped.append([])
            self.original_token_ids_grouped[-1].append(i)

        self.original_token_ids_grouped = [tuple(t) for t in self.original_token_ids_grouped]

        self.map_spaced_to_unspaced = {}
        for t in self.original_token_ids_grouped:
            decoded = tokenizer.decode(t, skip_special_tokens=True)
            stripped = decoded.lstrip()
            new_t = tuple(tokenizer.encode(stripped, add_special_tokens=False))
            self.map_spaced_to_unspaced[t] = new_t


    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:

        beam_prefixes = input_ids[:, self.prompt_length:]
        for beam_n, prefix in enumerate(beam_prefixes):

            # TODO Caching; I should keep using the same QuoteParser for each beam
            option_giver = OptionGiverForMultiQuote(self.original_token_ids_grouped, self.map_spaced_to_unspaced,
                                                    self.start_ids, self.sep_ids, self.end_ids,
                                                    empty_ids=self.empty_ids, start_quote_nospace=self.use_json_mode)
            options = option_giver.next_iter(prefix)

            if not options:
                logging.warning(f'No options for next word! This shouldn\'t really happen... {self.tokenizer.decode(prefix)}')

            options_tensor = torch.tensor(options, dtype=torch.int, device=input_ids.device)
            mask = torch.isin(torch.arange(scores[beam_n].numel(), device=input_ids.device), options_tensor)
            scores[beam_n][~mask] = float("-inf")
        return scores


def get_json_part_ids(tokenizer):
    """
    Finds the token ids for a bunch of json delimiters, required for guaranteeing json output.

    E.g., for Llama3:
    start   '["'    # 1204
    end     '"]'    # 1365
    comma   '",'    # 498
    next    ' "'    # 330
    start_empty  '['
    end_empty   ']'
    """
    json_parts = {'start': '["', 'end': '"]', 'comma': '",', 'next': ' "', 'start_empty': '[', 'end_empty': ']'}
    json_part_ids = {}
    for key, json_part in json_parts.items():
        json_part_encoded = tokenizer.encode(json_part, add_special_tokens=False)
        json_part_ids[key] = json_part_encoded
    return json_part_ids


def find_spans_for_multiquote(original: str, multiquote: list[str], must_exist=True, must_unique=False) -> list[dict]:
    """
    >>> find_spans_for_multiquote("the quick brown fox jumped over the quick brown dog onto another fox", ["the quick", "fox"])
    [{'start': 0, 'end': 9, 'text': 'the quick'}, {'start': 16, 'end': 19, 'text': 'fox'}]
    >>> find_spans_for_multiquote("the quick brown fox jumped over the quick brown dog onto another fox", ["the quick brown", "fox"])
    [{'start': 32, 'end': 47, 'text': 'the quick brown'}, {'start': 65, 'end': 68, 'text': 'fox'}]
    >>> find_spans_for_multiquote("the quick brown fox jumped over the quick brown dog onto another fox", ["the quick brown fox"])
    [{'start': 0, 'end': 19, 'text': 'the quick brown fox'}]
    >>> find_spans_for_multiquote("the quick brown fox jumped over the quick brown dog onto another fox", ["the quick", "fox"], must_unique=True)
    Traceback (most recent call last):
    ...
    ValueError: No unique quote found.
    >>> find_spans_for_multiquote("the quick brown fox jumped over the quick brown dog onto another fox", ["the hairy duck"])
    Traceback (most recent call last):
    ...
    ValueError: No quote found.
    >>> find_spans_for_multiquote("the quick brown fox jumped over the quick brown dog onto another fox", ["the hairy duck"], must_exist=False)
    [{'start': None, 'end': None, 'text': 'the hairy duck'}]
    """
    matches = (re.finditer(re.escape(quote), original) for quote in multiquote)

    candidate_multimatches = []
    for multimatch in itertools.product(*matches):
        if any(m1.span()[0] >= m2.span()[0] for m1, m2 in zip(multimatch, multimatch[1:])):
            continue    # spans not in order
        if any(m1 != m2 and set(range(*m1.span())) & set(range(*m2.span())) for m1, m2 in itertools.product(multimatch, multimatch)):
            continue    # spans not disjoint
        if any(m1.span()[1] == m2.span()[0] - 1 for m1, m2 in zip(multimatch, multimatch[1:])):
            continue    # spans directly adjacent (not allowed by quotellm)
        candidate_multimatches.append(multimatch)

    if not candidate_multimatches:
        if must_exist:
            raise ValueError(f'No quote found: {multiquote}.')
        else:
            logging.warning(f'No quote found, returning dict with start/end None: {multiquote}')
            return [{'start': None, 'end': None, 'text': text} for text in multiquote]

    if len(candidate_multimatches) > 1:
        if must_unique:
            raise ValueError(f'No unique quote found: {multiquote}')
        else:
            logging.warning(f'No unique quote found, going with the first, shortest distance one: {multiquote}')

    closest_multimatch = min(candidate_multimatches, key=lambda multimatch: max(m.span()[1] for m in multimatch) - min(m.span()[0] for m in multimatch))
    return [{'start': m.span()[0], 'end': m.span()[1], 'text': m.group()} for m in closest_multimatch]



######### Below is old stuff no longer used, but perhaps useful as clumsy fallback option in the future... ########


def retry_until_parse(pipe, chat_start, parser, n_retries, fail_ok=False, increase_temp=.1):
    n_try = 0
    result = None
    errors = []
    logging.info(f'Prompt: {chat_start[-1]["content"]}'.replace('\n', '//'))
    while result is None and n_try < n_retries:
        n_try += 1
        raw = pipe([chat_start])[0][0]['generated_text'][-1]['content']
        logging.info(f'(Attempt {n_try}): Model says: {raw}'.replace('\n', '//'))
        pipe = functools.partial(pipe, temperature=pipe.keywords['temperature'] + increase_temp)
        try:
            result = parser(raw)
        except ValueError as e:
            errors.append(str(e))
            continue
        return result
    else:
        if not fail_ok:
            raise ValueError(f'Max number of retries ({"; ".join(errors)})')
        else:
            logging.warning(f'Max number of retries ({"; ".join(errors)})')
            return None


def parse_string_quote_as_spans(quote: str, original: str, fuzzy=0.0, already_used=None, only_from_char=0) -> list[dict]:
    """
    >>> parse_string_quote_as_spans('de grote ... was lui', 'de grote grijze vos was lui')
    [{'start': 0, 'end': 8, 'text': 'de grote'}, {'start': 20, 'end': 27, 'text': 'was lui'}]
    >>> parse_string_quote_as_spans('de grote ... was lui', 'de grooote grijze vos was lui', fuzzy=.2)
    [{'start': 0, 'end': 8, 'text': 'de grooo'}, {'start': 22, 'end': 29, 'text': 'was lui'}]
    >>> parse_string_quote_as_spans('def', 'abc def ghij abc def ghij', already_used=[])
    [{'start': 4, 'end': 7, 'text': 'def'}]
    >>> parse_string_quote_as_spans('def', 'abc def ghij abc def ghij', already_used=[(4, 7)])
    [{'start': 17, 'end': 20, 'text': 'def'}]
    >>> parse_string_quote_as_spans('And when?', 'What for? And why? And if so, when? And for whom will this be done?')
    Traceback (most recent call last):
    ValueError: No match for And when?
    >>> parse_string_quote_as_spans('And ... when?', 'What for? And why? And if so, when? And for whom will this be done?', fuzzy=0.2)
    Traceback (most recent call last):
    ValueError: Multiple matches for And ... when?
    >>> parse_string_quote_as_spans('Herinnert u zich mijn schriftelijke vragen over het weigeren van mannelijke artsen door gesluierde vrouwen?', 'Herinnert u zich mijn schriftelijke vragen over het weigeren van mannelijke artsen door gesluierde vrouwen? Herinnert u zich uw antwoord dat zoveel mogelijk recht moet worden gedaan aan de keuzevrijheid van de cliënt, maar dat er wel grenzen zijn? Kunt u aangeven waar deze grenzen liggen en waarop deze zijn gebaseerd? .', fuzzy=0.2)
    [{'start': 0, 'end': 107, 'text': 'Herinnert u zich mijn schriftelijke vragen over het weigeren van mannelijke artsen door gesluierde vrouwen?'}]
    """

    quote_regex = dotted_quote_to_regex(quote, fuzzy)
    matches = list(quote_regex.finditer(original))

    if not matches:
        raise ValueError(f'No match for {quote}')

    matches = [m for m in matches if m.span()[0] >= only_from_char]

    if not matches:
        raise ValueError(f'No match for {quote} from character {only_from_char}')
    elif len(matches) == 1:
        match = matches[0]
    elif len(matches) > 1:
        if already_used is None:
            raise ValueError(f'Multiple matches for {quote}')
        else:
            for match in matches:
                if match.span(0) not in already_used:
                    already_used.append(match.span(0))
                    break
            else:
                raise ValueError(f'Multiple matches for {quote}')

    spans = []
    for n in range(1, len(match.groups()) + 1):
        start, end = match.span(n)
        spans.append({'start': start, 'end': end, 'text': match.group(n)})

    return spans


def dotted_quote_to_regex(quote: str, fuzzy: float, fuzzy_min_e: int = 2, fuzzy_max_e: int = 7) -> regex.Regex:
    """
    Turn a quote string into a regular expression with optional fuzzy matching.
    Each part of the quote string is put in a regex capturing group.

    fuzzy_max_e: max number of characters to change (as fuzzy * len(quote) becomes too big); bigger can mean (very) slow.

    >>> dotted_quote_to_regex("The quick brown ... over the ... dog", .2)
    regex.Regex('(?:(The\\ quick\\ brown).+(over\\ the).+(dog)){e<=7}', flags=regex.B | regex.I | regex.V0)
    """
    quote_chunks = quote.split('...')
    clean_quote_chunks = [regex.escape(chunk.strip()) for chunk in quote_chunks]
    # make final punctuation marks optional (because LLM often adds them):
    regex_quote_chunks = [f'({chunk + ("?" if chunk[-1] in ".?!" else "")})' for chunk in clean_quote_chunks]
    the_regex_str = '(?:' + ('[^?]+'.join(regex_quote_chunks)) + ')'

    if fuzzy:
        n_chars = int(math.ceil(fuzzy * len(quote)))
        capped_nchars = max(fuzzy_min_e, min(n_chars, fuzzy_max_e))
        the_regex_str += f'{{e<={capped_nchars}}}'

    return regex.compile(the_regex_str, flags=regex.IGNORECASE + regex.BESTMATCH)


##### Below is some old stuff from another attempt, to simply enumerate all possible quotes for use with the outlines library



# def PydanticClassFactory(original):
#
#     # TODO: Or use enum?
#     class Component(BaseModel):
#         subquestion: str
#         # spans: typing.Literal[*make_possible_quotes(original)]
#         spans: typing.Union[*[tuple[*[typing.Literal[word] for word in quote]] for quote in make_possible_quotes(original)]]
#
#         # # Yields an error, but does not actually restrict the logits during generation.
#         # @model_validator(mode='after')
#         # def check_quote(self):
#         #     if not self.spans.split(' / '):   # TODO not finished
#         #         raise ValueError('quote is not a substring')
#         #     return self
#
#     class Components(BaseModel):
#         components: list[Component]
#
#     return Components


def all_sublists(s):
    return sorted(
        (s[start:end]
        for start in range(0, len(s))
        for end in range(start+1, len(s)+1)),
        key=len
    )


def add_discontinuous_sublists(lists):
    for l in lists:
        yield [l]
        for start in range(1, len(l)-1):
            for end in range(start+1, len(l)):
                yield [l[:start], l[end:]]


def make_possible_quotes(original):
    """
    >>> make_possible_quotes('test an original string')
    ['test', 'an', 'original', 'string', 'test an', 'an original', 'original string', 'test an original', 'test / original', 'an original string', 'an / string', 'test an original string', 'test / original string', 'test / string', 'test an / string']
    """
    # TODO: cubic explosion; also "Waarom wandelen mensen in Japan graag, maar in China niet?" is wel opgesplitst, maar niet de juiste spans... Probleem met komma?
    logging.info('Computing possible quotes...')
    multispans = add_discontinuous_sublists(all_sublists(original.split()))
    as_strings = [' / '.join(' '.join(n) for n in m) for m in multispans]
    logging.info(f'Done! {len(as_strings)}')
    return as_strings
