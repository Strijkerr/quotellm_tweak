

import logging
import functools
import re
import typing

from transformers import LogitsProcessor
import torch
import math
import regex
import itertools



class QuoteParser:

    def __init__(self, original_ids, map_to_unspaced, sep_ids, start_ids, end_ids, punctuation, json_mode):

        # TODO: use punctuation too

        self.original = original_ids
        self.sep_ids = sep_ids[::-1]        # as stacks, to pop from the end
        self.start_ids = start_ids[::-1]
        self.end_ids = end_ids[::-1]
        self.json = json_mode

        self.map_to_unspaced = map_to_unspaced

        self.id_to_pos = {pos: id for pos, id in enumerate(self.original)}
        self.word_start_pos = set(pos for pos, id in enumerate(self.original) if pos == 0 or id in self.map_to_unspaced)

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
            self.current_pos = [p for p in self.word_start_pos]
            # ensure spaceless start, for both json and non-json mode:
            self.stack = [(self.map_to_unspaced.get(self.original[p], [self.original[p]])[::-1] + self.start_ids, p + 1) for p in self.current_pos if p + 1 < len(self.original)]
            self.current_pos = []
            # TODO: This presupposes substitution can be done per-subtoken... Revise this later.

        elif i == self.sep_ids[-1]:    # end of quote
            # Only continue from whole words, starting two tokens to the right
            min_pos = min(self.current_pos) + 2
            self.current_pos = []
            if self.json:   # ensure spaceless start, only for json mode
                self.stack = [(self.map_to_unspaced.get(self.original[p], [self.original[p]])[::-1] + self.sep_ids[:-1], p + 1) for p in self.word_start_pos if p > min_pos]  # expect spaceless subtokens instead
                # TODO: See previous.
            else:
                self.stack = [(self.sep_ids[:-1], p) for p in self.word_start_pos if p > min_pos]

        elif i == self.end_ids[-1]:  # end of list
            self.current_pos = []
            self.stack = [(self.end_ids[:-1], None)]

        else:  # consume stacked or original token
            self.current_pos = [p + 1 for p in self.current_pos if p + 1 < len(self.original) and self.original[p] == i]
            self.stack = [(s, p) for s, p in self.stack if s.pop() == i]

        # clean up empty stacks, adding their resultant positions to current_pos:
        if self.stack:
            self.current_pos.extend([p for s, p in self.stack if s == []])
            self.stack = [(s, p) for s, p in self.stack if s]

        logging.debug(f'Parsed: {i}\n  stack: {self.stack}\n  pos: {self.current_pos} \n     ({len(self.stack), len(self.current_pos)}')

        options = [s[-1] for s, p in self.stack if s] + [self.original[p] for p in self.current_pos]
        options += [self.sep_ids[-1], self.end_ids[-1]]  # TODO: sep only if pre-final;   TODO maybe end only if unique-determining quote?

        return options


class LogitsProcessorForMultiQuote(LogitsProcessor):

    """
    This class and associated functions based on https://yonigottesman.github.io/2023/08/10/extractive-generative.html
    A list of LogitsProcessor instances can be passed into the transformers model.generate function, to filter/modify
    the logits prior to sampling for generation.
    """

    def __init__(self, original, tokenizer, prompt_length: int, allow_empty: bool = False, json=False, sep='...') -> None:
        self.prompt_length = prompt_length
        self.allow_empty = allow_empty
        self.original = ' ' + original  # To make sure all words, including the first, are preceded by a space.

        self.tokenizer = tokenizer
        self.eos_token_id = tokenizer.eos_token_id if isinstance(tokenizer.eos_token_id, list) else [tokenizer.eos_token_id]

        self.sep = self.tokenizer.encode(sep, add_special_tokens=False)
        self.json = json

        self.json_parts = get_json_part_ids(self.tokenizer)
        self.punctuation = get_punctuation_ids(self.tokenizer)

        self.sep_tokens = [self.json_parts['comma'], self.json_parts['next']] if json else self.sep
        self.special_tokens = self.json_parts.values() if json else self.sep
        self.new_quote_tokens = [self.json_parts['next'], self.json_parts['start']] if json else self.sep
        self.end_tokens = [self.json_parts['end']] if json else [self.eos_token_id[0]]

        self.original_token_ids = tokenizer(self.original)["input_ids"]
        # self.trie = create_trie(self.original_token_ids)
        self.map_to_unspaced_tokens = {}
        self.map_to_spaced_tokens = {}
        for id in self.original_token_ids:
            token = tokenizer.decode(id)
            token_stripped = token.lstrip()
            if token != token_stripped or id == self.original_token_ids[1]:  # also for first token, even if it isn't spaced
                self.map_to_unspaced_tokens[id] = tokenizer.encode(token_stripped, add_special_tokens=False)


    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """
        Takes input_ids and logit scores, returns modified/filtered logit scores based on options for next token.
        """
        beam_prefixes = input_ids[:, self.prompt_length:]
        for beam_n, prefix in enumerate(beam_prefixes):

            parser = QuoteParser(self.original_token_ids, self.map_to_unspaced_tokens, self.sep_tokens, self.json_parts['start'], self.json_parts['end'] + [self.eos_token_id[0]], self.punctuation)
            options = parser.next(None)
            for i in prefix:
                options = parser.next(i)

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


def get_punctuation_ids(tokenizer):
    punct = '.!?,:;'
    return [tokenizer.encode(p, add_special_tokens=False)[0] for p in punct]


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
