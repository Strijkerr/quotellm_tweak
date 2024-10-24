

import logging
import functools
import typing

from transformers import LogitsProcessor
import torch
import math
import regex
import itertools


class ExtractiveGeneration(LogitsProcessor):

    """
    This class and associated functions from https://yonigottesman.github.io/2023/08/10/extractive-generative.html
    """

    def __init__(self, input_start_len: int, context_tokens: list[int], eos_token_id: int | list[int], nospace_mapping: dict = None, allow_empty=False, discontinuous_token_id: int = None, json_parts: dict = None) -> None:
        """
        For llama3, discontinuous_token_id // is 443
        """
        self.trie = create_trie(context_tokens)
        self.input_start_len = input_start_len
        self.nospace_mapping = nospace_mapping
        self.eos_token_id = eos_token_id
        self.allow_empty = allow_empty
        self.discontinuous_token_id = discontinuous_token_id
        self.json_parts = json_parts
        if not isinstance(self.eos_token_id, list):
            self.eos_token_id = [self.eos_token_id]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        beam_prefixes = input_ids[:, self.input_start_len:]
        for i, prefix in enumerate(beam_prefixes):
            prefix_list = prefix.tolist()
            if not prefix_list:
                options = [self.json_parts['start']]
            elif prefix_list[-1] == self.json_parts['end']:
                options = self.eos_token_id
            elif prefix_list[-1] == self.json_parts['comma']:
                options = [self.json_parts['next']]
            else:
                options = valid_next_tokens(self.trie, prefix_list, discontinuous_token_id=self.discontinuous_token_id, nospace_mapping=self.nospace_mapping, json_parts=self.json_parts)
                if prefix_list[-1] not in [self.json_parts['start'], self.json_parts['next']]:
                    options.append(self.json_parts['end'])
                    options_post_comma = valid_next_tokens(self.trie, prefix_list + [self.json_parts['comma']], discontinuous_token_id=self.discontinuous_token_id, nospace_mapping=self.nospace_mapping, json_parts=self.json_parts)
                    if options_post_comma:
                        options.append(self.json_parts['comma'])
            # if self.discontinuous_token_id and not (prefix_list and prefix_list[-1] == self.discontinuous_token_id):
            #     options.append(self.discontinuous_token_id)
            # if self.allow_empty or prefix_list:
            #     options.extend(self.eos_token_id)
            if not options:
                logging.debug(f'No options for next word! This can happen with beam search, not sure how to fix. {prefix}')
            options = torch.tensor(options, dtype=torch.int, device=input_ids.device)
            mask = torch.isin(torch.arange(scores[i].numel(), device=input_ids.device), options)
            scores[i][~mask] = float("-inf")
        return scores


def create_trie(token_ids: list[int]) -> dict[int, dict]:
    """
    >>> trie = create_trie([1, 4, 3, 1, 4, 6]); print(trie[1][4].keys()) # walk down the path 1->4 and print the next options
    dict_keys([3, 6])
    >>> create_trie([1, 4, 3, 1, 4, 6])
    {1: {4: {3: {1: {4: {6: {}}}}, 6: {}}}, 4: {3: {1: {4: {6: {}}}}, 6: {}}, 3: {1: {4: {6: {}}}}, 6: {}}
    >>> create_trie('the quick brown fox jumped over the lazy dog'.split())
    {'the': {'quick': {'brown': {'fox': {'jumped': {'over': {'the': {'lazy': {'dog': {}}}}}}}}, 'lazy': {'dog': {}}}, 'quick': {'brown': {'fox': {'jumped': {'over': {'the': {'lazy': {'dog': {}}}}}}}}, 'brown': {'fox': {'jumped': {'over': {'the': {'lazy': {'dog': {}}}}}}}, 'fox': {'jumped': {'over': {'the': {'lazy': {'dog': {}}}}}}, 'jumped': {'over': {'the': {'lazy': {'dog': {}}}}}, 'over': {'the': {'lazy': {'dog': {}}}}, 'lazy': {'dog': {}}, 'dog': {}}
    """
    trie = {}
    for suffix in [token_ids[i:] for i in range(len(token_ids))]:
        node = trie
        for token in suffix:
            if token not in node:
                node[token] = {}
            node = node[token]
    return trie


def valid_next_tokens(trie: dict[int, dict], prefix: list[int], discontinuous_token_id: int = None, nospace_mapping: dict = None, json_parts: dict = None) -> list[int]:
    """
    >>> trie = create_trie([1, 4, 3, 1, 4, 6]); print(valid_next_tokens(trie, [1, 4]))
    [3, 6]
    >>> trie = create_trie([1, 4, 3, 1, 4, 6]); set(valid_next_tokens(trie, [1, 99], discontinuous_token_id=99)) == {3, 1, 4, 6}
    True
    >>> trie = create_trie([1, 4, 3, 1, 4, 6]); set(valid_next_tokens(trie, [1, 4, 99], discontinuous_token_id=99)) == {1, 4, 6}
    True
    >>> trie = create_trie([1, 4, 3, 1, 4, 6]); set(valid_next_tokens(trie, [99], discontinuous_token_id=99)) == {4, 3, 1, 6}
    True
    >>> trie = create_trie([1, 4, 3, 1, 4, 6]); set(valid_next_tokens(trie, [1, 99, 6], discontinuous_token_id=99)) == set()
    True
    >>> trie = create_trie([1, 4, 3, 1, 4, 6, 7, 4]); set(valid_next_tokens(trie, [1, 99, 6, 99], discontinuous_token_id=99)) == {4}
    True
    >>> trie = create_trie('the quick brown fox jumped over the lazy dog'.split())
    >>> set(valid_next_tokens(trie, 'the //'.split(), discontinuous_token_id='//')) == {'lazy', 'fox', 'brown', 'jumped', 'the', 'over', 'dog'}
    True
    >>> set(valid_next_tokens(trie, 'the // dog'.split(), discontinuous_token_id='//')) == set()
    True
    >>> trie = create_trie([9, 8, 5, 6, 8, 7, 5, 6, 9]); set(valid_next_tokens(trie, [1, 5, 6], json_parts={'start': 1, 'end': 2, 'comma': 3, 'next': 4})) == {8, 9}
    True
    """
    remaining_trie = trie
    for i in prefix:  # TODO: refactor
        if discontinuous_token_id is not None and i == discontinuous_token_id:
            remaining_trie = merge_tries(*iter_subtries_recursively(remaining_trie))
        elif json_parts and i == json_parts['comma']:
            continue
        elif json_parts and nospace_mapping and i == json_parts['start']:
            remaining_trie = {key: val
                              for j, subtrie in remaining_trie.items()
                              for key, val in prepend_path_to_trie(nospace_mapping[j], subtrie).items()}
        elif json_parts and i == json_parts['next']:
            remaining_trie = merge_tries(*iter_subtries_recursively(remaining_trie))
            if nospace_mapping:
                remaining_trie = {key: val
                                  for j, subtrie in remaining_trie.items()
                                  for key, val in prepend_path_to_trie(nospace_mapping[j], subtrie).items()}
        else:
            remaining_trie = remaining_trie.get(i, {})

    result = list(remaining_trie.keys())
    return result


def prepend_path_to_trie(prefix: list[int], trie: dict[int, dict]) -> dict[int, dict]:
    """
    >>> prepend_path_to_trie([1, 2, 3], {4: 5})
    {1: {2: {3: {4: 5}}}}
    """
    if not prefix:
        return trie
    return {prefix[0]: prepend_path_to_trie(prefix[1:], trie)}



def iter_subtries_recursively(d: dict[int, dict]) -> typing.Generator[dict[int, dict], None, None]:
    """
    >>> list(iter_subtries_recursively({1: {2: {3: {}}}}))
    [{2: {3: {}}}, {3: {}}, {}]
    >>> list(iter_subtries_recursively({1: {2: {1: {}}}, 4: {2: {1: {}}, 7: {}}, 2: {}, 9: {4: {}, 12: {13: {}}}}))
    [{2: {1: {}}}, {1: {}}, {}, {2: {1: {}}, 7: {}}, {1: {}}, {}, {}, {}, {4: {}, 12: {13: {}}}, {}, {13: {}}, {}]
    >>> list(iter_subtries_recursively({4: {3: {1: {4: {6: {}}}}, 6: {}}}))
    [{3: {1: {4: {6: {}}}}, 6: {}}, {1: {4: {6: {}}}}, {4: {6: {}}}, {6: {}}, {}, {}]
    """
    for k, v in d.items():
        yield v
        yield from iter_subtries_recursively(v)


def merge_tries(*tries) -> dict:
    """
    >>> merge_tries({1: {}}, {2: {}})
    {1: {}, 2: {}}
    >>> merge_tries({1: {}, 2: {3: {}}}, {2: {4: {}}})
    {1: {}, 2: {3: {}, 4: {}}}
    >>> merge_tries({1: {2: {3: {}}, 4: {5: {}}}}, {2: {2: {4: {}, 6: {7: {}}}}})
    {1: {2: {3: {}}, 4: {5: {}}}, 2: {2: {4: {}, 6: {7: {}}}}}
    >>> merge_tries({1: {2: {}}}, {1: {3: {}}}, {1: {2: {5: {}}}})
    {1: {2: {5: {}}, 3: {}}}
    """
    new_trie = {}
    for key in set(itertools.chain(*(trie.keys() for trie in tries))):
        subtries = (trie.get(key, {}) for trie in tries)
        new_trie[key] = merge_tries(*subtries)
    return new_trie





######### Below is old stuff no longer used, but perhaps useful as fallback option? ########

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


