

import logging
import functools
import re
import typing

from transformers import LogitsProcessor
import torch
import math
import regex
import itertools
import copy

import string


class LogitsProcessorForQuotes(LogitsProcessor):

    def __init__(self, original: str, tokenizer, prompt_length: int, allow_empty: bool = False, force_json_response: bool = False, sep: str = '...') -> None:
        """
        An instance of this class can be passed into the transformers' model.generate function, to filter the logits based on permitted tokens.
        :param original: Original text to give quotes from
        :param tokenizer: transformers tokenizer to use
        :param prompt_length: this processor will get all inputs + generated tokens; it does constrained generation only from the prompt length onwards.
        :param allow_empty: whether to allow an empty quote ([] if json response, or simply empty string)
        :param force_json_response: whether to let the model answer a json list (["the quick", "brown fox"]), or a sep-separate string (the quick... brown fox)
        :param sep: which sep to use (default ...) to separate quotes in non-json responses
        """

        self.tokenizer = tokenizer
        self.start_constrained_generation_from = prompt_length

        json_parts = get_json_part_ids(self.tokenizer)

        token_ids_by_word = [[]]
        for tok_id in tokenizer.encode(original, add_special_tokens=False):
            tok_str = tokenizer.decode(tok_id, skip_special_tokens=True)
            if tok_str.startswith(' ') or all(c in string.punctuation for c in tok_str):
                token_ids_by_word.append([])
            token_ids_by_word[-1].append(tok_id)
        token_ids_by_word = [tuple(t) for t in token_ids_by_word]

        map_spaced_to_unspaced = {}
        for tok_ids in token_ids_by_word:
            decoded = tokenizer.decode(tok_ids, skip_special_tokens=True)
            stripped = decoded.lstrip()
            tok_ids_nospace = tuple(tokenizer.encode(stripped, add_special_tokens=False))
            map_spaced_to_unspaced[tok_ids] = tok_ids_nospace

        self.kwargs_for_optionsgenerator = {    # will be used to set up a parser to get options for next token
            'original_ids': token_ids_by_word,
            'map_to_unspaced': map_spaced_to_unspaced,
            'start_ids': [*json_parts['start']] if force_json_response else [],
            'sep_ids': [*json_parts['comma'], *json_parts['next']] if force_json_response else self.tokenizer.encode(sep, add_special_tokens=False),
            'empty_ids': ([*json_parts['start_empty'], *json_parts['end_empty']] if force_json_response else []) if allow_empty else None,
            'end_ids': [*json_parts['end'], tokenizer.eos_token_id] if force_json_response else [tokenizer.eos_token_id],
            'space_before_continued_quote': force_json_response,
        }

    def __call__(self, input_ids: torch.LongTensor, logits: torch.FloatTensor) -> torch.FloatTensor:
        """
        Called by the transformers' generate function to adapt the output logits based on the input_ids
        (prompt + generated thus far), prior to choosing the next token.
        """

        prefix_per_beam = input_ids[:, self.start_constrained_generation_from:]
        for beam_n, prefix in enumerate(prefix_per_beam):

            # TODO This creates a new parser for each token. Try caching instead? Kinda hard with multiple beams
            #  (and the beams themselves change), and probably not a huge time saver...
            options = QuoteParser(**self.kwargs_for_optionsgenerator).next_iter(prefix)

            if not options:
                logging.warning(f'No options for next word! This shouldn\'t really happen... {self.tokenizer.decode(prefix)}')

            options_tensor = torch.tensor(options, dtype=torch.int, device=input_ids.device)
            mask = torch.isin(torch.arange(logits[beam_n].numel(), device=input_ids.device), options_tensor)
            logits[beam_n][~mask] = float("-inf")
        return logits


class QuoteParser:

    def __init__(self, original_ids: list[int],
                 map_to_unspaced: dict[tuple[int], tuple[int]],
                 start_ids: list[int],
                 sep_ids: list[int],
                 end_ids: list[int],
                 empty_ids: list[int],
                 space_before_continued_quote: bool):
        """
        An instance of this class will parse the generated quotes thus far, and give permitted options for the next token.
        :param original_ids: generated token ids (ints) thus far.
        :param map_to_unspaced: a map from tokenized words with a space prefix (' fox'), to tokenized words with no space prefix ('fox').
        :param start_ids: token ids of the tokens that should prefix the model's response (the ids of `["`, for json output)
        :param sep_ids: token ids of the tokens to use for discontinuous quotes (the ids of `", "`, for json output; or `...` for non-json)
        :param end_ids: token ids of the tokens that should end hte model's response (the ids of `]"`, for json output)
        :param empty_ids: token ids of the tokens corresponding to an 'empty' quote response (`[]` for json output)
        :param space_before_continued_quote: whether the next part of a discontinuous quote (after sep) should start with a space or not.
        """

        self.original = original_ids
        self.sep_ids = sep_ids[::-1]        # store as stacks, to pop from the end
        self.start_ids = start_ids[::-1]
        self.end_ids = end_ids[::-1]
        self.space_before_continued_quote = space_before_continued_quote
        self.empty_ids = empty_ids[::-1] if empty_ids is not None else None
        self.map_to_unspaced = map_to_unspaced

        # State of this instance; these will be modified as `next` is called:
        self.current_positions = None
        self.stack = None

    def next_iter(self, prefix):
        options = self.next(None)
        for i in prefix:
            options = self.next(i)
        return options

    def next(self, i: int | None = None):
        """
        Process a token id (int), based on self.stack and self.current_positions.
        Initializing the stack and current_positions is done by doing next(None).

        The stack is used for special symbols and modified tokens (mainly: removing token-initial spaces). These are
        pushed onto the stack, and then consumed until the stack is empty, at which point we can start consuming from
        the resultant position in the original string.

        Because of choices, the self.stack is actually a list of multiple stacks, consumed in parallel.
        """

        # Initialize the stack and current_positions:
        if i is None:   # first function call, start of list
            # Only start from whole words
            self.current_positions = []
            # ensure spaceless start, for both json and non-json mode:
            self.stack = [
                (
                    [*self.map_to_unspaced.get(self.original[p], self.original[p])[::-1], *self.start_ids],
                    (p + 1 if p + 1 < len(self.original) else None)
                ) for p in range(len(self.original))
            ]
            if self.empty_ids is not None:
                self.stack.append(([*self.empty_ids], 0))

        # After interrupting a quote, continue from whole words, starting two tokens to the right:
        elif i == self.sep_ids[-1]:    # end of quote
            min_pos = min(self.current_positions) + 2
            self.current_positions = []
            if not self.space_before_continued_quote:
                self.stack = [
                    (
                        [*self.map_to_unspaced.get(self.original[p], self.original[p])[::-1], *self.sep_ids[:-1]],
                        (p + 1 if p + 1 < len(self.original) else None)
                    ) for p in range(min_pos, len(self.original))
                ]
            else:
                self.stack = [(self.sep_ids[:-1], p) for p in range(min_pos, len(self.original))]

        # At the end of the quote list:
        elif i == self.end_ids[-1]:
            self.current_positions = []
            self.stack = [(self.end_ids[:-1], None)]

        # Consume other types of token (stacked or from the current position):
        else:
            new_pos = []
            for p in self.current_positions:
                if self.original[p][0] == i:
                    self.stack.append(([*self.original[p][::-1]], (p + 1 if p + 1 < len(self.original) else None)))
            new_stack = []
            for s, p in self.stack:
                if s.pop() == i:
                    new_stack.append((s, p))
            self.current_positions = new_pos
            self.stack = new_stack

        # logging.debug(f'Parsed: {i}\n  stack: {self.stack}\n  pos: {self.current_pos}')

        # Having consumed a token, now clean up empty stacks, adding their resultant positions to current_pos:
        if self.stack:
            self.current_positions.extend([p for s, p in self.stack if s == [] and p is not None])
            self.stack = [(s, p) for s, p in self.stack if s]

        # Finally we can compute the list of options based on the stack and current positions:
        options_from_stack = [s[-1] for s, p in self.stack if s]
        options_from_current_pos = [self.original[p][0] for p in self.current_positions if p is not None and p < len(self.original)]

        options = options_from_stack + options_from_current_pos
        if self.current_positions or not options and not i == self.sep_ids[0]:
            options.append(self.end_ids[-1])
        if self.current_positions and any(p + 2 < len(self.original) for p in self.current_positions) and i != self.sep_ids[0]:
            options.append(self.sep_ids[-1])

        return list(set(options))


def get_json_part_ids(tokenizer):
    """
    Finds the token ids for a bunch of json delimiters, required for json output.

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


def find_spans_for_multiquote(original: str, quotes: list[str], must_exist=True, must_unique=False, case_sensitive=True) -> list[dict]:
    """
    Given a list of quotes (e.g., as obtained from the LLM), localize the corresponding spans in the original text.

    :param original: the original text, to which the given quotes are to be matched.
    :param quotes: list of quotes, to be matched to specific spans in the original text
    :param must_exist: if true, will raise ValueError if no match found.
    :param must_unique: if true, will raise ValueError if multiple matches found.
    :return: list of dictionaries with keys start, end and text.

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
    matches = (re.finditer(re.escape(quote), original, flags=re.IGNORECASE if not case_sensitive else 0) for quote in quotes)

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
            raise ValueError(f'No quote found: {quotes}.')
        else:
            logging.warning(f'No quote found, returning dict with start/end None: {quotes}')
            return [{'start': None, 'end': None, 'text': text} for text in quotes]

    elif len(candidate_multimatches) > 1:
        if must_unique:
            raise ValueError(f'No unique quote found: {quotes}')
        else:
            logging.warning(f'No unique quote found, going with the first, shortest distance one: {quotes}')

    closest_multimatch = min(candidate_multimatches, key=lambda multimatch: max(m.span()[1] for m in multimatch) - min(m.span()[0] for m in multimatch))
    return [{'start': m.span()[0], 'end': m.span()[1], 'text': m.group()} for m in closest_multimatch]

