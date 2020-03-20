import itertools
import random

import jkbc.types as t

'''
Alphabet = "AB"
A = 2
B = 4
4mer_weights = 1, 1.5, 1.5, 1
'''

T = t.TypeVar('T')
S = t.TypeVar('S')


def generate_signal__ref_reflengths(ref_length: int, alphabet: str, signal_dict: t.Dict[str, S], window_size: int = 300, min_max_bases_per_window: t.Tuple[int, int] = (45, 70), use_padding: bool = False, pad_val: t.Union[T, None] = None) -> t.Tuple[t.List[t.List[S]], t.List[int], t.List[int]]:
    '''
        Warning:
            If you get a KeyError then it is because the alphabet and signal_dict doesn't match.'''
    assert window_size <= ref_length, "window_size cannot be larger than ref_length"

    # Generate a reference string over the alphabet
    ref: T = "".join([random.choice(alphabet) for _ in range(ref_length)])

    # gram_size should equal the lengths of keys in the signal_dict
    gram_size: int = len(list(signal_dict.keys())[0])

    # Add padding to ref, i.e. adding blanks before and after reference (AGTC -> xxAGTCxx)
    if use_padding:
        pad_size = gram_size - 1
        padding = pad_val * pad_size
        ref = padding + ref + padding

    signals: t.List[T] = __gen_signal_from_ref(
        ref, signal_dict, gram_size)

    signals_in_windows, bases_per_window = __randomly_repeat_signals_into_windows(
        signals, min_max_bases_per_window, window_size)

    ref_corrected_for_blank_0 = [int(r) + 1 for r in ref]

    return signals_in_windows, ref_corrected_for_blank_0, bases_per_window


def make_signal_dict(gram_size: int = 4, alphabet_values: t.Dict[str, S] = {'A': 2, 'B': 4}, pos_multipliers: t.List[float] = [1, 1.5, 1.5, 1]) -> t.Dict[str, S]:
    assert gram_size == len(pos_multipliers)

    letter_combinations = __mk_n_product(alphabet_values.keys(), gram_size)
    letter_combinations.sort()

    comb_and_vals: t.Dict[T, S] = dict([("".join(mer), sum([int(alphabet_values[l] * pos_multipliers[i])
                                                            for l, i in zip(mer, range(gram_size))])) for mer in letter_combinations])
    return comb_and_vals


def __mk_n_product(ls, n):
    return ["".join(m) for m in itertools.product(
        ls, repeat=n)]


def __gen_signal_from_ref(ref: T, signal_dict: t.Dict[str, S], gram_size: int = 5) -> t.List[S]:
    assert len(list(signal_dict.keys(
    ))[0]) == gram_size, "gram_size has to match the length of the keys in signal_dict."
    n_grams: t.List[T] = __make_n_grams(ref, gram_size)
    return [signal_dict[gram] for gram in n_grams]


def __make_n_grams(ls, n: int = 5, join: bool = True) -> t.List:
    assert n <= len(ls), "n should be <= len(ls)"
    lists = [ls[i:] for i in range(n)]
    n_grams = zip(*lists)
    if join:
        n_grams = map("".join, n_grams)
    return list(n_grams)


def __concat(ls: t.Iterable[t.Iterable[T]]) -> t.Iterable[T]:
    return itertools.chain.from_iterable(ls)


def __randomly_repeat_signals_into_windows(ls: t.List[S], min_max_bases_per_window: t.Tuple[int, int], window_size: int) -> t.Tuple[t.List[t.List[S]], t.List[int]]:
    assert min_max_bases_per_window[0] <= min_max_bases_per_window[
        1], "Violation of n <= m in min_max_bases_per_window = (n,m)"

    max_repeats = window_size // min_max_bases_per_window[0]
    min_repeats = window_size // min_max_bases_per_window[1]

    windows = []
    window = []
    bases_per_window = []
    bases_in_current_window = 0
    for x in ls:
        repeats_for_x: int = random.randint(
            min_repeats, max_repeats)

        window += repeats_for_x * [x]
        bases_in_current_window += 1

        if len(window) >= window_size:
            # Cut off the remaining values up to window_size
            windows += [window[:window_size]]
            bases_per_window += [bases_in_current_window]
            window = []
            bases_in_current_window = 0

    assert len(windows) == len(bases_per_window), "Post-condition violated."
    return windows, bases_per_window


def test_end_to_end():
    ref_length: int = 10
    alphabet = "01"
    signal_dict = {'00': 1, '01': 2, '10': 3, '11': 4}

    x, y, y_lengths = generate_signal__ref_reflengths(
        ref_length, alphabet, signal_dict, window_size=10, min_max_bases_per_window=(2, 3), use_padding=False)
    return x, y, y_lengths
