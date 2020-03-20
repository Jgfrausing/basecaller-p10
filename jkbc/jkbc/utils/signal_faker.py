import itertools

import jkbc.types as t

'''
Alphabet = "AB"
A = 2
B = 4
4mer_weights = 1, 1.5, 1.5, 1
'''


def generate_signal_and_ref(ref_length: int, alphabet: str, signal_dict: t.Dict[str, int], window_size: int = 300, min_max_bases_per_window: t.Tuple[int, int] = (1, 1), pad_char: str = 'x') -> t.Tuple[t.List[t.List[int]], str]:
    '''
        Warning:
            If you get a KeyError then it is because the alphabet and signal_dict doesn't match.'''
    assert window_size <= ref_length, "window_size cannot be larger than ref_length"

    import random

    # Generate a reference string over the alphabet
    ref: str = "".join([random.choice(alphabet) for _ in range(ref_length)])

    # gram_size should equal the lengths of keys in the signal_dict
    gram_size: int = len(list(signal_dict.keys())[0])

    # Add padding to ref, i.e. adding blanks before and after reference (AGTC -> xxAGTCxx)
    pad_size = gram_size - 1
    padding = pad_char * pad_size
    ref = padding + ref + padding

    # [A, B, C, D] -> [0, 1, 2, 3]
    signals: t.List[int] = __gen_signal_from_ref(ref, signal_dict, gram_size)

    # [0, 1, 2, 3] -> [[0,0,0,1,1], [2,2,3,3,3]]
    signals_in_windows: t.List[t.List[int]] = __randomly_repeat_bases_into_windows(
        signals, min_max_bases_per_window, window_size)

    return signals_in_windows, ref


def make_signal_dict(gram_size: int = 4, pad_char: str = 'x', pad_val: int = 0, alphabet_values: t.Dict[str, int] = {'A': 2, 'B': 4}, pos_multipliers: t.List[float] = [1, 1.5, 1.5, 1]) -> t.Dict[str, int]:
    assert gram_size == len(pos_multipliers)

    def mk_n_product(n): return ["".join(m) for m in itertools.product(
        alphabet_values.keys(), repeat=n)]

    def make_padded(n, ls): return __concat(
        [[pad_char * n + ele, ele + pad_char * n] for ele in ls])

    # Create all combinations of the alphabet including the padded versions (fx xAAB)
    letter_combinations = []
    for n in range(1, gram_size + 1):
        pad_size = gram_size - n
        letter_combinations += make_padded(pad_size, mk_n_product(n))

    letter_combinations.sort()

    alphabet_values[pad_char] = pad_val

    comb_and_vals: t.Dict[str, int] = dict([("".join(mer), sum([int(alphabet_values[l] * pos_multipliers[i])
                                                                for l, i in zip(mer, range(gram_size))])) for mer in letter_combinations])
    return comb_and_vals


def __gen_signal_from_ref(ref: str, signal_dict: t.Dict[str, int], gram_size: int = 2) -> t.List[int]:
    assert len(list(signal_dict.keys(
    ))[0]) == gram_size, "gram_size has to match the length of the keys in signal_dict."
    n_grams: t.List[str] = __make_n_grams(ref, gram_size)
    return [signal_dict[gram] for gram in n_grams]


def __make_n_grams(ls, n: int = 2, join: bool = True) -> t.List:
    assert n <= len(ls), "n should be <= ls"
    lists = [ls[i:] for i in range(n)]
    n_grams = zip(*lists)
    if join:
        n_grams = map("".join, n_grams)
    return list(n_grams)


T = t.TypeVar('T')


def __concat(ls: t.Iterable[t.Iterable[T]]) -> t.Iterable[T]:
    return itertools.chain.from_iterable(ls)


def __randomly_repeat_bases_into_windows(ls: t.List[int], min_max_bases_per_window: t.Tuple[int, int], window_size: int) -> t.List[t.List[int]]:
    assert min_max_bases_per_window[0] <= min_max_bases_per_window[
        1], "Violation of n <= m in min_max_bases_per_window = (n,m)"
    import random
    windows = []

    max_repeats = window_size // min_max_bases_per_window[0]
    min_repeats = window_size // min_max_bases_per_window[1]

    window = []
    for x in ls:
        repeats_for_x: int = random.randint(
            min_repeats, max_repeats)
        window += repeats_for_x * [x]
        if len(window) >= window_size:
            # Cut off the remaining values up to window_size
            windows += [window[:window_size]]
            window = []

    return windows
