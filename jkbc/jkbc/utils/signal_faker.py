# import jkbc.types as t
import typing as t
import itertools

'''
Alphabet = "AB"
A = 2
B = 4
4mer_weights = 1, 1.5, 1.5, 1
'''


def generate_signal_and_ref(ref_length: int, alphabet: str, signal_dict: t.Dict[str, int], repeation_rate_range: t.Tuple[int, int] = (1, 1)) -> t.Tuple[t.List[int], str]:

    assert repeation_rate_range[0] <= repeation_rate_range[
        1], "Violation of n <= m in repeation_rate_range = (n,m)"
    import random

    ref: str = "".join([random.choice(alphabet) for _ in range(ref_length)])

    # gram_size should equal the lengths of keys in the signal_dict
    gram_size: int = len(list(signal_dict.keys())[0])

    # Add padding to ref
    pad_size = gram_size - 1
    ref = 'x' * pad_size + ref + 'x' * pad_size

    def add_random_repeats(ls: t.List[int]) -> t.List[int]:
        return list(concat([[x] * random.randint(*repeation_rate_range) for x in ls]))

    signal: t.List[int] = gen_signal_from_ref(ref, signal_dict, gram_size)
    signal_with_repeats: t.List[int] = add_random_repeats(signal)

    return signal_with_repeats, ref


def gen_signal_from_ref(ref: str, signal_dict: t.Dict[str, int], gram_size: int = 2) -> t.List[int]:
    assert len(list(signal_dict.keys(
    ))[0]) == gram_size, "gram_size has to match the length of the keys in signal_dict."
    n_grams: t.List[str] = make_n_grams(ref, gram_size)
    return [signal_dict[gram] for gram in n_grams]


def make_n_grams(ls, n: int = 2, join: bool = True) -> t.List:
    assert n <= len(ls), "n should be <= ls"
    lists = [ls[i:] for i in range(n)]
    n_grams = zip(*lists)
    if join:
        n_grams = map("".join, n_grams)
    return list(n_grams)


def make_signal_dict(gram_size: int = 4, pad_char: str = 'x', pad_val: int = 0, alphabet_values: t.Dict[str, int] = {'A': 2, 'B': 4}, pos_multipliers: t.List[float] = [1, 1.5, 1.5, 1]) -> t.Dict[str, int]:
    assert gram_size == len(pos_multipliers)

    def mk_n_product(n): return ["".join(m) for m in itertools.product(
        alphabet_values.keys(), repeat=n)]

    def make_padded(n, ls): return concat(
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


T = t.TypeVar('T')


def concat(ls: t.Iterable[t.Iterable[T]]) -> t.Iterable[T]:
    return list(itertools.chain.from_iterable(ls))
