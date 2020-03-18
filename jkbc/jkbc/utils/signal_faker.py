# import jkbc.types as t
import typing as t

'''
Alphabet = "AB"
A = 2
B = 4
4mer_weights = 1, 1.5, 1.5, 1
'''


def generate_ref_and_signal(ref_length: int, alphabet: str, signal_dict: t.Dict[str, int]) -> t.Tuple[str, t.List[int]]:
    import random

    ref: str = "".join([random.choice(alphabet) for _ in range(ref_length)])

    # gram_size should equal the lengths of keys in the signal_dict
    gram_size: int = len(list(signal_dict.keys())[0])

    # Add padding to ref
    pad_size = gram_size - 1
    ref = 'x' * pad_size + ref + 'x' * pad_size

    return ref, gen_signal_from_ref(ref, signal_dict, gram_size)


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

    import itertools

    def mk_n_product(n): return ["".join(m) for m in itertools.product(
        alphabet_values.keys(), repeat=n)]

    def concat(ls): return list(itertools.chain.from_iterable(ls))

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
