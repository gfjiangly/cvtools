import itertools
# ABCs from collections will be deprecated in python 3.8+,
# while collections.abc is not available in python 2.7
try:
    import collections.abc as collections_abc
except ImportError:
    import collections as collections_abc

import six


def is_str(x):
    """Whether the input is an string instance."""
    return isinstance(x, six.string_types)


def iter_cast(inputs, dst_type, return_type=None):
    """Cast elements of an iterable object into some type.

    Args:
        inputs (Iterable): The input object.
        dst_type (type): Destination type.
        return_type (type, optional): If specified, the output object will be
            converted to this type, otherwise an iterator.

    Returns:
        iterator or specified type: The converted object.
    """
    if not isinstance(inputs, collections_abc.Iterable):
        raise TypeError('inputs must be an iterable object')
    if not isinstance(dst_type, type):
        raise TypeError('"dst_type" must be a valid type')

    out_iterable = six.moves.map(dst_type, inputs)

    if return_type is None:
        return out_iterable
    else:
        return return_type(out_iterable)


def list_cast(inputs, dst_type):
    """Cast elements of an iterable object into a list of some type.

    A partial method of :func:`iter_cast`.
    """
    return iter_cast(inputs, dst_type, return_type=list)


def tuple_cast(inputs, dst_type):
    """Cast elements of an iterable object into a tuple of some type.

    A partial method of :func:`iter_cast`.
    """
    return iter_cast(inputs, dst_type, return_type=tuple)


def is_seq_of(seq, expected_type, seq_type=None):
    """Check whether it is a sequence of some type.

    Args:
        seq (Sequence): The sequence to be checked.
        expected_type (type): Expected type of sequence items.
        seq_type (type, optional): Expected sequence type.

    Returns:
        bool: Whether the sequence is valid.
    """
    if seq_type is None:
        exp_seq_type = collections_abc.Sequence
    else:
        assert isinstance(seq_type, type)
        exp_seq_type = seq_type
    if not isinstance(seq, exp_seq_type):
        return False
    for item in seq:
        if not isinstance(item, expected_type):
            return False
    return True


def is_list_of(seq, expected_type):
    """Check whether it is a list of some type.

    A partial method of :func:`is_seq_of`.
    """
    return is_seq_of(seq, expected_type, seq_type=list)


def is_tuple_of(seq, expected_type):
    """Check whether it is a tuple of some type.

    A partial method of :func:`is_seq_of`.
    """
    return is_seq_of(seq, expected_type, seq_type=tuple)


def slice_list(in_list, lens):
    """Slice a list into several sub lists by a list of given length.

    Args:
        in_list (list): The list to be sliced.
        lens(int or list): The expected length of each out list.

    Returns:
        list: A list of sliced list.
    """
    if not isinstance(lens, list):
        raise TypeError('"indices" must be a list of integers')
    elif sum(lens) != len(in_list):
        raise ValueError(
            'sum of lens and list length does not match: {} != {}'.format(
                sum(lens), len(in_list)))
    out_list = []
    idx = 0
    for i in range(len(lens)):
        out_list.append(in_list[idx:idx + lens[i]])
        idx += lens[i]
    return out_list


def concat_list(in_list):
    """Concatenate a list of list into a single list.

    Args:
        in_list (list): The list of list to be merged.

    Returns:
        list: The concatenated flat list.
    """
    return list(itertools.chain(*in_list))


