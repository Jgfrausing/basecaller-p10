import parasail

def align(ref, seq):
    """
    Get (and print) the alignment between `ref` and `seq`
    """
    alignment = parasail.sw_trace_striped_32(ref, seq, 8, 4, parasail.dnafull)
    align_tuple = (alignment.traceback.query, alignment.traceback.comp, alignment.traceback.ref)

    return alignment, align_tuple
