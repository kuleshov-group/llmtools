from . import \
    latticed4, \
    latticee8, \
    half_integer_2bit, \
    latticee8_shifted, \
    half_integer_2bit_8col, \
    kmeans_8col, \
    kmedoid_8col, \
    latticee8_padded12, \
    latticee8_1bit

# name: (id, codebook class)
codebook_id = {
    'D4': (0, latticed4.D4_codebook),
    'half_integer_2bit': (1, half_integer_2bit.half_integer_2bit),
    'E8': (2, latticee8.E8_codebook),
    'E8S': (3, latticee8_shifted.E8S_codebook),
    'half_integer_2bit_8col': (4, half_integer_2bit_8col.half_integer_2bit_8col),
    'kmeans_8col': (5, kmeans_8col.kmeans_8col),
    'kmedoid_8col': (6, kmedoid_8col.kmedoid_8col),
    'E8P12': (7, latticee8_padded12.E8P12_codebook),
    'E81B': (8, latticee8_1bit.E81B_codebook),
}

# id from above: quantized linear implementation
quantized_class = {
    0: latticed4.QuantizedD4Linear,
    1: half_integer_2bit.HalfInteger2BitLinear,
    2: latticee8.QuantizedE8Linear,
    3: latticee8_shifted.QuantizedE8SLinear,
    4: half_integer_2bit_8col.HalfInteger2Bit8ColLinear,
    5: kmeans_8col.KMeans8ColLinear,
    6: kmedoid_8col.KMedoid8ColLinear,
    7: latticee8_padded12.QuantizedE8P12Linear,
    8: latticee8_1bit.QuantizedE81BLinear,
}

cache_permute_set = {
    0,  # D4
}


def get_codebook(name):
    return codebook_id[name][1]()


def get_id(name):
    return codebook_id[name][0]


def get_quantized_class(id):
    return quantized_class[id]
