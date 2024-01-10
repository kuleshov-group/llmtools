from . import latticed4, latticee8_padded12, half_integer_4bit_1col, latticee8_padded12_rvq4bit
#latticee8, \
#half_integer_2bit, \
#latticee8_shifted, \
#half_integer_2bit_8col, \
#kmeans_8col, \
#kmedoid_8col, \
#latticee8_1bit, \
#latticed4_4bit, \
#latticee8_237bit, \
#latticed4_221bit, \
#latticed4_234bit, \
#latticed4_248bit, \
#latticed4_274bit, \
#latticed4_299bit

# name: (id, codebook class)
codebook_id = {
    'D4': (0, latticed4.D4_codebook),
    #'half_integer_2bit': (1, half_integer_2bit.half_integer_2bit),
    #'E8': (2, latticee8.E8_codebook),
    #'E8S': (3, latticee8_shifted.E8S_codebook),
    #'half_integer_2bit_8col': (4, half_integer_2bit_8col.half_integer_2bit_8col),
    #'kmeans_8col': (5, kmeans_8col.kmeans_8col),
    #'kmedoid_8col': (6, kmedoid_8col.kmedoid_8col),
    'E8P12': (7, latticee8_padded12.E8P12_codebook),
    #'E81B': (8, latticee8_1bit.E81B_codebook),
    #'D44B': (9, latticed4_4bit.D44B_codebook),
    'HI4B1C': (10, half_integer_4bit_1col.HI4B1C_codebook),
    #'E8237B': (11, latticee8_237bit.E8237B_codebook),
    #'D4221B': (12, latticed4_221bit.D4221B_codebook),
    #'D4234B': (13, latticed4_234bit.D4234B_codebook),
    #'D4248B': (14, latticed4_248bit.D4248B_codebook),
    #'D4274B': (15, latticed4_274bit.D4274B_codebook),
    #'D4299B': (16, latticed4_299bit.D4299B_codebook),
    'E8P12RVQ4B': (17, latticee8_padded12_rvq4bit.E8P12RVQ4B_codebook),
}

# id from above:6quantized linear implementation
quantized_class = {
    0: latticed4.QuantizedD4Linear,
    #1: half_integer_2bit.HalfInteger2BitLinear,
    #2: latticee8.QuantizedE8Linear,
    #3: latticee8_shifted.QuantizedE8SLinear,
    #4: half_integer_2bit_8col.HalfInteger2Bit8ColLinear,
    #5: kmeans_8col.KMeans8ColLinear,
    #6: kmedoid_8col.KMedoid8ColLinear,
    7: latticee8_padded12.QuantizedE8P12Linear,
    #8: latticee8_1bit.QuantizedE81BLinear,
    #9: latticed4_4bit.QuantizedD44BLinear,
    10: half_integer_4bit_1col.QuantizedHI4B1CLinear,
    #11: latticee8_237bit.QuantizedE8237BLinear,
    #12: latticed4_221bit.QuantizedD4221BLinear,
    #13: latticed4_234bit.QuantizedD4234BLinear,
    #14: latticed4_248bit.QuantizedD4248BLinear,
    #15: latticed4_274bit.QuantizedD4274BLinear,
    #16: latticed4_299bit.QuantizedD4299BLinear,
    17: latticee8_padded12_rvq4bit.QuantizedE8P12RVQ4BLinear,
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
