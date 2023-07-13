from llmtune.engine.inference.modules import QuantLinear

def make_quant(
    module, names, bits, groupsize=-1, name='', is_cuda=True
):
    if isinstance(module, QuantLinear):
        return
    for attr in dir(module):
        tmp = getattr(module, attr)
        name1 = name + '.' + attr if name != '' else attr
        if name1 in names:
            setattr(
                module, attr, QuantLinear(
                    bits=bits, 
                    groupsize=groupsize, 
                    in_features=tmp.in_features, 
                    out_features=tmp.out_features, 
                    bias=tmp.bias, 
                    is_cuda=is_cuda,
                )
            )
    for name1, child in module.named_children():
        make_quant(
            child, 
            names, 
            bits=bits,
            name=name + '.' + name1 if name != '' else name1, 
            groupsize=groupsize,
            is_cuda=is_cuda
        )
