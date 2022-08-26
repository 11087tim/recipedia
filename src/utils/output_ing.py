# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

replace_dict = {' .': '.',
                ' ,': ',',
                ' ;': ';',
                ' :': ':',
                '( ': '(',
                ' )': ')',
               " '": "'"}

def get_ingrs(ids, ingr_vocab_list):
    gen_ingrs = []
    for ingr_idx in ids:
        ingr_name = ingr_vocab_list[ingr_idx]
        if ingr_name == '<pad>':
            break
        gen_ingrs.append(ingr_name)
    return gen_ingrs


def prepare_output(gen_ingrs, ingr_vocab_list):

    if gen_ingrs is not None:
        gen_ingrs = get_ingrs(gen_ingrs, ingr_vocab_list)

    outs = {'ingrs': gen_ingrs}

    return outs
