import math
import numpy as np
import tenseal as ts
import json
import warnings
warnings.filterwarnings('ignore')

with open('ModDict.json', 'r') as fcc_file:
    schemeDict = json.load(fcc_file)

def generate_ckks_key(sec_level, mul_depth, poly_modulus_degree):
    params = schemeDict.get(str(sec_level), {}).get(str(mul_depth), {}).get(str(poly_modulus_degree), None)
    
    if not params:
        raise ValueError("Invalid parameters. Check sec_level, mul_depth, and poly_modulus_degree.")

    context = ts.context(
        scheme=ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=params["n"],
        coeff_mod_bit_sizes=params["qi_sizes"]
    )
    context.global_scale = params["scale"]
    context.generate_galois_keys()

    context.poly_modulus_degree = params["n"]
    return context

def enc_vector(context, arr_x):
    return ts.ckks_vector(context, arr_x)

def dec_vector(context, ctxt_x):
    return ctxt_x.decrypt()

def seg_enc_vector(context, vector, vecl):
    block_enc_arr = []
    block_len = context.poly_modulus_degree // 2 
    block_arr_len = math.ceil(vecl / block_len)
    
    for i in range(block_arr_len):
        start_index = block_len * i
        end_index = min(block_len * (i + 1), vecl)
        vector_block = vector[start_index:end_index]

        if len(vector_block) < block_len:
            vector_block = np.pad(vector_block, (0, block_len - len(vector_block)), 'constant', constant_values=0)
        
        enc_vector_block = enc_vector(context, vector_block)
        block_enc_arr.append(enc_vector_block)
    return block_enc_arr

def seg_dec_vector(context, block_enc_arr):
    dec_result = []
    for block_enc in block_enc_arr:
        decrypted_block = dec_vector(context, block_enc)
        dec_result.extend(decrypted_block)
    return np.array(dec_result)
