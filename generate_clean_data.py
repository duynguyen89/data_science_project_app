import sys
import matplotlib.pyplot as plt
import builtins
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

plt.show = lambda *args, **kwargs: None
builtins.display = lambda *args, **kwargs: None
builtins.print = lambda *args, **kwargs: None

with open('extracted_code.py', encoding='utf-8') as f:
    code = f.read()

code = code.replace('import streamlit as st', '')
# Remove some prints
import re
code = re.sub(r'print\(.*\)', '', code)

import traceback
try:
    exec(code, globals())
except Exception as e:
    sys.stderr.write(f"Failed at some point: {e}\n")

if 'data' in globals():
    data.to_csv('house_samples.csv', index=False)
    sys.stderr.write(f"house_samples.csv written! Data shape: {data.shape}\n")

if 'cosine_sim_new' in globals():
    import pickle
    with open('nha_cosine_sim.pkl', 'wb') as f:
        pickle.dump(cosine_sim_new, f)
    sys.stderr.write("nha_cosine_sim.pkl written!\n")
