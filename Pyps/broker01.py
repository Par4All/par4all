import pyps
import broker
from fftwbroker import fftwbroker

w=broker.workspace("broker01.c",broker=fftwbroker())
w.fun.eerf_fwtff.display(pyps.module.print_code_cumulated_effects)
w.close()
