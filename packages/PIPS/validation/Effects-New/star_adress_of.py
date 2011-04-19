



import pyps
pyps.workspace.delete("star_adress_of")
w = pyps.workspace("star_adress_of.c",name="star_adress_of")

main = w.fun.main

main.display(activate="print_code_proper_effects")


