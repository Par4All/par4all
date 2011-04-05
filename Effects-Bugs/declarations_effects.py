import pyps

w = pyps.workspace("declarations_effects.c");

w.props.memory_effects_only = False;
w.fun.main.print_code_proper_effects();

w.fun.main.display();
