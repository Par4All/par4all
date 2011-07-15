from validation import vworkspace

with vworkspace() as w:
  cumu = w.fun.cumu
  print "\n****** Proper effects ******\n"
  cumu.display(activate="print_code_proper_effects")

  print "\n****** Cumulated effects ******\n"
  cumu.display(activate="print_code_cumulated_effects")

  print "\n\n****** With non memory effects ******\n\n"
  w.props.memory_effects_only = False
  # force recomputing
  cumu.proper_effects()

  print "\n****** Proper effects ******\n"
  cumu.display(activate="print_code_proper_effects")
  print "\n****** Cumulated effects ******"
  print "****** We expect them to be the same as before ******\n\n"
  cumu.display(activate="print_code_cumulated_effects")



