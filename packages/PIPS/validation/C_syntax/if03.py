from validation import vworkspace

with vworkspace() as w:

  for f in w.all_functions:
    print '''#'''
    print '''# Parsed Code for ''' + f.name
    print '''#'''
    f.display(rc="PARSED_PRINTED_FILE")
  w.all_functions.validate_phases("print_code")


