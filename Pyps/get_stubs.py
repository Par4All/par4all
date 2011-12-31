from validation import vworkspace

with vworkspace() as w:
  print w.fun.a_stub.stub_p
  w.all_functions.flag_as_stub()
  print w.fun.a_stub.stub_p

