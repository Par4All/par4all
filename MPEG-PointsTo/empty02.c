// Make sure that preconditions do not core dump when an init
// returning function does not return anything... but the returned
// value is used. Undefined value for the C standard

int empty02() {
}

int caller() {
  int i, j;
  for(i=0; i<10;i++)
    j += empty02();
  return j;
}
