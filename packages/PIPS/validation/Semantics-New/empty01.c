// Make sure that preconditions do not core dump when an init
// returning function does not return anything... but the returned
// value is not used.

int empty01() {
}

int caller() {
  int i;
  for(i=0; i<10;i++)
    empty01();
}
