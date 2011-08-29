
// Try to inline an empty function with an implicit return (declared as int)
int empty() {
}

int caller() {
  int i;
  for(i=0; i<10;i++)
    empty();
}
