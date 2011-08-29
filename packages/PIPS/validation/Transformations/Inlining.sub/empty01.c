
// Try to inline an empty function
void empty() {
}

int caller() {
  int i;
  for(i=0; i<10;i++)
    empty();
}
