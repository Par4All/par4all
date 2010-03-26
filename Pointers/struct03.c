int struct03()
{
  struct three {
    int first;
    int second;
  };
  struct four {
    struct three un;
    struct three deux;
  } x[10];
  int i = 0;

  for(i=0;i<10;i++) {
    x[i].un.first = 1;
    x[i].deux.second = 0;
  }
}
