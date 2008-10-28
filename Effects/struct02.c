int struct02()
{
  struct two {
    int first;
    int second;
  } x[10];
  int i = 0;

  for(i=0;i<10;i++) {
    x[i].first = 1;
    x[i].second = 0;
  }
}
