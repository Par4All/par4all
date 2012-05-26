int struct02()
{
  struct two {
    int first;
    int second;
  } x[10];
  int i = 0;

  for(i=0;i<10;i++) {
    if(1) {
      x[i].first = 1;
      x[i].second = 0;
    }
    else {
      x[i].first = 0;
      x[i].second = 1;
    }
  }
  return 0;
}
