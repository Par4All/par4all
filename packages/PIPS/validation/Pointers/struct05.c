int struct05()
{
  struct two {
    int first;
    int second;
  } x[10];
  int i = 0;

  for(i=0;i<10;i++) {
    if(1) {
      x[i].first = x[i].second;
    }
    else {
      x[i].second = x[i].first;
    }
  }
  return 0;
}
