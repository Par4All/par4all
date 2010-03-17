/* struct01 + address expression in rhs */

int struct04()
{
  struct one {
    int first;
    int second;
  } x;
  int y[10];

  //  x.first = 1;
  //x.second = 2;

  //y[0] = 1;

  x.first = x.second + 1;
}
