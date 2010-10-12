//     Check nest parallelization: the two loops are parallel, but they
//    must also be interchanged

// Make sure that the contiguous loop direction is chosen when an
// offest is used

void nest03()
{
  float a[10][20];
  int i, j;

  for(i = 0; i<9; i++)
    for(j = 0; j<10; j++)
      a[j][i+1] = 0.;

}

