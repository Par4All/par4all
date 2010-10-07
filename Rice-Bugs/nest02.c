//     Check nest parallelization: the two loops are parallel, but they
//    must also be interchanged

void nest02()
{
  float a[10][20];
  int i, j;

  for(i = 0; i<10; i++)
    for(j = 0; j<11; j++)
      a[j][i] = 0.;

}

