int main() {
  int a[100];
  int i, j;

  // Test all the combinations of signs, strictness and hand-sideness

  // for(j = 10; j > 5; j--)
  for(j = 10; j > 5; j--)
    a[j] = 2;
  // for(j = 10; j > -20; j--)
  for(j = 10; j > -20; j--)
    a[j] = 2;
  // for(j = -10; j > -20; j--)
  for(j = -10; j > -20; j--)
    a[j] = 2;

  // for(j = 10; j < 50; j++)
  for(j = 10; j < 50; j++)
    a[j] = 2;
  // for(j = -50; j < 20; j++)
  for(j = -50; j < 20; j++)
    a[j] = 2;
  // for(j = -50; j < -20; j++)
  for(j = -50; j < -20; j++)
    a[j] = 2;

  // for(j = 10; j >= 5; j--)
  for(j = 10; j >= 5; j--)
    a[j] = 2;
  // for(j = 10; j >= -20; j--)
  for(j = 10; j >= -20; j--)
    a[j] = 2;
  // for(j = -10; j >= -20; j--)
  for(j = -10; j >= -20; j--)
    a[j] = 2;

  // for(j = 10; j <= 50; j++)
  for(j = 10; j <= 50; j++)
    a[j] = 2;
  // for(j = -50; j <= 20; j++)
  for(j = -50; j <= 20; j++)
    a[j] = 2;
  // for(j = -50; j <= -20; j++)
  for(j = -50; j <= -20; j++)
    a[j] = 2;


  // for(j = 10; 5 < j; j--)
  for(j = 10; 5 < j; j--)
    a[j] = 2;
  // for(j = 10; -20 < j; j--)
  for(j = 10; -20 < j; j--)
    a[j] = 2;
  // for(j = -10; -20 < j; j--)
  for(j = -10; -20 < j; j--)
    a[j] = 2;

  // for(j = 10; 50 > j; j++)
  for(j = 10; 50 > j; j++)
    a[j] = 2;
  // for(j = -50; 20 > j; j++)
  for(j = -50; 20 > j; j++)
    a[j] = 2;
  // for(j = -50; -20 > j; j++)
  for(j = -50; -20 > j; j++)
    a[j] = 2;

  // for(j = 10; 5 <= j; j--)
  for(j = 10; 5 <= j; j--)
    a[j] = 2;
  // for(j = 10; -20 <= j; j--)
  for(j = 10; -20 <= j; j--)
    a[j] = 2;
  // for(j = -10; -20 <= j; j--)
  for(j = -10; -20 <= j; j--)
    a[j] = 2;

  // for(j = 10; 50 >= j; j++)
  for(j = 10; 50 >= j; j++)
    a[j] = 2;
  // for(j = -50; 20 >= j; j++)
  for(j = -50; 20 >= j; j++)
    a[j] = 2;
  // for(j = -50; -20 >= j; j++)
  for(j = -50; -20 >= j; j++)
    a[j] = 2;

  return 0;
}
