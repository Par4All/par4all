
int main(int argc, char ** argv) 
{
  int tab[10];
  int *tab2[10];
  int tab3[10][10];
  int **tab4;
  int tab5[15];
  
  int i;

  foo(tab);
  printf("%s\n", "tab");
  myprint(tab);


  foo(tab2[4]);
  foo(tab3[5]);
  foo(tab4[6]);
  foo(&(tab[0]));
  foo(&tab5[3]);

  printf("%s:\n", "tab2[4]");
  myprint(tab2[4]);
  printf("%s\n", "tab3[5]");
  myprint(tab3[5]);
  printf("%s\n", "tab4[6]");
  myprint(tab4[6]);
  printf("%s\n", "tab");
  myprint(&(tab[0]));
  printf("%s\n", "tab5");
  myprint(&(tab5[3]));
  return 1;
}
