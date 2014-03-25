
int id_elim03() {
  int a_1, a_2;
  int b_1, b_2;
  int i = 1;
  
  a_1 = i;
  //useless
  a_1 = a_1;
  //not useless
  a_2 = a_1;
  b_1 = a_1*10 + i;
  //useless
  b_1 = b_1;
  //not useless
  b_2 = b_1;
  a_2 = b_2 - i;
  //not useless
  a_1 = a_2;
  //useless
  a_2 = a_2;
  
  return 0;
}

int main() {
  return id_elim03();
}
