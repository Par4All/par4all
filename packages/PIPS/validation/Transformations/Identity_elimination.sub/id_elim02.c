
int id_elim02() {
  int a_1, a_2;
  int i = 0;
  
  a_1 = i;
  //useless
  a_1 = a_1;
  //not useless
  a_2 = a_1;
  
  return a_2;
}

int main() {
  return id_elim02();
}
