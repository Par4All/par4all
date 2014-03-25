
int id_elim01() {
  int a;
  int i = 0;
  
  a = i;
  //useless
  a = a;
  
  return a;
}

int main() {
  return id_elim01();
}
