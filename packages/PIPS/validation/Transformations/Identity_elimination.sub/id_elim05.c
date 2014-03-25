
int id_elim05() {
  int b[5];
  int i = 0;
  
  //not useless because side effect;
  b[i++]=b[i++];
  b[i--]=b[i--];
  //useless
  b[i]=b[i];
  
  return 0;
}

int main() {
  return id_elim05();
}
