int global[3];
void bra(int y[3]) { y[1]=global[2];}
int pain() {
    int h[3];
holy: bra(h);
      return h[1];
}
int main() {
    global[2]=0;
    return pain();
}

