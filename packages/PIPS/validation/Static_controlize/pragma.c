#include <stdio.h>

// This test used to make static_controlize segfault

int main() {
  int a[10],j;
kernel: for(int j=0; j<1;j++) {
    #pragma toto
    printf("");
  }
}
