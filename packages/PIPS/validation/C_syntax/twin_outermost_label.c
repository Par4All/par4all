void miam(void) {
 bara:
 gwin:
  goto bara;
}

void comment(void) {
  // A bara label
 bara:
  // A gwin label
 gwin:
  // This goto bara should not disappear:
  goto bara;
}
