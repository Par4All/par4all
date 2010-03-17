void miam(void) {
 bara:
 gwin:
  goto gwin;
}

void comment(void) {
  // A bara label
 bara:
  // A gwin label
 gwin:
  // This goto gwin should not disappear:
  goto gwin;
}
