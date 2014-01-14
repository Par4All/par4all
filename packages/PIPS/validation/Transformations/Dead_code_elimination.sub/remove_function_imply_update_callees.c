


void toto() {

}



int main() {
  // This function call can be removed, but we have to update callees for main
  // if we don't do it, the callgraph is broken and it might lead to abort later in PIPS
  toto();
}
