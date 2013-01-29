/*
 Use to lead to a segfault
*/

void userfunction(int a, int _c,int d) {
}

int main(int argc, char* argv[]) {

	userfunction(1, 2, 1);

	userfunction(2, 3, 0);

}

