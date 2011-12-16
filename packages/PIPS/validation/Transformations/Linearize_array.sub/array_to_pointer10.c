int get_mat_size(int argc, char *argv[])
{
   int result = 0;
   result = argv[1][0];
   return result;
}

int main(int argc, char *argv[])
{
   int size = get_mat_size(argc, argv);
   return size;
}

