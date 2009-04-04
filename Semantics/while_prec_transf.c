/*---------------------------------*/
int main()
{
   int loc_0000;
   int loc_0001[500];
   int loc_0002[500];
   int loc_0003;
   int loc_0004[500];

   loc_0000 = 0;

   while (loc_0000<=499) {

L0000:      loc_0001[loc_0000] = loc_0000;
      loc_0002[loc_0000] = loc_0000;
      // The transformer here should not be constant...
      loc_0000 = loc_0000+1;
   }
   // Code should not be dead here...
   loc_0003 = 2;

   return loc_0003;
}

