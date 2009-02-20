double* g_410068 = (double *)0x410068;

double* g_40E120 = (double *)0x410120;

double* g_412000 = (double *)0x412000;

double* g_412010 = (double *)0x412010;

double* g_407480 = (double *)0x407480;

 

 

double* g_471F18 = (double *)0x471F18;

double* g_471F28 = (double *)0x471F28;

double* g_471F58 = (double *)0x471F58;

 

int* g_408330 = (int *)0x408330;

int* g_4081E8 = (int *)0x4081E8;

int* g_40746C = (int *)0x40746C;

 

int* g_xxxx = (int *) 0x123456;

 

//extern void parameters(int);
void parameters(int A) {
};

//extern void endloop(int);
void endloop(int i) {
};


 

 

 

void sub_00402680()

{

  int var1;

  int var2;

  int var3;

  int var4;

 

  var3 = 0;

 

 

loc_402685:

 

  var4 = *g_408330;

  var4 *= 25;

  var4 += var3;

  g_407480[var4]=0.0;

 

  ++var3;

 

  if (var3 < 25) goto loc_402685;

 

 

  parameters(1);

 

 

 

loc_4026A6:

 

   var2 = 0;

 

   var1 = *g_xxxx;

 

   if (var1 <= 0) goto loc_4026E9;

 

 

 

loc_4016B2:

 

 
g_40E120[var2]=(((g_412010[var2]*(*g_471F58))+(g_412000[var2]*(*g_471F28)))*
g_410068[var2])+(*g_471F18);

 

   ++var2;

 

   if (var2 < var1) goto loc_4016B2;

 

 

 

 

loc_4026E9:

 

   endloop(1);

 

   if (*g_4081E8 < *g_40746C) goto loc_4026A6;

 

}
