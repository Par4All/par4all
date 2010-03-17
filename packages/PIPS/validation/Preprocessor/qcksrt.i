#define M 7
#define NSTACK 50
#define FM 7875
#define FA 211
#define FC 1663

void qcksrt(n,arr)
int n;
float arr[];
{
	int l=1,jstack=0,j,ir,iq,i;
	int istack[NSTACK+1];
	long int fx=0L;
	float a;
	void nrerror();

	ir=n;
	for (;;) {
		if (ir-l < M) {
			for (j=l+1;j<=ir;j++) {
				a=arr[j];
				for (i=j-1;arr[i]>a && i>0;i--) arr[i+1]=arr[i];
				arr[i+1]=a;
			}
			if (jstack == 0) return;
			ir=istack[jstack--];
			l=istack[jstack--];
		} else {
			i=l;
			j=ir;
			fx=(fx*FA+FC) % FM;
			iq=l+((ir-l+1)*fx)/FM;
			a=arr[iq];
			arr[iq]=arr[l];
			for (;;) {
				while (j > 0 && a < arr[j]) j--;
				if (j <= i) {
					arr[i]=a;
					break;
				}
				arr[i++]=arr[j];
				while (a > arr[i] && i <= n) i++;
				if (j <= i) {
					arr[(i=j)]=a;
					break;
				}
				arr[j--]=arr[i];
			}
			if (ir-i >= i-l) {
				istack[++jstack]=i+1;
				istack[++jstack]=ir;
				ir=i-1;
			} else {
				istack[++jstack]=l;
				istack[++jstack]=i-1;
				l=i+1;
			}
			if (jstack > NSTACK) nrerror("NSTACK too small in QCKSRT");
		}
	}
}

#undef M
#undef NSTACK
#undef FM
#undef FA
#undef FC
