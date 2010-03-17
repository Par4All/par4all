      program forall
      integer i,j
      real A(10), B(10,10)
chpf$ independent
      forall(i=1:10) A(i)=1.0
chpf$ independent(i,j)
      forall(i=1:10,j=1:10) B(j,i)=i+j
      end
