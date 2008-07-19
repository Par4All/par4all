      PROGRAM test3
      REAL T(10,10)
      
      N=10
      do 10 I = 1,N
         T(I,I)=0
         do 10 K=1,I-1
            T(I,K)=1
            T(K,I)=-1
 10      continue
         end
