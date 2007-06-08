       program ch
       real a(1:100,1:100)
       real b(1:100)
       integer n
       do 10 j = 1,n
cvd$    concur
        do 20 i = max(1+1,j),n
         do 30 k = 1,min(j-1,i-1)
          a(i,j) = a(i,j)-a(i,k)*a(j,k)
   30    continue
   20   continue
        a(j,j) = sqrt(a(j,j))
cvd$    concur
        do 40 i = j+1,n
         a(i,j) = a(i,j)/a(j,j)
   40   continue
   10  continue
       end
