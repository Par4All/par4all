c     Dr. Wolfe's wonderful Choleksy Decomposition program
      subroutine chol(a,n)
      real a(n,n)
      do 10 k = 1,n
         a(k,k) = sqrt(a(k,k))
         do 20 i = k+1,n
            a(i,k) = a(i,k) / a(k,k)
            do 30 j = k+1,i
               a(i,j) = a(i,j) - a(i,k)*a(j,k)
 30         continue
 20      continue
 10   continue
      end
