      subroutine static(a,b,n)
      real a(100,200), b(100,200)
      integer m, n, i, j, k
      do 1 i=3,n,2
          m = 2*n-1
          do 2 j=1,n-2*i
              if(4*i .gt. 2*j-3 .and. i-m .lt. n) then
                  do 3 k=i+1,n
                      a(n-j,i+j-m) = sqrt(b(2*i-1,3*j+1) + 1.)/2.
3                 continue
              end if
2         continue
1      continue
       end
