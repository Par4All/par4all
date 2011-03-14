      program LZ1
      call sub(5.4,5,7)
      end
c
      subroutine sub(a,m,n)
      real a(m)
c      integer m,n
      k= 3 * n + 2
      do 10 i = 1, k
         t = t + 1.0
 10   continue
      return
      end
