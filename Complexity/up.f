      program up
      call sub(10)
      end
c
      subroutine sub(n)
      integer n,i
      do 10 i=1,n*(n+1),2
         t=t+1.
 10   continue
      return
      end
