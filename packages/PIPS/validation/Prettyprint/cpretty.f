      program main
      integer i, fun1
      external fun1
      i = 1
      if (i.eq.1) then
         i = 2 
      else
         i = 3
      endif
      call sub(i, 10)
      call sub2(10,20)
      call sub3
      i = fun1(2)
      end
      subroutine sub(j, k)
      integer i
      parameter (n=10)
      real*8 a(0:n)
      do i=0, k
         a(i) = 0
      enddo
      call sub4(a)
      end
      subroutine sub2(n, m)
      real*4 b(0:10,0:20)
      integer i,j
      do i=0, m
         do j=0, n
            b(j,i) = i + j
         enddo
      enddo
      end
      subroutine sub3
      parameter (n=2, m=3)
      integer i, fun1
      external fun1
      i = n+m
      do while (i.gt.0) 
         i = i - 1
         i = i - fun1(0)
      enddo
      end
      subroutine sub4(a)
      parameter (n=10)
      integer i
      real*8 a(0:n)
      real*8 b(0:n)
      do i=0, n
         b(i) = a(i)
      enddo
      do i=1, n-1
         a(i) = 0.25*(b(i-1)+2*b(i)+b(i+1))
      enddo
      end
      integer function fun1(i)
      integer i
      fun1 = 3
      end
