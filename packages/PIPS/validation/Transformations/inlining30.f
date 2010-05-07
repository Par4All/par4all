      program test
      implicit none
      call scalar
      call array
      end

      subroutine scalar
      implicit none
      integer a,b
      call sa0(a)
      call sa1(a)
      call sb(a,b)
      print *,b
      end

      subroutine sa0(a)
      implicit none
      integer a
      a=0
      end

      subroutine sa1(a)
      implicit none
      integer a
      a=1
      end

      subroutine sb(a,b)
      implicit none
      integer a,b
      b=a
      end   

      subroutine array
      implicit none
      integer n
      parameter (n=10)
      integer a(n),b(n)
      call aa0(n,a)
      call aa1(n,a)
      call ab(n,a,b)
      print *,b
      end

      subroutine aa0(n,a)
      implicit none
      integer n, a(n),i
      
      do i=1,n
         a(i)=0
      enddo
      end   
      subroutine aa1(n,a)
      implicit none
      integer n, a(n),i
      
      do i=1,n
         a(i)=1
      enddo
      end 
      subroutine ab(n,a,b)
      implicit none
      integer n, a(n),b(n),i
      
      do i=1,n
         b(i)=a(i)
      enddo
      end   
 
