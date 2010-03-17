c prescriptive mapping
      program prescriptive
      integer n
      parameter(n=20)
      integer A(n,n)
chpf$ processors P(2,2)
chpf$ distribute A(block,block) onto P
      call init(A)
      print *, A(5,15)
      end
c
      subroutine init(Ax)
      integer n, i, j
      parameter(n=20)
      integer Ax(n,n)
chpf$ processors P(4)
chpf$ distribute Ax(*,block) onto P
chpf$ independent(j,i)
      do j=1, n
         do i=1, n
            Ax(i,j) = i+n*j
         enddo
      enddo
      end
