c square matrix multiplication
      program matmul
      integer n
      parameter (n=400)
      real*8 a(n,n), b(n,n), c(n,n)
      integer i, j, k, l, u
c
c initial mapping: A, B and C are aligned and block distributed
c
chpf$ dynamic a, b
chpf$ template t(n,n)
chpf$ processors p(2,2)
chpf$ distribute t(block,block) onto p
chpf$ align a, b, c with t
c
c initialization of array A and B
c
chpf$ independent(j,i)
      do j=1, n
         do i=1, n
            a(i,j) = real(i-(n/2))/real(j)
            b(i,j) = real(j-3)/real(i)
         enddo
      enddo
c
c matrix multiply: C=A*B
c a remapping is needed to ensure data locality
c
cfcd$ timeon
chpf$ realign a(i,*) with t(i,*)
chpf$ realign b(*,j) with t(*,j)
chpf$ independent(j,i)
      do j=1, n
         do i=1, n
            c(i,j) = 0.
            do k=1, n
               c(i,j) = c(i,j) + a(i,k)*b(k,j)
            enddo
         enddo
      enddo
cfcd$ timeoff('matrix multiplication')
c
c output of the result
c
      l = (n/2)-4
      u = (n/2)+5
      print *, ((c(i,j), i=l, u), j=l, u)
cfcd$ timeon
chpf$ realign a, b with t
cfcd$ timeoff('back to initial alignment')
      print *, a(5,5), b(5,5)
      end
