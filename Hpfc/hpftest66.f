c demonstration paradigme decembre 1994
c
c MAIN
c
      program hpftest66
      integer n, m
      parameter(n=100, m=77)
      integer compute
      external compute
      integer iter, again, last
      print *, 'HPFTEST66 RUNNING'
c
c number of iterations
c
 10   print *, 'PLEASE ENTER THE NUMBER OF ITERATIONS (1-', n, ')'
      read *, iter
      if (iter.lt.1.or.iter.gt.n) goto 10
c
c procedure calls
c
      call init
      last = compute(iter)
      call output(last)
c
c again ?
c
      print *, 'AGAIN (0/1) ?'
      read *, again
      if (again.eq.1) goto 10
      print *, 'HPFTEST66 ENDING'
      end
c
c INIT
c
      subroutine init
      integer n, m
      parameter(n=100, m=77)
      common /arrays/ a(n,m,2), b(n), avm
      real*8 a, b, avm
chpf$ template t(n,m)
chpf$ processors p(2,2)
chpf$ align a(i,j,*) with t(i,j)
chpf$ align b(i) with t(i,1)
chpf$ distribute t(block,block) onto p
      integer i,j
      real*8 tmp
      print *, 'INITIALIZING ARRAY A'
chpf$ independent(j), new(tmp)
      do j=1, m
         tmp = 7.84*j
chpf$ independent(i)
         do i=1, n
            a(i,j,1) = tmp + (3.14/4.2*i)
         enddo
      enddo
chpf$ independent(i)
      do i=1, n
         b(i)=78.0/i
      enddo
      end
c
c COMPUTE
c
      integer function compute(it)
      integer it
      integer n, m
      parameter(n=100, m=77)
      common /arrays/ a(n,m,2), b(n), avm
      real*8 a, b, avm
chpf$ template t(n,m)
chpf$ processors p(2,2)
chpf$ align a(i,j,*) with t(i,j)
chpf$ align b(i) with t(i,1)
chpf$ distribute t(block,block) onto p
      real*8 redmin3
      external redmin3
      integer i,j,time,new,old
      real*8 amin
      print *, 'COMPUTING ', it, ' ITERATIONS'
      new=2
      old=1
      avm = 0.0
      do time=1, it
chpf$ independent(j,i)
         do j=2,m-1
            do i=3,n
               a(i,j,new) = 0.25*
     $              (a(i,j,old) +
     $               a(i-1,j,old) +
     $               a(i-1,j-1,old) +
     $               a(i-2,j+1,old))
            enddo
         enddo
         amin = redmin3(a(1,1,1),3,n,2,m-1,new,new)
         avm = avm + amin
         old=new
         new=3-new
      enddo
      avm = avm/it
      compute=old
      end
c
c OUTPUT
c
      subroutine output(last)
      integer last
      integer n, m
      parameter(n=100, m=77)
      common /arrays/ a(n,m,2), b(n), avm
      real*8 a, b, avm
chpf$ template t(n,m)
chpf$ processors p(2,2)
chpf$ align a(i,j,*) with t(i,j)
chpf$ align b(i) with t(i,1)
chpf$ distribute t(block,block) onto p
      integer i, up
      up=min(n,m)
      print *, 'average min is ', avm
      print *, (a(i,i,last), i=1, up)
      end
c
c REDUCTION MIN 3D
c
      real*8 function redmin3(t,l1,u1,l2,u2,l3,u3)
      integer n, m
      parameter(n=100, m=77)
      integer l1,u1,l2,u2,l3,u3
      real*8 t(n,m,2)
      integer k,j,i
      real*8 result
      result=t(l1,l2,l3)
      do k=l3,u3
         do j=l2,u2
            do i=l1,u1
               if (result.gt.t(i,j,k)) result=t(i,j,k)
            enddo
         enddo
      enddo
      redmin3=result
      end
c
c END
c
