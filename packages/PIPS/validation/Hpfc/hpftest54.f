c SOR 2x2 parameter and flip-flop and real*8
      program hpftest54
c
c version avec 4 processeurs
c
      integer n, times
      parameter (n=500)
      parameter (times=100)
      real*8 temp(n,n,2), north(n), x
      external REDMAX1
      real*8 REDMAX1
      external TIME
      integer TIME
      integer old, new, t1, t2, t3, t4, t5
      integer i, j, k
CHPF$ TEMPLATE t(n,n)
CHPF$ ALIGN temp(i,j,*) with t(i,j)
CHPF$ ALIGN north(i) with t(1,i)
CHPF$ PROCESSORS p(2,2)
CHPF$ DISTRIBUTE t(block,block) ONTO p
      print *, 'HPFTEST54 RUNNING, THERMO'
c
c initialization
c
      print *, 'INITIALIZING'
      t1 = time()
c
chpf$ independent(i)
      do i=1,n
         north(i) = 100.0
      enddo
      do k=1,2
chpf$ independent(i)
         do i=1,n
            temp(1,i,k) = north(i)
         enddo
chpf$ independent(j,i)
         do j=1,n
            do i=2,n
               temp(i,j,k) = 10.0
            enddo
         enddo
      enddo
c
      print *, 'RUNNING'
      t2 = time()
c
      new = 2
      do k=1,times
         old = new
         new = 3-new
chpf$ independent(j,i)         
         do j=2,n-1
            do i=2,n-1
               temp(i,j,new) = 0.25*
     $              (temp(i-1,j,old) + temp(i+1,j,old) + 
     $               temp(i,j-1,old) + temp(i,j+1,old))
            enddo
         enddo
      enddo
c
c print results
c
      print *, 'REDUCTION'
      t3 = time()
c
chpf$ independent(i)
      do i=1, n
         north(i) = temp(2,i,old)
      enddo
      x = REDMAX1(north(1), 1, n)
c
      print *, 'RESULTS:'
      t4 = time()
c
      print *, 'MAX is ', x
 10   format(F8.2, F8.2, F8.2, F8.2, F8.2)
      do i=2, 10, 2
         write (6,10) 
     $        temp(i,12,old), temp(i,24,old), temp(i,36,old),
     $        temp(i,48,old), temp(i,60,old)
      enddo
      print *, 'HPFTEST52 ENDED'
      t5 = time()
c
      print *, 'Timing: init ', t2-t1, ' run ', t3-t2, 
     $     ' red ', t4-t3, ' IO ', t5-t4, ' total ', t5-t1
      end
c
c reduction
c
      real*8 function REDMAX1(a,l,u)
      integer l, u, i
      real*8 a(l:u), amax
      amax = a(l)
      do i=l+1, u
         if (a(i).GT.amax) amax = a(i)
      enddo
      redmax1 = amax
      return
      end
