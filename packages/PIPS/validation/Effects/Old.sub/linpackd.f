C introduced a program header by Manju : 17-4-96
C introduced a dummy function second() by Manju : 17-4-96
      PROGRAM Linpackd
      double precision aa(200,200),a(201,200),b(200),x(200)
      double precision time(8,6),cray,ops,total,norma,normx
      double precision resid,residn,eps,epslon
      integer ipvt(200)
      lda = 201
      ldaa = 200
c
      n = 100
      cray = .056
      write(6,1)
    1 format(' Please send the results of this run to:'//
     $       ' Jack J. Dongarra'/
     $       ' Computer Science Department'/
     $       ' University of Tennessee'/
     $       ' Knoxville, Tennessee 37996-1300'//
     $       ' Fax: 615-974-8296'//
     $       ' Internet: dongarra@cs.utk.edu'/)
      ops = (2.0d0*n**3)/3.0d0 + 2.0d0*n**2
c
         call matgen(a,lda,n,b,norma)
         t1 = second()
         call dgefa(a,lda,n,ipvt,info)
         time(1,1) = second() - t1
         t1 = second()
         call dgesl(a,lda,n,ipvt,b,0)
         time(1,2) = second() - t1
         total = time(1,1) + time(1,2)
c
c     compute a residual to verify results.
c
         do 10 i = 1,n
            x(i) = b(i)
   10    continue
         call matgen(a,lda,n,b,norma)
         do 20 i = 1,n
            b(i) = -b(i)
   20    continue
         call dmxpy(n,b,n,lda,x,a)
         resid = 0.0
         normx = 0.0
         do 30 i = 1,n
            resid = dmax1( resid, dabs(b(i)) )
            normx = dmax1( normx, dabs(x(i)) )
   30    continue
         eps = epslon(1.0d0)
         residn = resid/( n*norma*normx*eps )
         write(6,40)
   40    format('     norm. resid      resid           machep',
     $          '         x(1)          x(n)')
         write(6,50) residn,resid,eps,x(1),x(n)
   50    format(1p5e16.8)
c
         write(6,60) n
   60    format(//'    times are reported for matrices of order ',i5)
         write(6,70)
   70    format(6x,'dgefa',6x,'dgesl',6x,'total',5x,'mflops',7x,'unit',
     $         6x,'ratio')
c
         time(1,3) = total
         time(1,4) = ops/(1.0d6*total)
         time(1,5) = 2.0d0/time(1,4)
         time(1,6) = total/cray
         write(6,80) lda
   80    format(' times for array with leading dimension of',i4)
         write(6,110) (time(1,i),i=1,6)
c
         call matgen(a,lda,n,b,norma)
         t1 = second()
         call dgefa(a,lda,n,ipvt,info)
         time(2,1) = second() - t1
         t1 = second()
         call dgesl(a,lda,n,ipvt,b,0)
         time(2,2) = second() - t1
         total = time(2,1) + time(2,2)
         time(2,3) = total
         time(2,4) = ops/(1.0d6*total)
         time(2,5) = 2.0d0/time(2,4)
         time(2,6) = total/cray
c
         call matgen(a,lda,n,b,norma)
         t1 = second()
         call dgefa(a,lda,n,ipvt,info)
         time(3,1) = second() - t1
         t1 = second()
         call dgesl(a,lda,n,ipvt,b,0)
         time(3,2) = second() - t1
         total = time(3,1) + time(3,2)
         time(3,3) = total
         time(3,4) = ops/(1.0d6*total)
         time(3,5) = 2.0d0/time(3,4)
         time(3,6) = total/cray
c
         ntimes = 10
         tm2 = 0
         t1 = second()
         do 90 i = 1,ntimes
            tm = second()
            call matgen(a,lda,n,b,norma)
            tm2 = tm2 + second() - tm
            call dgefa(a,lda,n,ipvt,info)
   90    continue
         time(4,1) = (second() - t1 - tm2)/ntimes
         t1 = second()
         do 100 i = 1,ntimes
            call dgesl(a,lda,n,ipvt,b,0)
  100    continue
         time(4,2) = (second() - t1)/ntimes
         total = time(4,1) + time(4,2)
         time(4,3) = total
         time(4,4) = ops/(1.0d6*total)
         time(4,5) = 2.0d0/time(4,4)
         time(4,6) = total/cray
c
         write(6,110) (time(2,i),i=1,6)
         write(6,110) (time(3,i),i=1,6)
         write(6,110) (time(4,i),i=1,6)
  110    format(6(1pe11.3))
c
         call matgen(aa,ldaa,n,b,norma)
         t1 = second()
         call dgefa(aa,ldaa,n,ipvt,info)
         time(5,1) = second() - t1
         t1 = second()
         call dgesl(aa,ldaa,n,ipvt,b,0)
         time(5,2) = second() - t1
         total = time(5,1) + time(5,2)
         time(5,3) = total
         time(5,4) = ops/(1.0d6*total)
         time(5,5) = 2.0d0/time(5,4)
         time(5,6) = total/cray
c
         call matgen(aa,ldaa,n,b,norma)
         t1 = second()
         call dgefa(aa,ldaa,n,ipvt,info)
         time(6,1) = second() - t1
         t1 = second()
         call dgesl(aa,ldaa,n,ipvt,b,0)
         time(6,2) = second() - t1
         total = time(6,1) + time(6,2)
         time(6,3) = total
         time(6,4) = ops/(1.0d6*total)
         time(6,5) = 2.0d0/time(6,4)
         time(6,6) = total/cray
c
         call matgen(aa,ldaa,n,b,norma)
         t1 = second()
         call dgefa(aa,ldaa,n,ipvt,info)
         time(7,1) = second() - t1
         t1 = second()
         call dgesl(aa,ldaa,n,ipvt,b,0)
         time(7,2) = second() - t1
         total = time(7,1) + time(7,2)
         time(7,3) = total
         time(7,4) = ops/(1.0d6*total)
         time(7,5) = 2.0d0/time(7,4)
         time(7,6) = total/cray
c
         ntimes = 10
         tm2 = 0
         t1 = second()
         do 120 i = 1,ntimes
            tm = second()
            call matgen(aa,ldaa,n,b,norma)
            tm2 = tm2 + second() - tm
            call dgefa(aa,ldaa,n,ipvt,info)
  120    continue
         time(8,1) = (second() - t1 - tm2)/ntimes
         t1 = second()
         do 130 i = 1,ntimes
            call dgesl(aa,ldaa,n,ipvt,b,0)
  130    continue
         time(8,2) = (second() - t1)/ntimes
         total = time(8,1) + time(8,2)
         time(8,3) = total
         time(8,4) = ops/(1.0d6*total)
         time(8,5) = 2.0d0/time(8,4)
         time(8,6) = total/cray
c
         write(6,140) ldaa
  140    format(/' times for array with leading dimension of',i4)
         write(6,110) (time(5,i),i=1,6)
         write(6,110) (time(6,i),i=1,6)
         write(6,110) (time(7,i),i=1,6)
         write(6,110) (time(8,i),i=1,6)
      stop
      end

      subroutine second()
      INTEGER k
C dummy computation
      k = 1
      return
      end

      subroutine matgen(a,lda,n,b,norma)
      double precision a(lda,1),b(1),norma
c
      init = 1325
      norma = 0.0
      do 30 j = 1,n
         do 20 i = 1,n
            init = mod(3125*init,65536)
            a(i,j) = (init - 32768.0)/16384.0
            norma = dmax1(dabs(a(i,j)), norma)
   20    continue
   30 continue
      do 35 i = 1,n
          b(i) = 0.0
   35 continue
      do 50 j = 1,n
         do 40 i = 1,n
            b(i) = b(i) + a(i,j)
   40    continue
   50 continue
      return
      end
      subroutine dgefa(a,lda,n,ipvt,info)
      integer lda,n,ipvt(1),info
      double precision a(lda,1)
c
c     dgefa factors a double precision matrix by gaussian elimination.
c
c     dgefa is usually called by dgeco, but it can be called
c     directly with a saving in time if  rcond  is not needed.
c     (time for dgeco) = (1 + 9/n)*(time for dgefa) .
c
c     on entry
c
c        a       double precision(lda, n)
c                the matrix to be factored.
c
c        lda     integer
c                the leading dimension of the array  a .
c
c        n       integer
c                the order of the matrix  a .
c
c     on return
c
c        a       an upper triangular matrix and the multipliers
c                which were used to obtain it.
c                the factorization can be written  a = l*u  where
c                l  is a product of permutation and unit lower
c                triangular matrices and  u  is upper triangular.
c
c        ipvt    integer(n)
c                an integer vector of pivot indices.
c
c        info    integer
c                = 0  normal value.
c                = k  if  u(k,k) .eq. 0.0 .  this is not an error
c                     condition for this subroutine, but it does
c                     indicate that dgesl or dgedi will divide by zero
c                     if called.  use  rcond  in dgeco for a reliable
c                     indication of singularity.
c
c     linpack. this version dated 08/14/78 .
c     cleve moler, university of new mexico, argonne national lab.
c
c     subroutines and functions
c
c     blas daxpy,dscal,idamax
c
c     internal variables
c
      double precision t
      integer idamax,j,k,kp1,l,nm1
c
c
c     gaussian elimination with partial pivoting
c
      info = 0
      nm1 = n - 1
      if (nm1 .lt. 1) go to 70
      do 60 k = 1, nm1
         kp1 = k + 1
c
c        find l = pivot index
c
         l = idamax(n-k+1,a(k,k),1) + k - 1
         ipvt(k) = l
c
c        zero pivot implies this column already triangularized
c
         if (a(l,k) .eq. 0.0d0) go to 40
c
c           interchange if necessary
c
            if (l .eq. k) go to 10
               t = a(l,k)
               a(l,k) = a(k,k)
               a(k,k) = t
   10       continue
c
c           compute multipliers
c
            t = -1.0d0/a(k,k)
            call dscal(n-k,t,a(k+1,k),1)
c
c           row elimination with column indexing
c
            do 30 j = kp1, n
               t = a(l,j)
               if (l .eq. k) go to 20
                  a(l,j) = a(k,j)
                  a(k,j) = t
   20          continue
               call daxpy(n-k,t,a(k+1,k),1,a(k+1,j),1)
   30       continue
         go to 50
   40    continue
            info = k
   50    continue
   60 continue
   70 continue
      ipvt(n) = n
      if (a(n,n) .eq. 0.0d0) info = n
      return
      end
      subroutine dgesl(a,lda,n,ipvt,b,job)
      integer lda,n,ipvt(1),job
      double precision a(lda,1),b(1)
c
c     dgesl solves the double precision system
c     a * x = b  or  trans(a) * x = b
c     using the factors computed by dgeco or dgefa.
c
c     on entry
c
c        a       double precision(lda, n)
c                the output from dgeco or dgefa.
c
c        lda     integer
c                the leading dimension of the array  a .
c
c        n       integer
c                the order of the matrix  a .
c
c        ipvt    integer(n)
c                the pivot vector from dgeco or dgefa.
c
c        b       double precision(n)
c                the right hand side vector.
c
c        job     integer
c                = 0         to solve  a*x = b ,
c                = nonzero   to solve  trans(a)*x = b  where
c                            trans(a)  is the transpose.
c
c     on return
c
c        b       the solution vector  x .
c
c     error condition
c
c        a division by zero will occur if the input factor contains a
c        zero on the diagonal.  technically this indicates singularity
c        but it is often caused by improper arguments or improper
c        setting of lda .  it will not occur if the subroutines are
c        called correctly and if dgeco has set rcond .gt. 0.0
c        or dgefa has set info .eq. 0 .
c
c     to compute  inverse(a) * c  where  c  is a matrix
c     with  p  columns
c           call dgeco(a,lda,n,ipvt,rcond,z)
c           if (rcond is too small) go to ...
c           do 10 j = 1, p
c              call dgesl(a,lda,n,ipvt,c(1,j),0)
c        10 continue
c
c     linpack. this version dated 08/14/78 .
c     cleve moler, university of new mexico, argonne national lab.
c
c     subroutines and functions
c
c     blas daxpy,ddot
c
c     internal variables
c
      double precision ddot,t
      integer k,kb,l,nm1
c
      nm1 = n - 1
      if (job .ne. 0) go to 50
c
c        job = 0 , solve  a * x = b
c        first solve  l*y = b
c
         if (nm1 .lt. 1) go to 30
         do 20 k = 1, nm1
            l = ipvt(k)
            t = b(l)
            if (l .eq. k) go to 10
               b(l) = b(k)
               b(k) = t
   10       continue
            call daxpy(n-k,t,a(k+1,k),1,b(k+1),1)
   20    continue
   30    continue
c
c        now solve  u*x = y
c
         do 40 kb = 1, n
            k = n + 1 - kb
            b(k) = b(k)/a(k,k)
            t = -b(k)
            call daxpy(k-1,t,a(1,k),1,b(1),1)
   40    continue
      go to 100
   50 continue
c
c        job = nonzero, solve  trans(a) * x = b
c        first solve  trans(u)*y = b
c
         do 60 k = 1, n
            t = ddot(k-1,a(1,k),1,b(1),1)
            b(k) = (b(k) - t)/a(k,k)
   60    continue
c
c        now solve trans(l)*x = y
c
         if (nm1 .lt. 1) go to 90
         do 80 kb = 1, nm1
            k = n - kb
            b(k) = b(k) + ddot(n-k,a(k+1,k),1,b(k+1),1)
            l = ipvt(k)
            if (l .eq. k) go to 70
               t = b(l)
               b(l) = b(k)
               b(k) = t
   70       continue
   80    continue
   90    continue
  100 continue
      return
      end
      subroutine daxpy(n,da,dx,incx,dy,incy)
c
c     constant times a vector plus a vector.
c     jack dongarra, linpack, 3/11/78.
c
      double precision dx(1),dy(1),da
      integer i,incx,incy,ix,iy,m,mp1,n
c
      if(n.le.0)return
      if (da .eq. 0.0d0) return
      if(incx.eq.1.and.incy.eq.1)go to 20
c
c        code for unequal increments or equal increments
c          not equal to 1
c
      ix = 1
      iy = 1
      if(incx.lt.0)ix = (-n+1)*incx + 1
      if(incy.lt.0)iy = (-n+1)*incy + 1
      do 10 i = 1,n
        dy(iy) = dy(iy) + da*dx(ix)
        ix = ix + incx
        iy = iy + incy
   10 continue
      return
c
c        code for both increments equal to 1
c
   20 continue
      do 30 i = 1,n
        dy(i) = dy(i) + da*dx(i)
   30 continue
      return
      end
      double precision function ddot(n,dx,incx,dy,incy)
c
c     forms the dot product of two vectors.
c     jack dongarra, linpack, 3/11/78.
c
      double precision dx(1),dy(1),dtemp
      integer i,incx,incy,ix,iy,m,mp1,n
c
      ddot = 0.0d0
      dtemp = 0.0d0
      if(n.le.0)return
      if(incx.eq.1.and.incy.eq.1)go to 20
c
c        code for unequal increments or equal increments
c          not equal to 1
c
      ix = 1
      iy = 1
      if(incx.lt.0)ix = (-n+1)*incx + 1
      if(incy.lt.0)iy = (-n+1)*incy + 1
      do 10 i = 1,n
        dtemp = dtemp + dx(ix)*dy(iy)
        ix = ix + incx
        iy = iy + incy
   10 continue
      ddot = dtemp
      return
c
c        code for both increments equal to 1
c
   20 continue
      do 30 i = 1,n
        dtemp = dtemp + dx(i)*dy(i)
   30 continue
      ddot = dtemp
      return
      end
      subroutine  dscal(n,da,dx,incx)
c
c     scales a vector by a constant.
c     jack dongarra, linpack, 3/11/78.
c
      double precision da,dx(1)
      integer i,incx,m,mp1,n,nincx
c
      if(n.le.0)return
      if(incx.eq.1)go to 20
c
c        code for increment not equal to 1
c
      nincx = n*incx
      do 10 i = 1,nincx,incx
        dx(i) = da*dx(i)
   10 continue
      return
c
c        code for increment equal to 1
c
   20 continue
      do 30 i = 1,n
        dx(i) = da*dx(i)
   30 continue
      return
      end
      integer function idamax(n,dx,incx)
c
c     finds the index of element having max. dabsolute value.
c     jack dongarra, linpack, 3/11/78.
c
      double precision dx(1),dmax
      integer i,incx,ix,n
c
      idamax = 0
      if( n .lt. 1 ) return
      idamax = 1
      if(n.eq.1)return
      if(incx.eq.1)go to 20
c
c        code for increment not equal to 1
c
      ix = 1
      dmax = dabs(dx(1))
      ix = ix + incx
      do 10 i = 2,n
         if(dabs(dx(ix)).le.dmax) go to 5
         idamax = i
         dmax = dabs(dx(ix))
    5    ix = ix + incx
   10 continue
      return
c
c        code for increment equal to 1
c
   20 dmax = dabs(dx(1))
      do 30 i = 2,n
         if(dabs(dx(i)).le.dmax) go to 30
         idamax = i
         dmax = dabs(dx(i))
   30 continue
      return
      end
      double precision function epslon (x)
      double precision x
c
c     estimate unit roundoff in quantities of size x.
c
      double precision a,b,c,eps
c
c     this program should function properly on all systems
c     satisfying the following two assumptions,
c        1.  the base used in representing dfloating point
c            numbers is not a power of three.
c        2.  the quantity  a  in statement 10 is represented to 
c            the accuracy used in dfloating point variables
c            that are stored in memory.
c     the statement number 10 and the go to 10 are intended to
c     force optimizing compilers to generate code satisfying 
c     assumption 2.
c     under these assumptions, it should be true that,
c            a  is not exactly equal to four-thirds,
c            b  has a zero for its last bit or digit,
c            c  is not exactly equal to one,
c            eps  measures the separation of 1.0 from
c                 the next larger dfloating point number.
c     the developers of eispack would appreciate being informed
c     about any systems where these assumptions do not hold.
c
c     *****************************************************************
c     this routine is one of the auxiliary routines used by eispack iii
c     to avoid machine dependencies.
c     *****************************************************************
c
c     this version dated 4/6/83.
c
      a = 4.0d0/3.0d0
   10 b = a - 1.0d0
      c = b + b + b
      eps = dabs(c-1.0d0)
      if (eps .eq. 0.0d0) go to 10
      epslon = eps*dabs(x)
      return
      end
      subroutine dmxpy (n1, y, n2, ldm, x, m)
      double precision y(*), x(*), m(ldm,*)
c
c   purpose:
c     multiply matrix m times vector x and add the result to vector y.
c
c   parameters:
c
c     n1 integer, number of elements in vector y, and number of rows in
c         matrix m
c
c     y double precision(n1), vector of length n1 to which is added 
c         the product m*x
c
c     n2 integer, number of elements in vector x, and number of columns
c         in matrix m
c
c     ldm integer, leading dimension of array m
c
c     x double precision(n2), vector of length n2
c
c     m double precision(ldm,n2), matrix of n1 rows and n2 columns
c
c ----------------------------------------------------------------------
c
c   cleanup odd vector
c
      j = mod(n2,2)
      if (j .ge. 1) then
         do 10 i = 1, n1
            y(i) = (y(i)) + x(j)*m(i,j)
   10    continue
      endif
c
c   cleanup odd group of two vectors
c
      j = mod(n2,4)
      if (j .ge. 2) then
         do 20 i = 1, n1
            y(i) = ( (y(i))
     $             + x(j-1)*m(i,j-1)) + x(j)*m(i,j)
   20    continue
      endif
c
c   cleanup odd group of four vectors
c
      j = mod(n2,8)
      if (j .ge. 4) then
         do 30 i = 1, n1
            y(i) = ((( (y(i))
     $             + x(j-3)*m(i,j-3)) + x(j-2)*m(i,j-2))
     $             + x(j-1)*m(i,j-1)) + x(j)  *m(i,j)
   30    continue
      endif
c
c   cleanup odd group of eight vectors
c
      j = mod(n2,16)
      if (j .ge. 8) then
         do 40 i = 1, n1
            y(i) = ((((((( (y(i))
     $             + x(j-7)*m(i,j-7)) + x(j-6)*m(i,j-6))
     $             + x(j-5)*m(i,j-5)) + x(j-4)*m(i,j-4))
     $             + x(j-3)*m(i,j-3)) + x(j-2)*m(i,j-2))
     $             + x(j-1)*m(i,j-1)) + x(j)  *m(i,j)
   40    continue
      endif
c
c   main loop - groups of sixteen vectors
c
      jmin = j+16
      do 60 j = jmin, n2, 16
         do 50 i = 1, n1
            y(i) = ((((((((((((((( (y(i))
     $             + x(j-15)*m(i,j-15)) + x(j-14)*m(i,j-14))
     $             + x(j-13)*m(i,j-13)) + x(j-12)*m(i,j-12))
     $             + x(j-11)*m(i,j-11)) + x(j-10)*m(i,j-10))
     $             + x(j- 9)*m(i,j- 9)) + x(j- 8)*m(i,j- 8))
     $             + x(j- 7)*m(i,j- 7)) + x(j- 6)*m(i,j- 6))
     $             + x(j- 5)*m(i,j- 5)) + x(j- 4)*m(i,j- 4))
     $             + x(j- 3)*m(i,j- 3)) + x(j- 2)*m(i,j- 2))
     $             + x(j- 1)*m(i,j- 1)) + x(j)   *m(i,j)
   50    continue
   60 continue
      return
      end
