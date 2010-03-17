      program applu
c
c***driver for the performance evaluation of the solver for
c   five coupled parabolic/elliptic partial differential equations.
c
c Author: Sisira Weeratunga
c         NASA Ames Research Center
c         (10/25/90)
c
c Changes: Jeff Reilly, 9/25/94
c	   All write statements have output device changed from
c	   output devide "iout" to "6" (stdout). 


c #include 'applu.incl'  

      implicit real*8 (a-h,o-z)
c
      parameter ( c1 = 1.40d+00, c2 = 0.40d+00,
     $            c3 = 1.00d-01, c4 = 1.00d+00,
     $            c5 = 1.40d+00 )
c
c***grid
c
      common/cgcon/ nx, ny, nz,
     $              ii1, ii2, ji1, ji2, ki1, ki2, itwj,
     $              dxi, deta, dzeta,
     $              tx1, tx2, tx3,
     $              ty1, ty2, ty3,
     $              tz1, tz2, tz3
c
c***dissipation
c
      common/disp/ dx1,dx2,dx3,dx4,dx5,
     $             dy1,dy2,dy3,dy4,dy5,
     $             dz1,dz2,dz3,dz4,dz5,
     $             dssp
c
c***field variables and residuals
c
      common/cvar/ u(5,33,33,33),
     $             rsd(5,33,33,33),
     $             frct(5,33,33,33)
c
c***output control parameters
c
      common/cprcon/ ipr, iout, inorm
c
c***newton-raphson iteration control parameters
c
      common/ctscon/ itmax, invert,
     $               dt, omega, tolrsd(5),
     $               rsdnm(5), errnm(5), frc, ttotal
c
      common/cjac/ a(5,5,33,33,33),
     $             b(5,5,33,33,33),
     $             c(5,5,33,33,33),
     $             d(5,5,33,33,33)
c
c***coefficients of the exact solution
c
      common/cexact/ ce(5,13)
c



      WRITE(6,*) 'Version: %Z%'

c
c***open file for input data
c      open (unit=5,file= 'applu.inp33',status='old',
c
c      open (unit=5,file='applu.in',status='old',
c     *      access='sequential',form='formatted')
c      rewind 5
c
c***read the unit number for output data
c
      read (5,*)
      read (5,*)
      read (5,*) iout
c
c***flag that controls printing of the progress of iterations
c
      read (5,*)
      read (5,*)
      read (5,*) ipr, inorm
c
c***set the maximum number of pseudo-time steps to be taken
c
      read (5,*)
      read (5,*)
      read (5,*) itmax
c
c***set the magnitude of the time step
c
      read (5,*)
      read (5,*)
      read (5,*) dt
c
c***set the method of inverting the jacobian martix
c   (invert = 1 : use Block approximate factorization method,
c    invert = 2 : use Diagonalized approximate factorization method,
c    invert = 3 : use SSOR methd)
c
      read (5,*)
      read (5,*)
      read (5,*) invert
c
c***set the value of over-relaxation factor for SSOR iterations
c
      read (5,*)
      read (5,*)
      read (5,*) omega
c
c***set the steady-state residual tolerance levels
c
      read (5,*)
      read (5,*)
      read (5,*) tolrsd(1),tolrsd(2),
     $           tolrsd(3),tolrsd(4),tolrsd(5)
c
c***read problem specification parameters
c
c
c***specify the number of grid points in xi, eta and zeta directions
c
      read (5,*)
      read (5,*)
      read (5,*) nx, ny, nz
c
c***open the file for output data
c
c      if ( iout .eq. 7 ) then
c
c         open ( unit = 7, file = 'output.data', status = 'unknown',
c     $         access = 'sequential', form = 'formatted' )
c         rewind 7
c
c      end if
c
      if ( ( nx .lt. 5 ) .or.
     $     ( ny .lt. 5 ) .or.
     $     ( nz .lt. 5 ) ) then
c
         write (6,2001)
 2001    format (5x,'PROBLEM SIZE IS TOO SMALL - ',
     $        /5x,'SET EACH OF NX, NY AND NZ AT LEAST EQUAL TO 5')
         stop
c
      end if
c
      if ( ( nx .gt. 33 ) .or.
     $     ( ny .gt. 33 ) .or.
     $     ( nz .gt. 33 ) ) then
c
         write (6,2002)
 2002    format (5x,'PROBLEM SIZE IS TOO LARGE - ',
     $        /5x,'NX, NY AND NZ SHOULD BE LESS THAN OR EQUAL TO ',
     $        /5x,'ISIZ1, ISIZ2 AND ISIZ3 RESPECTIVELY')
c
      end if
c
      dxi = 1.0d+00 / ( nx - 1 )
      deta = 1.0d+00 / ( ny - 1 )
      dzeta = 1.0d+00 / ( nz - 1 )
c
      tx1 = 1.0d+00 / ( dxi * dxi )
      tx2 = 1.0d+00 / ( 2.0d+00 * dxi )
      tx3 = 1.0d+00 / dxi
c
      ty1 = 1.0d+00 / ( deta * deta )
      ty2 = 1.0d+00 / ( 2.0d+00 * deta )
      ty3 = 1.0d+00 / deta
c
      tz1 = 1.0d+00 / ( dzeta * dzeta )
      tz2 = 1.0d+00 / ( 2.0d+00 * dzeta )
      tz3 = 1.0d+00 / dzeta
c
      ii1 = 2
      ii2 = nx - 1
      ji1 = 2
      ji2 = ny - 2
      ki1 = 3
      ki2 = nz - 1
      itwj = 0
c
c***diffusion coefficients
c
      dx1 = 0.75d+00
      dx2 = dx1
      dx3 = dx1
      dx4 = dx1
      dx5 = dx1
c
      dy1 = 0.75d+00
      dy2 = dy1
      dy3 = dy1
      dy4 = dy1
      dy5 = dy1
c
      dz1 = 1.00d+00
      dz2 = dz1
      dz3 = dz1
      dz4 = dz1 
      dz5 = dz1
c
c***fourth difference dissipation 
c
      dssp = ( max (dx1, dy1, dz1 ) ) / 4.0d+00
c
c***coefficients of the exact solution to the first pde
c
      ce(1,1) = 2.0d+00
      ce(1,2) = 0.0d+00
      ce(1,3) = 0.0d+00
      ce(1,4) = 4.0d+00
      ce(1,5) = 5.0d+00
      ce(1,6) = 3.0d+00
      ce(1,7) = 5.0d-01
      ce(1,8) = 2.0d-02
      ce(1,9) = 1.0d-02
      ce(1,10) = 3.0d-02
      ce(1,11) = 5.0d-01
      ce(1,12) = 4.0d-01
      ce(1,13) = 3.0d-01
c
c***coefficients of the exact solution to the second pde
c
      ce(2,1) = 1.0d+00
      ce(2,2) = 0.0d+00
      ce(2,3) = 0.0d+00
      ce(2,4) = 0.0d+00
      ce(2,5) = 1.0d+00
      ce(2,6) = 2.0d+00
      ce(2,7) = 3.0d+00
      ce(2,8) = 1.0d-02
      ce(2,9) = 3.0d-02
      ce(2,10) = 2.0d-02
      ce(2,11) = 4.0d-01
      ce(2,12) = 3.0d-01
      ce(2,13) = 5.0d-01
c
c***coefficients of the exact solution to the third pde
c
      ce(3,1) = 2.0d+00
      ce(3,2) = 2.0d+00
      ce(3,3) = 0.0d+00
      ce(3,4) = 0.0d+00
      ce(3,5) = 0.0d+00
      ce(3,6) = 2.0d+00
      ce(3,7) = 3.0d+00
      ce(3,8) = 4.0d-02
      ce(3,9) = 3.0d-02
      ce(3,10) = 5.0d-02
      ce(3,11) = 3.0d-01
      ce(3,12) = 5.0d-01
      ce(3,13) = 4.0d-01
c
c***coefficients of the exact solution to the fourth pde
c
      ce(4,1) = 2.0d+00
      ce(4,2) = 2.0d+00
      ce(4,3) = 0.0d+00
      ce(4,4) = 0.0d+00
      ce(4,5) = 0.0d+00
      ce(4,6) = 2.0d+00
      ce(4,7) = 3.0d+00
      ce(4,8) = 3.0d-02
      ce(4,9) = 5.0d-02
      ce(4,10) = 4.0d-02
      ce(4,11) = 2.0d-01
      ce(4,12) = 1.0d-01
      ce(4,13) = 3.0d-01
c
c***coefficients of the exact solution to the fifth pde
c
      ce(5,1) = 5.0d+00
      ce(5,2) = 4.0d+00
      ce(5,3) = 3.0d+00
      ce(5,4) = 2.0d+00
      ce(5,5) = 1.0d-01
      ce(5,6) = 4.0d-01
      ce(5,7) = 3.0d-01
      ce(5,8) = 5.0d-02
      ce(5,9) = 4.0d-02
      ce(5,10) = 3.0d-02
      ce(5,11) = 1.0d-01
      ce(5,12) = 3.0d-01
      ce(5,13) = 2.0d-01
c
c***set the boundary values for dependent variables
c
      call setbv
c
c***set the initial values for dependent variables
c
      call setiv
c
c***compute the forcing term based on prescribed exact solution
c
      call erhs
c
c***perform the SSOR iterations
c
      call ssor
c
c***compute the solution error
c
      call error
c
c***compute the surface integral
c
      call pintgr
c
c***verification test
c
      call verify ( rsdnm, errnm, frc )
c
c***print the CPU time
c
c
      stop
      end
c
c
c
c
c
      subroutine setbv
c
c***set the boundary values of dependent variables
c
c Author: Sisira Weeratunga
c         NASA Ames Research Center
c         (10/25/90)
c
c #include 'applu.incl'  

      implicit real*8 (a-h,o-z)
c
      parameter ( c1 = 1.40d+00, c2 = 0.40d+00,
     $            c3 = 1.00d-01, c4 = 1.00d+00,
     $            c5 = 1.40d+00 )
c
c***grid
c
      common/cgcon/ nx, ny, nz,
     $              ii1, ii2, ji1, ji2, ki1, ki2, itwj,
     $              dxi, deta, dzeta,
     $              tx1, tx2, tx3,
     $              ty1, ty2, ty3,
     $              tz1, tz2, tz3
c
c***dissipation
c
      common/disp/ dx1,dx2,dx3,dx4,dx5,
     $             dy1,dy2,dy3,dy4,dy5,
     $             dz1,dz2,dz3,dz4,dz5,
     $             dssp
c
c***field variables and residuals
c
      common/cvar/ u(5,33,33,33),
     $             rsd(5,33,33,33),
     $             frct(5,33,33,33)
c
c***output control parameters
c
      common/cprcon/ ipr, iout, inorm
c
c***newton-raphson iteration control parameters
c
      common/ctscon/ itmax, invert,
     $               dt, omega, tolrsd(5),
     $               rsdnm(5), errnm(5), frc, ttotal
c
      common/cjac/ a(5,5,33,33,33),
     $             b(5,5,33,33,33),
     $             c(5,5,33,33,33),
     $             d(5,5,33,33,33)
c
c***coefficients of the exact solution
c
      common/cexact/ ce(5,13)
c



c
c***set the dependent variable values along the top and bottom faces
c
      do j = 1, ny
         do i = 1, nx
c
            call exact ( i, j, 1, u( 1, i, j, 1 ) )
            call exact ( i, j, nz, u( 1, i, j, nz ) )
c
         end do
      end do
c
c***set the dependent variable values along north and south faces
c
      do k = 1, nz
         do i = 1, nx
c
            call exact ( i, 1, k, u( 1, i, 1, k ) )
            call exact ( i, ny, k, u( 1, i, ny, k ) )
c
         end do
      end do
c
c***set the dependent variable values along east and west faces
c
      do k = 1, nz
         do j = 1, ny
c
            call exact ( 1, j, k, u( 1, 1, j, k ) )
            call exact ( nx, j, k, u( 1, nx, j, k ) )
c
         end do
      end do
c
      return
      end
c
c
c
c
c
      subroutine setiv
c
c***set the initial values of independent variables based on tri-linear
c   interpolation of boundary values in the computational space.
c
c Author: Sisira Weeratunga
c         NASA Ames Research Center
c         (10/25/90)
c
c #include 'applu.incl'  

      implicit real*8 (a-h,o-z)
c
      parameter ( c1 = 1.40d+00, c2 = 0.40d+00,
     $            c3 = 1.00d-01, c4 = 1.00d+00,
     $            c5 = 1.40d+00 )
c
c***grid
c
      common/cgcon/ nx, ny, nz,
     $              ii1, ii2, ji1, ji2, ki1, ki2, itwj,
     $              dxi, deta, dzeta,
     $              tx1, tx2, tx3,
     $              ty1, ty2, ty3,
     $              tz1, tz2, tz3
c
c***dissipation
c
      common/disp/ dx1,dx2,dx3,dx4,dx5,
     $             dy1,dy2,dy3,dy4,dy5,
     $             dz1,dz2,dz3,dz4,dz5,
     $             dssp
c
c***field variables and residuals
c
      common/cvar/ u(5,33,33,33),
     $             rsd(5,33,33,33),
     $             frct(5,33,33,33)
c
c***output control parameters
c
      common/cprcon/ ipr, iout, inorm
c
c***newton-raphson iteration control parameters
c
      common/ctscon/ itmax, invert,
     $               dt, omega, tolrsd(5),
     $               rsdnm(5), errnm(5), frc, ttotal
c
      common/cjac/ a(5,5,33,33,33),
     $             b(5,5,33,33,33),
     $             c(5,5,33,33,33),
     $             d(5,5,33,33,33)
c
c***coefficients of the exact solution
c
      common/cexact/ ce(5,13)
c



c
      do k = 2, nz-1
         zeta = ( dfloat (k-1) ) / (nz-1)
         do j = 2, ny-1
            eta = ( dfloat (j-1) ) / (ny-1)
            do i = 2, nx-1
               xi = ( dfloat (i-1) ) / (nx-1)
               do m = 1, 5
c
                  pxi =   ( 1.0d+00 - xi ) * u(m,1,j,k)
     $                              + xi   * u(m,nx,j,k)
                  peta =  ( 1.0d+00 - eta ) * u(m,i,1,k)
     $                              + eta   * u(m,i,ny,k)
                  pzeta = ( 1.0d+00 - zeta ) * u(m,i,j,1)
     $                              + zeta   * u(m,i,j,nz)
c
                  u( m, i, j, k ) = pxi + peta + pzeta
     $                 - pxi * peta - peta * pzeta - pzeta * pxi
     $                 + pxi * peta * pzeta
c
               end do
            end do
         end do
      end do
c
      return
      end
c
c
c
c
c
      subroutine blts ( ldmx, ldmy, ldmz,
     $                  nx, ny, nz,
     $                  omega,
     $                  v,
     $                  ldz, ldy, ldx, d )
c
c***compute the regular-sparse, block lower triangular solution:
c
c                     v <-- ( L-inv ) * v
c
c Author: Sisira Weeratunga
c         NASA Ames Research Center
c         (10/25/90)
c
      implicit real*8 ( a-h, o-z )
c
      real*8 ldx, ldy, ldz, d
c
      dimension v( 5, ldmx, ldmy, *),
     $          ldz( 5, 5, ldmx, ldmy, *),
     $          ldy( 5, 5, ldmx, ldmy, *),
     $          ldx( 5, 5, ldmx, ldmy, *),
     $          d( 5, 5, ldmx, ldmy, * )
c
      dimension tmat(5,5)
c
      do k = 2, nz-1
c
         do j = 2, ny-1
c
            do i = 2, nx-1
c
               do m = 1, 5
c
                  do l = 1, 5
c
                     v( m, i, j, k ) =  v( m, i, j, k )
     $    - omega * (  ldz( m, l, i, j, k ) * v( l, i, j, k-1 )
     $               + ldy( m, l, i, j, k ) * v( l, i, j-1, k )
     $               + ldx( m, l, i, j, k ) * v( l, i-1, j, k ) )
c
                  end do
c
               end do
c
c***diagonal block inversion
c
c***forward elimination
c
               do m = 1, 5
                  do l = 1, 5
                     tmat( m, l ) = d( m, l, i, j, k )
                  end do
               end do
c
               do ip = 1, 4
c
                  tmp1 = 1.0d+00 / tmat( ip, ip )
c
                  do m = ip+1, 5
c
                     tmp = tmp1 * tmat( m, ip )
c     
                     do l = ip+1, 5
c
                        tmat( m, l ) =  tmat( m, l )
     $                       - tmp * tmat( ip, l )
c     
                     end do
c
                     v( m, i, j, k ) = v( m, i, j, k )
     $                 - v( ip, i, j, k ) * tmp
c     
                  end do
c
               end do
c
c***back substitution
c
               do m = 5, 1, -1
c
                  do l = m+1, 5
c
                     v( m, i, j, k ) = v( m, i, j, k )
     $                    - tmat( m, l ) * v( l, i, j, k )
c
                  end do
c
                  v( m, i, j, k ) = v( m, i, j, k )
     $                            / tmat( m, m )
c
              end do
c
            end do
c
         end do
c     
      end do
c
      return
      end
c
c
c
c
c
      subroutine buts ( ldmx, ldmy, ldmz,
     $                  nx, ny, nz,
     $                  omega,
     $                  v,
     $                  d, udx, udy, udz )
c
c***compute the regular-sparse, block upper triangular solution:
c
c                     v <-- ( U-inv ) * v
c
c Author: Sisira Weeratunga
c         NASA Ames Research Center
c         (10/25/90)
c
      implicit real*8 ( a-h, o-z )
c
      real*8 udx, udy, udz, d
c
      dimension v( 5, ldmx, ldmy, *),
     $          d( 5, 5, ldmx, ldmy, *),
     $          udx( 5, 5, ldmx, ldmy, *),
     $          udy( 5, 5, ldmx, ldmy, *),
     $          udz( 5, 5, ldmx, ldmy, * )
c
      dimension tmat(5,5), tv(5)
c
      do k = nz-1, 2, -1
c
         do j = ny-1, 2, -1
c
            do i = nx-1, 2, -1
c
               do m = 1, 5
c
                  tv( m ) = 0.0d+00
c
                  do l = 1, 5
c
                     tv( m ) = tv( m ) + 
     $      omega * (  udz( m, l, i, j, k ) * v( l, i, j, k+1 )
     $               + udy( m, l, i, j, k ) * v( l, i, j+1, k )
     $               + udx( m, l, i, j, k ) * v( l, i+1, j, k ) )
c
                  end do
c
               end do
c     
c***diagonal block inversion
c
               do m = 1, 5
                  do l = 1, 5
                     tmat( m, l ) = d( m, l, i, j, k )
                  end do
               end do
c
               do ip = 1, 4
c
                  tmp1 = 1.0d+00 / tmat( ip, ip )
c
                  do m = ip+1, 5
c
                     tmp = tmp1 * tmat( m, ip )
c     
                     do l = ip+1, 5
c
                        tmat( m, l ) =  tmat( m, l )
     $                       - tmp * tmat( ip, l )
c     
                     end do
c
                     tv( m ) = tv( m )
     $                 - tv( ip ) * tmp
c     
                  end do
c
               end do
c
c***back substitution
c
               do m = 5, 1, -1
c
                  do l = m+1, 5
c
                     tv( m ) = tv( m )
     $                    - tmat( m, l ) * tv( l )
c
                  end do
c
                  tv( m ) = tv( m ) / tmat( m, m )
c
               end do
c
               do m = 1, 5
c
                  v( m, i, j, k ) = v( m, i, j, k ) - tv( m )
c
               end do
c
            end do
c
         end do
c     
      end do
c
      return
      end
c
c
c
c
c
      subroutine erhs
c
c***compute the right hand side based on exact solution
c
c Author: Sisira Weeratunga
c         NASA Ames Research Center
c         (10/25/90)
c
c #include 'applu.incl'  

      implicit real*8 (a-h,o-z)
c
      parameter ( c1 = 1.40d+00, c2 = 0.40d+00,
     $            c3 = 1.00d-01, c4 = 1.00d+00,
     $            c5 = 1.40d+00 )
c
c***grid
c
      common/cgcon/ nx, ny, nz,
     $              ii1, ii2, ji1, ji2, ki1, ki2, itwj,
     $              dxi, deta, dzeta,
     $              tx1, tx2, tx3,
     $              ty1, ty2, ty3,
     $              tz1, tz2, tz3
c
c***dissipation
c
      common/disp/ dx1,dx2,dx3,dx4,dx5,
     $             dy1,dy2,dy3,dy4,dy5,
     $             dz1,dz2,dz3,dz4,dz5,
     $             dssp
c
c***field variables and residuals
c
      common/cvar/ u(5,33,33,33),
     $             rsd(5,33,33,33),
     $             frct(5,33,33,33)
c
c***output control parameters
c
      common/cprcon/ ipr, iout, inorm
c
c***newton-raphson iteration control parameters
c
      common/ctscon/ itmax, invert,
     $               dt, omega, tolrsd(5),
     $               rsdnm(5), errnm(5), frc, ttotal
c
      common/cjac/ a(5,5,33,33,33),
     $             b(5,5,33,33,33),
     $             c(5,5,33,33,33),
     $             d(5,5,33,33,33)
c
c***coefficients of the exact solution
c
      common/cexact/ ce(5,13)
c



c
      dimension flux(5,33), ue(5,33)
c
      dsspm = dssp
c
      do k = 1, nz
         do j = 1, ny
            do i = 1, nx
               do m = 1, 5
                  frct( m, i, j, k ) = 0.0d+00
               end do
            end do
         end do
      end do
c
c***xi-direction flux differences
c
      do k = 2, nz - 1
c
         zeta = ( dfloat(k-1) ) / ( nz - 1 )
c
         do j = 2, ny - 1
c
            eta = ( dfloat(j-1) ) / ( ny - 1 )
c
            do i = 1, nx
c
               xi = ( dfloat(i-1) ) / ( nx - 1 )
c
               do m = 1, 5
c
                  ue(m,i) =  ce(m,1)
     $                 + ce(m,2) * xi
     $                 + ce(m,3) * eta
     $                 + ce(m,4) * zeta
     $                 + ce(m,5) * xi * xi
     $                 + ce(m,6) * eta * eta
     $                 + ce(m,7) * zeta * zeta
     $                 + ce(m,8) * xi * xi * xi
     $                 + ce(m,9) * eta * eta * eta
     $                 + ce(m,10) * zeta * zeta * zeta
     $                 + ce(m,11) * xi * xi * xi * xi
     $                 + ce(m,12) * eta * eta * eta * eta
     $                 + ce(m,13) * zeta * zeta * zeta * zeta
c
               end do
c
               flux(1,i) = ue(2,i)
c
               u21 = ue(2,i) / ue(1,i)
c
               q = 0.50d+00 * (  ue(2,i) * ue(2,i)
     $                         + ue(3,i) * ue(3,i)
     $                         + ue(4,i) * ue(4,i) )
     $                      / ue(1,i)
c
               flux(2,i) = ue(2,i) * u21 + c2 * ( ue(5,i) - q )
c
               flux(3,i) = ue(3,i) * u21
c
               flux(4,i) = ue(4,i) * u21
c
               flux(5,i) = ( c1 * ue(5,i) - c2 * q ) * u21
c    
            end do
c
            do i = 2, nx - 1
c
               do m = 1, 5
c
                  frct(m,i,j,k) =  frct(m,i,j,k)
     $                       - tx2 * ( flux(m,i+1) - flux(m,i-1) )
c
               end do
c
            end do
c
            do i = 2, nx
c
               tmp = 1.0d+00 / ue(1,i)
c
               u21i = tmp * ue(2,i)
               u31i = tmp * ue(3,i)
               u41i = tmp * ue(4,i)
               u51i = tmp * ue(5,i)
c
               tmp = 1.0d+00 / ue(1,i-1)
c
               u21im1 = tmp * ue(2,i-1)
               u31im1 = tmp * ue(3,i-1)
               u41im1 = tmp * ue(4,i-1)
               u51im1 = tmp * ue(5,i-1)
c
               flux(2,i) = (4.0d+00/3.0d+00) * tx3 * ( u21i - u21im1 )
               flux(3,i) = tx3 * ( u31i - u31im1 )
               flux(4,i) = tx3 * ( u41i - u41im1 )
               flux(5,i) = 0.50d+00 * ( 1.0d+00 - c1*c5 )
     $              * tx3 * ( ( u21i  **2 + u31i  **2 + u41i  **2 )
     $                      - ( u21im1**2 + u31im1**2 + u41im1**2 ) )
     $              + (1.0d+00/6.0d+00)
     $              * tx3 * ( u21i**2 - u21im1**2 )
     $              + c1 * c5 * tx3 * ( u51i - u51im1 )
c
            end do
c
            do i = 2, nx - 1
c
               frct(1,i,j,k) = frct(1,i,j,k)
     $              + dx1 * tx1 * (            ue(1,i-1)
     $                             - 2.0d+00 * ue(1,i)
     $                             +           ue(1,i+1) )
c
               frct(2,i,j,k) = frct(2,i,j,k)
     $              + tx3 * c3 * c4 * ( flux(2,i+1) - flux(2,i) )
     $              + dx2 * tx1 * (            ue(2,i-1)
     $                             - 2.0d+00 * ue(2,i)
     $                             +           ue(2,i+1) )
c
               frct(3,i,j,k) = frct(3,i,j,k)
     $              + tx3 * c3 * c4 * ( flux(3,i+1) - flux(3,i) )
     $              + dx3 * tx1 * (            ue(3,i-1)
     $                             - 2.0d+00 * ue(3,i)
     $                             +           ue(3,i+1) )
c
               frct(4,i,j,k) = frct(4,i,j,k)
     $              + tx3 * c3 * c4 * ( flux(4,i+1) - flux(4,i) )
     $              + dx4 * tx1 * (            ue(4,i-1)
     $                             - 2.0d+00 * ue(4,i)
     $                             +           ue(4,i+1) )
c
               frct(5,i,j,k) = frct(5,i,j,k)
     $              + tx3 * c3 * c4 * ( flux(5,i+1) - flux(5,i) )
     $              + dx5 * tx1 * (            ue(5,i-1)
     $                             - 2.0d+00 * ue(5,i)
     $                             +           ue(5,i+1) )
c
            end do
c
c***Fourth-order dissipation
c
            do m = 1, 5
c
               frct(m,2,j,k) = frct(m,2,j,k)
     $           - dsspm * ( + 5.0d+00 * ue(m,2)
     $                       - 4.0d+00 * ue(m,3)
     $                       +           ue(m,4) )
c
               frct(m,3,j,k) = frct(m,3,j,k)
     $           - dsspm * ( - 4.0d+00 * ue(m,2)
     $                       + 6.0d+00 * ue(m,3)
     $                       - 4.0d+00 * ue(m,4)
     $                       +           ue(m,5) )
c
            end do
c
            do i = 4, nx - 3
c
               do m = 1, 5
c
                  frct(m,i,j,k) = frct(m,i,j,k)
     $              - dsspm * (            ue(m,i-2)
     $                         - 4.0d+00 * ue(m,i-1)
     $                         + 6.0d+00 * ue(m,i)
     $                         - 4.0d+00 * ue(m,i+1)
     $                         +           ue(m,i+2) )
c
               end do
c
            end do
c
            do m = 1, 5
c
               frct(m,nx-2,j,k) = frct(m,nx-2,j,k)
     $           - dsspm * (             ue(m,nx-4)
     $                       - 4.0d+00 * ue(m,nx-3)
     $                       + 6.0d+00 * ue(m,nx-2)
     $                       - 4.0d+00 * ue(m,nx-1)  )
c
               frct(m,nx-1,j,k) = frct(m,nx-1,j,k)
     $           - dsspm * (             ue(m,nx-3)
     $                       - 4.0d+00 * ue(m,nx-2)
     $                       + 5.0d+00 * ue(m,nx-1) )
c
            end do
c
         end do
c
      end do
c
c***eta-direction flux differences
c
      do k = 2, nz - 1
c
         zeta = ( dfloat(k-1) ) / ( nz - 1 )
c
         do i = 2, nx - 1
c
            xi = ( dfloat(i-1) ) / ( nx - 1 )
c 
            do j = 1, ny
c
               eta = ( dfloat(j-1) ) / ( ny - 1 )
c
               do m = 1, 5
c
                  ue(m,j) =  ce(m,1)
     $                 + ce(m,2) * xi
     $                 + ce(m,3) * eta
     $                 + ce(m,4) * zeta
     $                 + ce(m,5) * xi * xi
     $                 + ce(m,6) * eta * eta
     $                 + ce(m,7) * zeta * zeta
     $                 + ce(m,8) * xi * xi * xi
     $                 + ce(m,9) * eta * eta * eta
     $                 + ce(m,10) * zeta * zeta * zeta
     $                 + ce(m,11) * xi * xi * xi * xi
     $                 + ce(m,12) * eta * eta * eta * eta
     $                 + ce(m,13) * zeta * zeta * zeta * zeta
c
               end do
c
               flux(1,j) = ue(3,j)
c
               u31 = ue(3,j) / ue(1,j)
c
               q = 0.50d+00 * (  ue(2,j) * ue(2,j)
     $                         + ue(3,j) * ue(3,j)
     $                         + ue(4,j) * ue(4,j) )
     $                      / ue(1,j)
c
               flux(2,j) = ue(2,j) * u31 
c
               flux(3,j) = ue(3,j) * u31 + c2 * ( ue(5,j) - q )
c
               flux(4,j) = ue(4,j) * u31
c
               flux(5,j) = ( c1 * ue(5,j) - c2 * q ) * u31
c    
            end do
c
            do j = 2, ny - 1
c
               do m = 1, 5
c
                  frct(m,i,j,k) =  frct(m,i,j,k)
     $                       - ty2 * ( flux(m,j+1) - flux(m,j-1) )
c
               end do
c
            end do
c
            do j = 2, ny
c
               tmp = 1.0d+00 / ue(1,j)
c
               u21j = tmp * ue(2,j)
               u31j = tmp * ue(3,j)
               u41j = tmp * ue(4,j)
               u51j = tmp * ue(5,j)
c
               tmp = 1.0d+00 / ue(1,j-1)
c
               u21jm1 = tmp * ue(2,j-1)
               u31jm1 = tmp * ue(3,j-1)
               u41jm1 = tmp * ue(4,j-1)
               u51jm1 = tmp * ue(5,j-1)
c
               flux(2,j) = ty3 * ( u21j - u21jm1 )
               flux(3,j) = (4.0d+00/3.0d+00) * ty3 * ( u31j - u31jm1 )
               flux(4,j) = ty3 * ( u41j - u41jm1 )
               flux(5,j) = 0.50d+00 * ( 1.0d+00 - c1*c5 )
     $              * ty3 * ( ( u21j  **2 + u31j  **2 + u41j  **2 )
     $                      - ( u21jm1**2 + u31jm1**2 + u41jm1**2 ) )
     $              + (1.0d+00/6.0d+00)
     $              * ty3 * ( u31j**2 - u31jm1**2 )
     $              + c1 * c5 * ty3 * ( u51j - u51jm1 )
c
            end do
c
            do j = 2, ny - 1
c
               frct(1,i,j,k) = frct(1,i,j,k)
     $              + dy1 * ty1 * (            ue(1,j-1)
     $                             - 2.0d+00 * ue(1,j)
     $                             +           ue(1,j+1) )
c
               frct(2,i,j,k) = frct(2,i,j,k)
     $              + ty3 * c3 * c4 * ( flux(2,j+1) - flux(2,j) )
     $              + dy2 * ty1 * (            ue(2,j-1)
     $                             - 2.0d+00 * ue(2,j)
     $                             +           ue(2,j+1) )
c
               frct(3,i,j,k) = frct(3,i,j,k)
     $              + ty3 * c3 * c4 * ( flux(3,j+1) - flux(3,j) )
     $              + dy3 * ty1 * (            ue(3,j-1)
     $                             - 2.0d+00 * ue(3,j)
     $                             +           ue(3,j+1) )
c
               frct(4,i,j,k) = frct(4,i,j,k)
     $              + ty3 * c3 * c4 * ( flux(4,j+1) - flux(4,j) )
     $              + dy4 * ty1 * (            ue(4,j-1)
     $                             - 2.0d+00 * ue(4,j)
     $                             +           ue(4,j+1) )
c
               frct(5,i,j,k) = frct(5,i,j,k)
     $              + ty3 * c3 * c4 * ( flux(5,j+1) - flux(5,j) )
     $              + dy5 * ty1 * (            ue(5,j-1)
     $                             - 2.0d+00 * ue(5,j)
     $                             +           ue(5,j+1) )
c
            end do
c
c***fourth-order dissipation
c
            do m = 1, 5
c
               frct(m,i,2,k) = frct(m,i,2,k)
     $           - dsspm * ( + 5.0d+00 * ue(m,2)
     $                       - 4.0d+00 * ue(m,3)
     $                       +           ue(m,4) )
c
               frct(m,i,3,k) = frct(m,i,3,k)
     $           - dsspm * ( - 4.0d+00 * ue(m,2)
     $                       + 6.0d+00 * ue(m,3)
     $                       - 4.0d+00 * ue(m,4)
     $                       +           ue(m,5) )
c
            end do
c
            do j = 4, ny - 3
c
               do m = 1, 5
c
                  frct(m,i,j,k) = frct(m,i,j,k)
     $              - dsspm * (            ue(m,j-2)
     $                        - 4.0d+00 * ue(m,j-1)
     $                        + 6.0d+00 * ue(m,j)
     $                        - 4.0d+00 * ue(m,j+1)
     $                        +           ue(m,j+2) )
c
               end do
c
            end do
c
            do m = 1, 5
c
               frct(m,i,ny-2,k) = frct(m,i,ny-2,k)
     $           - dsspm * (             ue(m,ny-4)
     $                       - 4.0d+00 * ue(m,ny-3)
     $                       + 6.0d+00 * ue(m,ny-2)
     $                       - 4.0d+00 * ue(m,ny-1)  )
c
               frct(m,i,ny-1,k) = frct(m,i,ny-1,k)
     $           - dsspm * (             ue(m,ny-3)
     $                       - 4.0d+00 * ue(m,ny-2)
     $                       + 5.0d+00 * ue(m,ny-1)  )
c
            end do
c
         end do
c
      end do
c
c***zeta-direction flux differences
c
      do j = 2, ny - 1
c
         eta = ( dfloat(j-1) ) / ( ny - 1 )
c
         do i = 2, nx - 1
c
            xi = ( dfloat(i-1) ) / ( nx - 1 )
c
            do k = 1, nz
c
               zeta = ( dfloat(k-1) ) / ( nz - 1 )
c
               do m = 1, 5
c
                  ue(m,k) =  ce(m,1)
     $                 + ce(m,2) * xi
     $                 + ce(m,3) * eta
     $                 + ce(m,4) * zeta
     $                 + ce(m,5) * xi * xi
     $                 + ce(m,6) * eta * eta
     $                 + ce(m,7) * zeta * zeta
     $                 + ce(m,8) * xi * xi * xi
     $                 + ce(m,9) * eta * eta * eta
     $                 + ce(m,10) * zeta * zeta * zeta
     $                 + ce(m,11) * xi * xi * xi * xi
     $                 + ce(m,12) * eta * eta * eta * eta
     $                 + ce(m,13) * zeta * zeta * zeta * zeta
c
               end do
c
               flux(1,k) = ue(4,k)
c
               u41 = ue(4,k) / ue(1,k)
c
               q = 0.50d+00 * (  ue(2,k) * ue(2,k)
     $                         + ue(3,k) * ue(3,k)
     $                         + ue(4,k) * ue(4,k) )
     $                      / ue(1,k)
c
               flux(2,k) = ue(2,k) * u41 
c
               flux(3,k) = ue(3,k) * u41 
c
               flux(4,k) = ue(4,k) * u41 + c2 * ( ue(5,k) - q )
c
               flux(5,k) = ( c1 * ue(5,k) - c2 * q ) * u41
c    
            end do
c
            do k = 2, nz - 1
c
               do m = 1, 5
c
                  frct(m,i,j,k) =  frct(m,i,j,k)
     $                       - tz2 * ( flux(m,k+1) - flux(m,k-1) )
c
               end do
c
            end do
c
            do k = 2, nz
c
               tmp = 1.0d+00 / ue(1,k)
c
               u21k = tmp * ue(2,k)
               u31k = tmp * ue(3,k)
               u41k = tmp * ue(4,k)
               u51k = tmp * ue(5,k)
c
               tmp = 1.0d+00 / ue(1,k-1)
c
               u21km1 = tmp * ue(2,k-1)
               u31km1 = tmp * ue(3,k-1)
               u41km1 = tmp * ue(4,k-1)
               u51km1 = tmp * ue(5,k-1)
c
               flux(2,k) = tz3 * ( u21k - u21km1 )
               flux(3,k) = tz3 * ( u31k - u31km1 )
               flux(4,k) = (4.0d+00/3.0d+00) * tz3 * ( u41k - u41km1 )
               flux(5,k) = 0.50d+00 * ( 1.0d+00 - c1*c5 )
     $              * tz3 * ( ( u21k  **2 + u31k  **2 + u41k  **2 )
     $                      - ( u21km1**2 + u31km1**2 + u41km1**2 ) )
     $              + (1.0d+00/6.0d+00)
     $              * tz3 * ( u41k**2 - u41km1**2 )
     $              + c1 * c5 * tz3 * ( u51k - u51km1 )
c
            end do
c
            do k = 2, nz - 1
c
               frct(1,i,j,k) = frct(1,i,j,k)
     $              + dz1 * tz1 * (            ue(1,k+1)
     $                             - 2.0d+00 * ue(1,k)
     $                             +           ue(1,k-1) )
c
               frct(2,i,j,k) = frct(2,i,j,k)
     $              + tz3 * c3 * c4 * ( flux(2,k+1) - flux(2,k) )
     $              + dz2 * tz1 * (            ue(2,k+1)
     $                             - 2.0d+00 * ue(2,k)
     $                             +           ue(2,k-1) )
c
               frct(3,i,j,k) = frct(3,i,j,k)
     $              + tz3 * c3 * c4 * ( flux(3,k+1) - flux(3,k) )
     $              + dz3 * tz1 * (            ue(3,k+1)
     $                             - 2.0d+00 * ue(3,k)
     $                             +           ue(3,k-1) )
c
               frct(4,i,j,k) = frct(4,i,j,k)
     $              + tz3 * c3 * c4 * ( flux(4,k+1) - flux(4,k) )
     $              + dz4 * tz1 * (            ue(4,k+1)
     $                             - 2.0d+00 * ue(4,k)
     $                             +           ue(4,k-1) )
c
               frct(5,i,j,k) = frct(5,i,j,k)
     $              + tz3 * c3 * c4 * ( flux(5,k+1) - flux(5,k) )
     $              + dz5 * tz1 * (            ue(5,k+1)
     $                             - 2.0d+00 * ue(5,k)
     $                             +           ue(5,k-1) )
c
            end do
c
c***fourth-order dissipation
c
            do m = 1, 5
c
               frct(m,i,j,2) = frct(m,i,j,2)
     $           - dsspm * ( + 5.0d+00 * ue(m,2)
     $                       - 4.0d+00 * ue(m,3)
     $                       +           ue(m,4) )
c
               frct(m,i,j,3) = frct(m,i,j,3)
     $           - dsspm * (- 4.0d+00 * ue(m,2)
     $                      + 6.0d+00 * ue(m,3)
     $                      - 4.0d+00 * ue(m,4)
     $                      +           ue(m,5) )
c
            end do
c
            do k = 4, nz - 3
c
               do m = 1, 5
c
                  frct(m,i,j,k) = frct(m,i,j,k)
     $              - dsspm * (           ue(m,k-2)
     $                        - 4.0d+00 * ue(m,k-1)
     $                        + 6.0d+00 * ue(m,k)
     $                        - 4.0d+00 * ue(m,k+1)
     $                        +           ue(m,k+2) )
c
               end do
c
            end do
c
            do m = 1, 5
c
               frct(m,i,j,nz-2) = frct(m,i,j,nz-2)
     $           - dsspm * (            ue(m,nz-4)
     $                      - 4.0d+00 * ue(m,nz-3)
     $                      + 6.0d+00 * ue(m,nz-2)
     $                      - 4.0d+00 * ue(m,nz-1)  )
c
               frct(m,i,j,nz-1) = frct(m,i,j,nz-1)
     $           - dsspm * (             ue(m,nz-3)
     $                       - 4.0d+00 * ue(m,nz-2)
     $                       + 5.0d+00 * ue(m,nz-1)  )
c
            end do
c
         end do
c
      end do
      return
      end
c
c
c
c
c
      subroutine exact ( i, j, k, u000ijk )
c
c***compute the exact solution at (i,j,k)
c
c Author: Sisira Weeratunga 
c         NASA Ames Research Center
c         (10/25/90)
c
c #include 'applu.incl'  

      implicit real*8 (a-h,o-z)
c
      parameter ( c1 = 1.40d+00, c2 = 0.40d+00,
     $            c3 = 1.00d-01, c4 = 1.00d+00,
     $            c5 = 1.40d+00 )
c
c***grid
c
      common/cgcon/ nx, ny, nz,
     $              ii1, ii2, ji1, ji2, ki1, ki2, itwj,
     $              dxi, deta, dzeta,
     $              tx1, tx2, tx3,
     $              ty1, ty2, ty3,
     $              tz1, tz2, tz3
c
c***dissipation
c
      common/disp/ dx1,dx2,dx3,dx4,dx5,
     $             dy1,dy2,dy3,dy4,dy5,
     $             dz1,dz2,dz3,dz4,dz5,
     $             dssp
c
c***field variables and residuals
c
      common/cvar/ u(5,33,33,33),
     $             rsd(5,33,33,33),
     $             frct(5,33,33,33)
c
c***output control parameters
c
      common/cprcon/ ipr, iout, inorm
c
c***newton-raphson iteration control parameters
c
      common/ctscon/ itmax, invert,
     $               dt, omega, tolrsd(5),
     $               rsdnm(5), errnm(5), frc, ttotal
c
      common/cjac/ a(5,5,33,33,33),
     $             b(5,5,33,33,33),
     $             c(5,5,33,33,33),
     $             d(5,5,33,33,33)
c
c***coefficients of the exact solution
c
      common/cexact/ ce(5,13)
c



c
      dimension u000ijk(*)
c
      xi  = ( dfloat ( i - 1 ) ) / ( nx - 1 )
      eta  = ( dfloat ( j - 1 ) ) / ( ny - 1 )
      zeta = ( dfloat ( k - 1 ) ) / ( nz - 1 )
c
      do m = 1, 5
c
         u000ijk(m) =  ce(m,1)
     $        + ce(m,2) * xi
     $        + ce(m,3) * eta
     $        + ce(m,4) * zeta
     $        + ce(m,5) * xi * xi
     $        + ce(m,6) * eta * eta
     $        + ce(m,7) * zeta * zeta
     $        + ce(m,8) * xi * xi * xi
     $        + ce(m,9) * eta * eta * eta
     $        + ce(m,10) * zeta * zeta * zeta
     $        + ce(m,11) * xi * xi * xi * xi
     $        + ce(m,12) * eta * eta * eta * eta
     $        + ce(m,13) * zeta * zeta * zeta * zeta
c
      end do
c
      return
      end
c
c
c
c
c
      subroutine error
c
c***compute the solution error
c
c Author: Sisira Weeratunga 
c         NASA Ames Research Center
c         (10/25/90)
c
c #include 'applu.incl'  

      implicit real*8 (a-h,o-z)
c
      parameter ( c1 = 1.40d+00, c2 = 0.40d+00,
     $            c3 = 1.00d-01, c4 = 1.00d+00,
     $            c5 = 1.40d+00 )
c
c***grid
c
      common/cgcon/ nx, ny, nz,
     $              ii1, ii2, ji1, ji2, ki1, ki2, itwj,
     $              dxi, deta, dzeta,
     $              tx1, tx2, tx3,
     $              ty1, ty2, ty3,
     $              tz1, tz2, tz3
c
c***dissipation
c
      common/disp/ dx1,dx2,dx3,dx4,dx5,
     $             dy1,dy2,dy3,dy4,dy5,
     $             dz1,dz2,dz3,dz4,dz5,
     $             dssp
c
c***field variables and residuals
c
      common/cvar/ u(5,33,33,33),
     $             rsd(5,33,33,33),
     $             frct(5,33,33,33)
c
c***output control parameters
c
      common/cprcon/ ipr, iout, inorm
c
c***newton-raphson iteration control parameters
c
      common/ctscon/ itmax, invert,
     $               dt, omega, tolrsd(5),
     $               rsdnm(5), errnm(5), frc, ttotal
c
      common/cjac/ a(5,5,33,33,33),
     $             b(5,5,33,33,33),
     $             c(5,5,33,33,33),
     $             d(5,5,33,33,33)
c
c***coefficients of the exact solution
c
      common/cexact/ ce(5,13)
c



c
      dimension imax(5), jmax(5), kmax(5),
     $          u000ijk(5), errmax(5)
c
      lnorm = 2
c
      if ( lnorm .eq. 1 ) then
c
         do m = 1, 5
            errmax(m) = - 1.0d+20
         end do
c
         do k = 2, nz-1
            do j = 2, ny-1
               do i = 2, nx-1
c
                  call exact ( i, j, k, u000ijk )
c
                  do m = 1, 5
c
                     tmp = abs ( u000ijk(m) - u(m,i,j,k) )
c
                     if ( tmp .gt. errmax(m) ) then
c
                        errmax(m) = tmp
                        imax(m) = i
                        jmax(m) = j
                        kmax(m) = k
c
                     end if
c
                  end do
c
               end do
            end do
         end do
c
         write (6,1001) ( errmax(m),
     $        imax(m), jmax(m), kmax(m), m = 1, 5 )
c
      else if ( lnorm .eq. 2 ) then
c
         do m = 1, 5
            errnm(m) = 0.0d+00
         end do
c
         do k = 2, nz-1
            do j = 2, ny-1
               do i = 2, nx-1
c
                  call exact ( i, j, k, u000ijk )
c
                  do m = 1, 5
c
                     tmp = ( u000ijk(m) - u(m,i,j,k) )
c
                     errnm(m) = errnm(m) + tmp ** 2
c
                  end do
c
               end do
            end do
         end do
c
         do m = 1, 5
            errnm(m) = sqrt ( errnm(m) / ( (nx-2)*(ny-2)*(nz-2) ) )
         end do
c
         write (6,1002) ( errnm(m), m = 1, 5 )
c
      end if
c
 1001 format (/5x,'max. error in soln. to first pde  =',1pe12.4/,
     $5x,'and its location                  = (',i4','i4,','i4,' )'/,
     $/5x,'max. error in soln. to second pde =',1pe12.4/,
     $5x,'and its location                  = (',i4','i4,','i4,' )'/,
     $/5x,'max. error in soln. to third pde  =',1pe12.4/,
     $5x,'and its location                  = (',i4','i4,','i4,' )'/,
     $/5x,'max. error in soln. to fourth pde =',1pe12.4/,
     $5x,'and its location                  = (',i4','i4,','i4,' )'/,
     $/5x,'max. error in soln. to fifth pde  =',1pe12.4/,
     $5x,'and its location                  = (',i4','i4,','i4,' )' )
c
 1002 format (1x/1x,'RMS-norm of error in soln. to ',
     $ 'first pde  = ',1pe12.5/,
     $ 1x,'RMS-norm of error in soln. to ',
     $ 'second pde = ',1pe12.5/,
     $ 1x,'RMS-norm of error in soln. to ',
     $ 'third pde  = ',1pe12.5/,
     $ 1x,'RMS-norm of error in soln. to ',
     $ 'fourth pde = ',1pe12.5/,
     $ 1x,'RMS-norm of error in soln. to ',
     $ 'fifth pde  = ',1pe12.5)
c
      return
      end
c
c
c
c
c
      subroutine jacld
c
c***compute the lower triangular part of the jacobian matrix
c
c Author: Sisira Weeratunga
c         NASA Ames Research Center
c         (10/25/90)
c
c #include 'applu.incl'  

      implicit real*8 (a-h,o-z)
c
      parameter ( c1 = 1.40d+00, c2 = 0.40d+00,
     $            c3 = 1.00d-01, c4 = 1.00d+00,
     $            c5 = 1.40d+00 )
c
c***grid
c
      common/cgcon/ nx, ny, nz,
     $              ii1, ii2, ji1, ji2, ki1, ki2, itwj,
     $              dxi, deta, dzeta,
     $              tx1, tx2, tx3,
     $              ty1, ty2, ty3,
     $              tz1, tz2, tz3
c
c***dissipation
c
      common/disp/ dx1,dx2,dx3,dx4,dx5,
     $             dy1,dy2,dy3,dy4,dy5,
     $             dz1,dz2,dz3,dz4,dz5,
     $             dssp
c
c***field variables and residuals
c
      common/cvar/ u(5,33,33,33),
     $             rsd(5,33,33,33),
     $             frct(5,33,33,33)
c
c***output control parameters
c
      common/cprcon/ ipr, iout, inorm
c
c***newton-raphson iteration control parameters
c
      common/ctscon/ itmax, invert,
     $               dt, omega, tolrsd(5),
     $               rsdnm(5), errnm(5), frc, ttotal
c
      common/cjac/ a(5,5,33,33,33),
     $             b(5,5,33,33,33),
     $             c(5,5,33,33,33),
     $             d(5,5,33,33,33)
c
c***coefficients of the exact solution
c
      common/cexact/ ce(5,13)
c



c
      r43 = ( 4.0d+00 / 3.0d+00 )
      c1345 = c1 * c3 * c4 * c5
      c34 = c3 * c4
c      
      do k = 2, nz - 1
c
         do j = 2, ny - 1
c
            do i = 2, nx - 1
c
c***form the block daigonal
c
               tmp1 = 1.0d+00 / u(1,i,j,k)
               tmp2 = tmp1 * tmp1
               tmp3 = tmp1 * tmp2
c
               d(1,1,i,j,k) =  1.0d+00
     $                       + dt * 2.0d+00 * (   tx1 * dx1
     $                                          + ty1 * dy1
     $                                          + tz1 * dz1 )
               d(1,2,i,j,k) =  0.0d+00
               d(1,3,i,j,k) =  0.0d+00
               d(1,4,i,j,k) =  0.0d+00
               d(1,5,i,j,k) =  0.0d+00
c
               d(2,1,i,j,k) =  dt * 2.0d+00
     $          * (  tx1 * ( - r43 * c34 * tmp2 * u(2,i,j,k) )
     $             + ty1 * ( -       c34 * tmp2 * u(2,i,j,k) )
     $             + tz1 * ( -       c34 * tmp2 * u(2,i,j,k) ) )
               d(2,2,i,j,k) =  1.0d+00
     $          + dt * 2.0d+00 
     $          * (  tx1 * r43 * c34 * tmp1
     $             + ty1 *       c34 * tmp1
     $             + tz1 *       c34 * tmp1 )
     $          + dt * 2.0d+00 * (   tx1 * dx2
     $                             + ty1 * dy2
     $                             + tz1 * dz2  )
               d(2,3,i,j,k) = 0.0d+00
               d(2,4,i,j,k) = 0.0d+00
               d(2,5,i,j,k) = 0.0d+00
c
               d(3,1,i,j,k) = dt * 2.0d+00
     $      * (  tx1 * ( -       c34 * tmp2 * u(3,i,j,k) )
     $         + ty1 * ( - r43 * c34 * tmp2 * u(3,i,j,k) )
     $         + tz1 * ( -       c34 * tmp2 * u(3,i,j,k) ) )
               d(3,2,i,j,k) = 0.0d+00
               d(3,3,i,j,k) = 1.0d+00
     $         + dt * 2.0d+00
     $              * (  tx1 *       c34 * tmp1
     $                 + ty1 * r43 * c34 * tmp1
     $                 + tz1 *       c34 * tmp1 )
     $         + dt * 2.0d+00 * (  tx1 * dx3
     $                           + ty1 * dy3
     $                           + tz1 * dz3 )
               d(3,4,i,j,k) = 0.0d+00
               d(3,5,i,j,k) = 0.0d+00
c
               d(4,1,i,j,k) = dt * 2.0d+00
     $      * (  tx1 * ( -       c34 * tmp2 * u(4,i,j,k) )
     $         + ty1 * ( -       c34 * tmp2 * u(4,i,j,k) )
     $         + tz1 * ( - r43 * c34 * tmp2 * u(4,i,j,k) ) )
               d(4,2,i,j,k) = 0.0d+00
               d(4,3,i,j,k) = 0.0d+00
               d(4,4,i,j,k) = 1.0d+00
     $         + dt * 2.0d+00
     $              * (  tx1 *       c34 * tmp1
     $                 + ty1 *       c34 * tmp1
     $                 + tz1 * r43 * c34 * tmp1 )
     $         + dt * 2.0d+00 * (  tx1 * dx4
     $                           + ty1 * dy4
     $                           + tz1 * dz4 )
               d(4,5,i,j,k) = 0.0d+00
c
               d(5,1,i,j,k) = dt * 2.0d+00
     $ * ( tx1 * ( - ( r43*c34 - c1345 ) * tmp3 * ( u(2,i,j,k) ** 2 )
     $             - ( c34 - c1345 ) * tmp3 * ( u(3,i,j,k) ** 2 )
     $             - ( c34 - c1345 ) * tmp3 * ( u(4,i,j,k) ** 2 )
     $             - ( c1345 ) * tmp2 * u(5,i,j,k) )
     $   + ty1 * ( - ( c34 - c1345 ) * tmp3 * ( u(2,i,j,k) ** 2 )
     $             - ( r43*c34 - c1345 ) * tmp3 * ( u(3,i,j,k) ** 2 )
     $             - ( c34 - c1345 ) * tmp3 * ( u(4,i,j,k) ** 2 )
     $             - ( c1345 ) * tmp2 * u(5,i,j,k) )
     $   + tz1 * ( - ( c34 - c1345 ) * tmp3 * ( u(2,i,j,k) ** 2 )
     $             - ( c34 - c1345 ) * tmp3 * ( u(3,i,j,k) ** 2 )
     $             - ( r43*c34 - c1345 ) * tmp3 * ( u(4,i,j,k) ** 2 )
     $             - ( c1345 ) * tmp2 * u(5,i,j,k) ) )
               d(5,2,i,j,k) = dt * 2.0d+00
     $ * ( tx1 * ( r43*c34 - c1345 ) * tmp2 * u(2,i,j,k)
     $   + ty1 * (     c34 - c1345 ) * tmp2 * u(2,i,j,k)
     $   + tz1 * (     c34 - c1345 ) * tmp2 * u(2,i,j,k) )
               d(5,3,i,j,k) = dt * 2.0d+00
     $ * ( tx1 * ( c34 - c1345 ) * tmp2 * u(3,i,j,k)
     $   + ty1 * ( r43*c34 -c1345 ) * tmp2 * u(3,i,j,k)
     $   + tz1 * ( c34 - c1345 ) * tmp2 * u(3,i,j,k) )
               d(5,4,i,j,k) = dt * 2.0d+00
     $ * ( tx1 * ( c34 - c1345 ) * tmp2 * u(4,i,j,k)
     $   + ty1 * ( c34 - c1345 ) * tmp2 * u(4,i,j,k)
     $   + tz1 * ( r43*c34 - c1345 ) * tmp2 * u(4,i,j,k) )
               d(5,5,i,j,k) = 1.0d+00
     $   + dt * 2.0d+00 * ( tx1 * c1345 * tmp1
     $                    + ty1 * c1345 * tmp1
     $                    + tz1 * c1345 * tmp1 )
     $   + dt * 2.0d+00 * (  tx1 * dx5
     $                    +  ty1 * dy5
     $                    +  tz1 * dz5 )
c
c***form the first block sub-diagonal
c
               tmp1 = 1.0d+00 / u(1,i,j,k-1)
               tmp2 = tmp1 * tmp1
               tmp3 = tmp1 * tmp2
c
               a(1,1,i,j,k) = - dt * tz1 * dz1
               a(1,2,i,j,k) =   0.0d+00
               a(1,3,i,j,k) =   0.0d+00
               a(1,4,i,j,k) = - dt * tz2
               a(1,5,i,j,k) =   0.0d+00
c
               a(2,1,i,j,k) = - dt * tz2
     $           * ( - ( u(2,i,j,k-1)*u(4,i,j,k-1) ) * tmp2 )
     $           - dt * tz1 * ( - c34 * tmp2 * u(2,i,j,k-1) )
               a(2,2,i,j,k) = - dt * tz2 * ( u(4,i,j,k-1) * tmp1 )
     $           - dt * tz1 * c34 * tmp1
     $           - dt * tz1 * dz2 
               a(2,3,i,j,k) = 0.0d+00
               a(2,4,i,j,k) = - dt * tz2 * ( u(2,i,j,k-1) * tmp1 )
               a(2,5,i,j,k) = 0.0d+00
c
               a(3,1,i,j,k) = - dt * tz2
     $           * ( - ( u(3,i,j,k-1)*u(4,i,j,k-1) ) * tmp2 )
     $           - dt * tz1 * ( - c34 * tmp2 * u(3,i,j,k-1) )
               a(3,2,i,j,k) = 0.0d+00
               a(3,3,i,j,k) = - dt * tz2 * ( u(4,i,j,k-1) * tmp1 )
     $           - dt * tz1 * ( c34 * tmp1 )
     $           - dt * tz1 * dz3
               a(3,4,i,j,k) = - dt * tz2 * ( u(3,i,j,k-1) * tmp1 )
               a(3,5,i,j,k) = 0.0d+00
c
               a(4,1,i,j,k) = - dt * tz2
     $        * ( - ( u(4,i,j,k-1) * tmp1 ) ** 2
     $            + 0.50d+00 * c2
     $            * ( ( u(2,i,j,k-1) * u(2,i,j,k-1)
     $                + u(3,i,j,k-1) * u(3,i,j,k-1)
     $                + u(4,i,j,k-1) * u(4,i,j,k-1) ) * tmp2 ) )
     $        - dt * tz1 * ( - r43 * c34 * tmp2 * u(4,i,j,k-1) )
               a(4,2,i,j,k) = - dt * tz2
     $             * ( - c2 * ( u(2,i,j,k-1) * tmp1 ) )
               a(4,3,i,j,k) = - dt * tz2
     $             * ( - c2 * ( u(3,i,j,k-1) * tmp1 ) )
               a(4,4,i,j,k) = - dt * tz2 * ( 2.0d+00 - c2 )
     $             * ( u(4,i,j,k-1) * tmp1 )
     $             - dt * tz1 * ( r43 * c34 * tmp1 )
     $             - dt * tz1 * dz4
               a(4,5,i,j,k) = - dt * tz2 * c2
c
               a(5,1,i,j,k) = - dt * tz2
     $     * ( ( c2 * (  u(2,i,j,k-1) * u(2,i,j,k-1)
     $                 + u(3,i,j,k-1) * u(3,i,j,k-1)
     $                 + u(4,i,j,k-1) * u(4,i,j,k-1) ) * tmp2
     $       - c1 * ( u(5,i,j,k-1) * tmp1 ) )
     $            * ( u(4,i,j,k-1) * tmp1 ) )
     $       - dt * tz1
     $       * ( - ( c34 - c1345 ) * tmp3 * (u(2,i,j,k-1)**2)
     $           - ( c34 - c1345 ) * tmp3 * (u(3,i,j,k-1)**2)
     $           - ( r43*c34 - c1345 )* tmp3 * (u(4,i,j,k-1)**2)
     $          - c1345 * tmp2 * u(5,i,j,k-1) )
               a(5,2,i,j,k) = - dt * tz2
     $       * ( - c2 * ( u(2,i,j,k-1)*u(4,i,j,k-1) ) * tmp2 )
     $       - dt * tz1 * ( c34 - c1345 ) * tmp2 * u(2,i,j,k-1)
               a(5,3,i,j,k) = - dt * tz2
     $       * ( - c2 * ( u(3,i,j,k-1)*u(4,i,j,k-1) ) * tmp2 )
     $       - dt * tz1 * ( c34 - c1345 ) * tmp2 * u(3,i,j,k-1)
               a(5,4,i,j,k) = - dt * tz2
     $       * ( c1 * ( u(5,i,j,k-1) * tmp1 )
     $       - 0.50d+00 * c2
     $       * ( (  u(2,i,j,k-1)*u(2,i,j,k-1)
     $            + u(3,i,j,k-1)*u(3,i,j,k-1)
     $            + 3.0d+00*u(4,i,j,k-1)*u(4,i,j,k-1) ) * tmp2 ) )
     $       - dt * tz1 * ( r43*c34 - c1345 ) * tmp2 * u(4,i,j,k-1)
               a(5,5,i,j,k) = - dt * tz2
     $       * ( c1 * ( u(4,i,j,k-1) * tmp1 ) )
     $       - dt * tz1 * c1345 * tmp1
     $       - dt * tz1 * dz5
c
c***form the second block sub-diagonal
c
               tmp1 = 1.0d+00 / u(1,i,j-1,k)
               tmp2 = tmp1 * tmp1
               tmp3 = tmp1 * tmp2
c
               b(1,1,i,j,k) = - dt * ty1 * dy1
               b(1,2,i,j,k) =   0.0d+00
               b(1,3,i,j,k) = - dt * ty2
               b(1,4,i,j,k) =   0.0d+00
               b(1,5,i,j,k) =   0.0d+00
c
               b(2,1,i,j,k) = - dt * ty2
     $           * ( - ( u(2,i,j-1,k)*u(3,i,j-1,k) ) * tmp2 )
     $           - dt * ty1 * ( - c34 * tmp2 * u(2,i,j-1,k) )
               b(2,2,i,j,k) = - dt * ty2 * ( u(3,i,j-1,k) * tmp1 )
     $          - dt * ty1 * ( c34 * tmp1 )
     $          - dt * ty1 * dy2
               b(2,3,i,j,k) = - dt * ty2 * ( u(2,i,j-1,k) * tmp1 )
               b(2,4,i,j,k) = 0.0d+00
               b(2,5,i,j,k) = 0.0d+00
c
               b(3,1,i,j,k) = - dt * ty2
     $           * ( - ( u(3,i,j-1,k) * tmp1 ) ** 2
     $      + 0.50d+00 * c2 * ( (  u(2,i,j-1,k) * u(2,i,j-1,k)
     $                           + u(3,i,j-1,k) * u(3,i,j-1,k)
     $                           + u(4,i,j-1,k) * u(4,i,j-1,k) )
     $                          * tmp2 ) )
     $       - dt * ty1 * ( - r43 * c34 * tmp2 * u(3,i,j-1,k) )
               b(3,2,i,j,k) = - dt * ty2
     $                   * ( - c2 * ( u(2,i,j-1,k) * tmp1 ) )
               b(3,3,i,j,k) = - dt * ty2 * ( ( 2.0d+00 - c2 )
     $                   * ( u(3,i,j-1,k) * tmp1 ) )
     $       - dt * ty1 * ( r43 * c34 * tmp1 )
     $       - dt * ty1 * dy3
               b(3,4,i,j,k) = - dt * ty2
     $                   * ( - c2 * ( u(4,i,j-1,k) * tmp1 ) )
               b(3,5,i,j,k) = - dt * ty2 * c2
c
               b(4,1,i,j,k) = - dt * ty2
     $              * ( - ( u(3,i,j-1,k)*u(4,i,j-1,k) ) * tmp2 )
     $       - dt * ty1 * ( - c34 * tmp2 * u(4,i,j-1,k) )
               b(4,2,i,j,k) = 0.0d+00
               b(4,3,i,j,k) = - dt * ty2 * ( u(4,i,j-1,k) * tmp1 )
               b(4,4,i,j,k) = - dt * ty2 * ( u(3,i,j-1,k) * tmp1 )
     $                        - dt * ty1 * ( c34 * tmp1 )
     $                        - dt * ty1 * dy4
               b(4,5,i,j,k) = 0.0d+00
c
               b(5,1,i,j,k) = - dt * ty2
     $          * ( ( c2 * (  u(2,i,j-1,k) * u(2,i,j-1,k)
     $                      + u(3,i,j-1,k) * u(3,i,j-1,k)
     $                      + u(4,i,j-1,k) * u(4,i,j-1,k) ) * tmp2
     $               - c1 * ( u(5,i,j-1,k) * tmp1 ) )
     $          * ( u(3,i,j-1,k) * tmp1 ) )
     $          - dt * ty1
     $          * ( - (     c34 - c1345 )*tmp3*(u(2,i,j-1,k)**2)
     $              - ( r43*c34 - c1345 )*tmp3*(u(3,i,j-1,k)**2)
     $              - (     c34 - c1345 )*tmp3*(u(4,i,j-1,k)**2)
     $              - c1345*tmp2*u(5,i,j-1,k) )
               b(5,2,i,j,k) = - dt * ty2
     $          * ( - c2 * ( u(2,i,j-1,k)*u(3,i,j-1,k) ) * tmp2 )
     $          - dt * ty1
     $          * ( c34 - c1345 ) * tmp2 * u(2,i,j-1,k)
               b(5,3,i,j,k) = - dt * ty2
     $          * ( c1 * ( u(5,i,j-1,k) * tmp1 )
     $          - 0.50d+00 * c2 
     $          * ( (  u(2,i,j-1,k)*u(2,i,j-1,k)
     $               + 3.0d+00 * u(3,i,j-1,k)*u(3,i,j-1,k)
     $               + u(4,i,j-1,k)*u(4,i,j-1,k) ) * tmp2 ) )
     $          - dt * ty1
     $          * ( r43*c34 - c1345 ) * tmp2 * u(3,i,j-1,k)
               b(5,4,i,j,k) = - dt * ty2
     $          * ( - c2 * ( u(3,i,j-1,k)*u(4,i,j-1,k) ) * tmp2 )
     $          - dt * ty1 * ( c34 - c1345 ) * tmp2 * u(4,i,j-1,k)
               b(5,5,i,j,k) = - dt * ty2
     $          * ( c1 * ( u(3,i,j-1,k) * tmp1 ) )
     $          - dt * ty1 * c1345 * tmp1
     $          - dt * ty1 * dy5
c               
c***form the third block sub-diagonal
c
               tmp1 = 1.0d+00 / u(1,i-1,j,k)
               tmp2 = tmp1 * tmp1
               tmp3 = tmp1 * tmp2
c
               c(1,1,i,j,k) = - dt * tx1 * dx1
               c(1,2,i,j,k) = - dt * tx2
               c(1,3,i,j,k) =   0.0d+00
               c(1,4,i,j,k) =   0.0d+00
               c(1,5,i,j,k) =   0.0d+00
c
               c(2,1,i,j,k) = - dt * tx2
     $          * ( - ( u(2,i-1,j,k) * tmp1 ) ** 2
     $     + c2 * 0.50d+00 * (  u(2,i-1,j,k) * u(2,i-1,j,k)
     $                        + u(3,i-1,j,k) * u(3,i-1,j,k)
     $                        + u(4,i-1,j,k) * u(4,i-1,j,k) ) * tmp2 )
     $          - dt * tx1 * ( - r43 * c34 * tmp2 * u(2,i-1,j,k) )
               c(2,2,i,j,k) = - dt * tx2
     $          * ( ( 2.0d+00 - c2 ) * ( u(2,i-1,j,k) * tmp1 ) )
     $          - dt * tx1 * ( r43 * c34 * tmp1 )
     $          - dt * tx1 * dx2
               c(2,3,i,j,k) = - dt * tx2
     $              * ( - c2 * ( u(3,i-1,j,k) * tmp1 ) )
               c(2,4,i,j,k) = - dt * tx2
     $              * ( - c2 * ( u(4,i-1,j,k) * tmp1 ) )
               c(2,5,i,j,k) = - dt * tx2 * c2 
c
               c(3,1,i,j,k) = - dt * tx2
     $              * ( - ( u(2,i-1,j,k) * u(3,i-1,j,k) ) * tmp2 )
     $         - dt * tx1 * ( - c34 * tmp2 * u(3,i-1,j,k) )
               c(3,2,i,j,k) = - dt * tx2 * ( u(3,i-1,j,k) * tmp1 )
               c(3,3,i,j,k) = - dt * tx2 * ( u(2,i-1,j,k) * tmp1 )
     $          - dt * tx1 * ( c34 * tmp1 )
     $          - dt * tx1 * dx3
               c(3,4,i,j,k) = 0.0d+00
               c(3,5,i,j,k) = 0.0d+00
c
               c(4,1,i,j,k) = - dt * tx2
     $          * ( - ( u(2,i-1,j,k)*u(4,i-1,j,k) ) * tmp2 )
     $          - dt * tx1 * ( - c34 * tmp2 * u(4,i-1,j,k) )
               c(4,2,i,j,k) = - dt * tx2 * ( u(4,i-1,j,k) * tmp1 )
               c(4,3,i,j,k) = 0.0d+00
               c(4,4,i,j,k) = - dt * tx2 * ( u(2,i-1,j,k) * tmp1 )
     $          - dt * tx1 * ( c34 * tmp1 )
     $          - dt * tx1 * dx4
               c(4,5,i,j,k) = 0.0d+00
c
               c(5,1,i,j,k) = - dt * tx2
     $          * ( ( c2 * (  u(2,i-1,j,k) * u(2,i-1,j,k)
     $                      + u(3,i-1,j,k) * u(3,i-1,j,k)
     $                      + u(4,i-1,j,k) * u(4,i-1,j,k) ) * tmp2
     $              - c1 * ( u(5,i-1,j,k) * tmp1 ) )
     $          * ( u(2,i-1,j,k) * tmp1 ) )
     $          - dt * tx1
     $          * ( - ( r43*c34 - c1345 ) * tmp3 * ( u(2,i-1,j,k)**2 )
     $              - (     c34 - c1345 ) * tmp3 * ( u(3,i-1,j,k)**2 )
     $              - (     c34 - c1345 ) * tmp3 * ( u(4,i-1,j,k)**2 )
     $              - c1345 * tmp2 * u(5,i-1,j,k) )
               c(5,2,i,j,k) = - dt * tx2
     $          * ( c1 * ( u(5,i-1,j,k) * tmp1 )
     $             - 0.50d+00 * c2
     $             * ( (  3.0d+00*u(2,i-1,j,k)*u(2,i-1,j,k)
     $                  + u(3,i-1,j,k)*u(3,i-1,j,k)
     $                  + u(4,i-1,j,k)*u(4,i-1,j,k) ) * tmp2 ) )
     $           - dt * tx1
     $           * ( r43*c34 - c1345 ) * tmp2 * u(2,i-1,j,k)
               c(5,3,i,j,k) = - dt * tx2
     $           * ( - c2 * ( u(3,i-1,j,k)*u(2,i-1,j,k) ) * tmp2 )
     $           - dt * tx1
     $           * (  c34 - c1345 ) * tmp2 * u(3,i-1,j,k)
               c(5,4,i,j,k) = - dt * tx2
     $           * ( - c2 * ( u(4,i-1,j,k)*u(2,i-1,j,k) ) * tmp2 )
     $           - dt * tx1
     $           * (  c34 - c1345 ) * tmp2 * u(4,i-1,j,k)
               c(5,5,i,j,k) = - dt * tx2
     $           * ( c1 * ( u(2,i-1,j,k) * tmp1 ) )
     $           - dt * tx1 * c1345 * tmp1
     $           - dt * tx1 * dx5
c
            end do
c
         end do
c
      end do               
c
      return
      end
c
c
c
c
c
      subroutine jacu
c
c***compute the upper triangular part of the jacobian matrix
c
c Author: Sisira Weeratunga
c         NASA Ames Research Center
c         (10/25/90)
c
c #include 'applu.incl'  

      implicit real*8 (a-h,o-z)
c
      parameter ( c1 = 1.40d+00, c2 = 0.40d+00,
     $            c3 = 1.00d-01, c4 = 1.00d+00,
     $            c5 = 1.40d+00 )
c
c***grid
c
      common/cgcon/ nx, ny, nz,
     $              ii1, ii2, ji1, ji2, ki1, ki2, itwj,
     $              dxi, deta, dzeta,
     $              tx1, tx2, tx3,
     $              ty1, ty2, ty3,
     $              tz1, tz2, tz3
c
c***dissipation
c
      common/disp/ dx1,dx2,dx3,dx4,dx5,
     $             dy1,dy2,dy3,dy4,dy5,
     $             dz1,dz2,dz3,dz4,dz5,
     $             dssp
c
c***field variables and residuals
c
      common/cvar/ u(5,33,33,33),
     $             rsd(5,33,33,33),
     $             frct(5,33,33,33)
c
c***output control parameters
c
      common/cprcon/ ipr, iout, inorm
c
c***newton-raphson iteration control parameters
c
      common/ctscon/ itmax, invert,
     $               dt, omega, tolrsd(5),
     $               rsdnm(5), errnm(5), frc, ttotal
c
      common/cjac/ a(5,5,33,33,33),
     $             b(5,5,33,33,33),
     $             c(5,5,33,33,33),
     $             d(5,5,33,33,33)
c
c***coefficients of the exact solution
c
      common/cexact/ ce(5,13)
c



c
      r43 = ( 4.0d+00 / 3.0d+00 )
      c1345 = c1 * c3 * c4 * c5
      c34 = c3 * c4
c      
      do k = 2, nz - 1
c
         do j = 2, ny - 1
c
            do i = 2, nx - 1
c               
c***form the first block sub-diagonal
c
               tmp1 = 1.0d+00 / u(1,i+1,j,k)
               tmp2 = tmp1 * tmp1
               tmp3 = tmp1 * tmp2
c
               a(1,1,i,j,k) = - dt * tx1 * dx1
               a(1,2,i,j,k) =   dt * tx2
               a(1,3,i,j,k) =   0.0d+00
               a(1,4,i,j,k) =   0.0d+00
               a(1,5,i,j,k) =   0.0d+00
c
               a(2,1,i,j,k) =  dt * tx2
     $          * ( - ( u(2,i+1,j,k) * tmp1 ) ** 2
     $     + c2 * 0.50d+00 * (  u(2,i+1,j,k) * u(2,i+1,j,k)
     $                        + u(3,i+1,j,k) * u(3,i+1,j,k)
     $                        + u(4,i+1,j,k) * u(4,i+1,j,k) ) * tmp2 )
     $          - dt * tx1 * ( - r43 * c34 * tmp2 * u(2,i+1,j,k) )
               a(2,2,i,j,k) =  dt * tx2
     $          * ( ( 2.0d+00 - c2 ) * ( u(2,i+1,j,k) * tmp1 ) )
     $          - dt * tx1 * ( r43 * c34 * tmp1 )
     $          - dt * tx1 * dx2
               a(2,3,i,j,k) =  dt * tx2
     $              * ( - c2 * ( u(3,i+1,j,k) * tmp1 ) )
               a(2,4,i,j,k) =  dt * tx2
     $              * ( - c2 * ( u(4,i+1,j,k) * tmp1 ) )
               a(2,5,i,j,k) =  dt * tx2 * c2 
c
               a(3,1,i,j,k) =  dt * tx2
     $              * ( - ( u(2,i+1,j,k) * u(3,i+1,j,k) ) * tmp2 )
     $         - dt * tx1 * ( - c34 * tmp2 * u(3,i+1,j,k) )
               a(3,2,i,j,k) =  dt * tx2 * ( u(3,i+1,j,k) * tmp1 )
               a(3,3,i,j,k) =  dt * tx2 * ( u(2,i+1,j,k) * tmp1 )
     $          - dt * tx1 * ( c34 * tmp1 )
     $          - dt * tx1 * dx3
               a(3,4,i,j,k) = 0.0d+00
               a(3,5,i,j,k) = 0.0d+00
c
               a(4,1,i,j,k) = dt * tx2
     $          * ( - ( u(2,i+1,j,k)*u(4,i+1,j,k) ) * tmp2 )
     $          - dt * tx1 * ( - c34 * tmp2 * u(4,i+1,j,k) )
               a(4,2,i,j,k) = dt * tx2 * ( u(4,i+1,j,k) * tmp1 )
               a(4,3,i,j,k) = 0.0d+00
               a(4,4,i,j,k) = dt * tx2 * ( u(2,i+1,j,k) * tmp1 )
     $          - dt * tx1 * ( c34 * tmp1 )
     $          - dt * tx1 * dx4
               a(4,5,i,j,k) = 0.0d+00
c
               a(5,1,i,j,k) = dt * tx2
     $          * ( ( c2 * (  u(2,i+1,j,k) * u(2,i+1,j,k)
     $                      + u(3,i+1,j,k) * u(3,i+1,j,k)
     $                      + u(4,i+1,j,k) * u(4,i+1,j,k) ) * tmp2
     $              - c1 * ( u(5,i+1,j,k) * tmp1 ) )
     $          * ( u(2,i+1,j,k) * tmp1 ) )
     $          - dt * tx1
     $          * ( - ( r43*c34 - c1345 ) * tmp3 * ( u(2,i+1,j,k)**2 )
     $              - (     c34 - c1345 ) * tmp3 * ( u(3,i+1,j,k)**2 )
     $              - (     c34 - c1345 ) * tmp3 * ( u(4,i+1,j,k)**2 )
     $              - c1345 * tmp2 * u(5,i+1,j,k) )
               a(5,2,i,j,k) = dt * tx2
     $          * ( c1 * ( u(5,i+1,j,k) * tmp1 )
     $             - 0.50d+00 * c2
     $             * ( (  3.0d+00*u(2,i+1,j,k)*u(2,i+1,j,k)
     $                  + u(3,i+1,j,k)*u(3,i+1,j,k)
     $                  + u(4,i+1,j,k)*u(4,i+1,j,k) ) * tmp2 ) )
     $           - dt * tx1
     $           * ( r43*c34 - c1345 ) * tmp2 * u(2,i+1,j,k)
               a(5,3,i,j,k) = dt * tx2
     $           * ( - c2 * ( u(3,i+1,j,k)*u(2,i+1,j,k) ) * tmp2 )
     $           - dt * tx1
     $           * (  c34 - c1345 ) * tmp2 * u(3,i+1,j,k)
               a(5,4,i,j,k) = dt * tx2
     $           * ( - c2 * ( u(4,i+1,j,k)*u(2,i+1,j,k) ) * tmp2 )
     $           - dt * tx1
     $           * (  c34 - c1345 ) * tmp2 * u(4,i+1,j,k)
               a(5,5,i,j,k) = dt * tx2
     $           * ( c1 * ( u(2,i+1,j,k) * tmp1 ) )
     $           - dt * tx1 * c1345 * tmp1
     $           - dt * tx1 * dx5
c
c***form the second block sub-diagonal
c
               tmp1 = 1.0d+00 / u(1,i,j+1,k)
               tmp2 = tmp1 * tmp1
               tmp3 = tmp1 * tmp2
c
               b(1,1,i,j,k) = - dt * ty1 * dy1
               b(1,2,i,j,k) =   0.0d+00
               b(1,3,i,j,k) =  dt * ty2
               b(1,4,i,j,k) =   0.0d+00
               b(1,5,i,j,k) =   0.0d+00
c
               b(2,1,i,j,k) =  dt * ty2
     $           * ( - ( u(2,i,j+1,k)*u(3,i,j+1,k) ) * tmp2 )
     $           - dt * ty1 * ( - c34 * tmp2 * u(2,i,j+1,k) )
               b(2,2,i,j,k) =  dt * ty2 * ( u(3,i,j+1,k) * tmp1 )
     $          - dt * ty1 * ( c34 * tmp1 )
     $          - dt * ty1 * dy2
               b(2,3,i,j,k) =  dt * ty2 * ( u(2,i,j+1,k) * tmp1 )
               b(2,4,i,j,k) = 0.0d+00
               b(2,5,i,j,k) = 0.0d+00
c
               b(3,1,i,j,k) =  dt * ty2
     $           * ( - ( u(3,i,j+1,k) * tmp1 ) ** 2
     $      + 0.50d+00 * c2 * ( (  u(2,i,j+1,k) * u(2,i,j+1,k)
     $                           + u(3,i,j+1,k) * u(3,i,j+1,k)
     $                           + u(4,i,j+1,k) * u(4,i,j+1,k) )
     $                          * tmp2 ) )
     $       - dt * ty1 * ( - r43 * c34 * tmp2 * u(3,i,j+1,k) )
               b(3,2,i,j,k) =  dt * ty2
     $                   * ( - c2 * ( u(2,i,j+1,k) * tmp1 ) )
               b(3,3,i,j,k) =  dt * ty2 * ( ( 2.0d+00 - c2 )
     $                   * ( u(3,i,j+1,k) * tmp1 ) )
     $       - dt * ty1 * ( r43 * c34 * tmp1 )
     $       - dt * ty1 * dy3
               b(3,4,i,j,k) =  dt * ty2
     $                   * ( - c2 * ( u(4,i,j+1,k) * tmp1 ) )
               b(3,5,i,j,k) =  dt * ty2 * c2
c
               b(4,1,i,j,k) =  dt * ty2
     $              * ( - ( u(3,i,j+1,k)*u(4,i,j+1,k) ) * tmp2 )
     $       - dt * ty1 * ( - c34 * tmp2 * u(4,i,j+1,k) )
               b(4,2,i,j,k) = 0.0d+00
               b(4,3,i,j,k) =  dt * ty2 * ( u(4,i,j+1,k) * tmp1 )
               b(4,4,i,j,k) =  dt * ty2 * ( u(3,i,j+1,k) * tmp1 )
     $                        - dt * ty1 * ( c34 * tmp1 )
     $                        - dt * ty1 * dy4
               b(4,5,i,j,k) = 0.0d+00
c
               b(5,1,i,j,k) =  dt * ty2
     $          * ( ( c2 * (  u(2,i,j+1,k) * u(2,i,j+1,k)
     $                      + u(3,i,j+1,k) * u(3,i,j+1,k)
     $                      + u(4,i,j+1,k) * u(4,i,j+1,k) ) * tmp2
     $               - c1 * ( u(5,i,j+1,k) * tmp1 ) )
     $          * ( u(3,i,j+1,k) * tmp1 ) )
     $          - dt * ty1
     $          * ( - (     c34 - c1345 )*tmp3*(u(2,i,j+1,k)**2)
     $              - ( r43*c34 - c1345 )*tmp3*(u(3,i,j+1,k)**2)
     $              - (     c34 - c1345 )*tmp3*(u(4,i,j+1,k)**2)
     $              - c1345*tmp2*u(5,i,j+1,k) )
               b(5,2,i,j,k) =  dt * ty2
     $          * ( - c2 * ( u(2,i,j+1,k)*u(3,i,j+1,k) ) * tmp2 )
     $          - dt * ty1
     $          * ( c34 - c1345 ) * tmp2 * u(2,i,j+1,k)
               b(5,3,i,j,k) =  dt * ty2
     $          * ( c1 * ( u(5,i,j+1,k) * tmp1 )
     $          - 0.50d+00 * c2 
     $          * ( (  u(2,i,j+1,k)*u(2,i,j+1,k)
     $               + 3.0d+00 * u(3,i,j+1,k)*u(3,i,j+1,k)
     $               + u(4,i,j+1,k)*u(4,i,j+1,k) ) * tmp2 ) )
     $          - dt * ty1
     $          * ( r43*c34 - c1345 ) * tmp2 * u(3,i,j+1,k)
               b(5,4,i,j,k) =  dt * ty2
     $          * ( - c2 * ( u(3,i,j+1,k)*u(4,i,j+1,k) ) * tmp2 )
     $          - dt * ty1 * ( c34 - c1345 ) * tmp2 * u(4,i,j+1,k)
               b(5,5,i,j,k) =  dt * ty2
     $          * ( c1 * ( u(3,i,j+1,k) * tmp1 ) )
     $          - dt * ty1 * c1345 * tmp1
     $          - dt * ty1 * dy5
c
c***form the third block sub-diagonal
c
               tmp1 = 1.0d+00 / u(1,i,j,k+1)
               tmp2 = tmp1 * tmp1
               tmp3 = tmp1 * tmp2
c
               c(1,1,i,j,k) = - dt * tz1 * dz1
               c(1,2,i,j,k) =   0.0d+00
               c(1,3,i,j,k) =   0.0d+00
               c(1,4,i,j,k) = dt * tz2
               c(1,5,i,j,k) =   0.0d+00
c
               c(2,1,i,j,k) = dt * tz2
     $           * ( - ( u(2,i,j,k+1)*u(4,i,j,k+1) ) * tmp2 )
     $           - dt * tz1 * ( - c34 * tmp2 * u(2,i,j,k+1) )
               c(2,2,i,j,k) = dt * tz2 * ( u(4,i,j,k+1) * tmp1 )
     $           - dt * tz1 * c34 * tmp1
     $           - dt * tz1 * dz2 
               c(2,3,i,j,k) = 0.0d+00
               c(2,4,i,j,k) = dt * tz2 * ( u(2,i,j,k+1) * tmp1 )
               c(2,5,i,j,k) = 0.0d+00
c
               c(3,1,i,j,k) = dt * tz2
     $           * ( - ( u(3,i,j,k+1)*u(4,i,j,k+1) ) * tmp2 )
     $           - dt * tz1 * ( - c34 * tmp2 * u(3,i,j,k+1) )
               c(3,2,i,j,k) = 0.0d+00
               c(3,3,i,j,k) = dt * tz2 * ( u(4,i,j,k+1) * tmp1 )
     $           - dt * tz1 * ( c34 * tmp1 )
     $           - dt * tz1 * dz3
               c(3,4,i,j,k) = dt * tz2 * ( u(3,i,j,k+1) * tmp1 )
               c(3,5,i,j,k) = 0.0d+00
c
               c(4,1,i,j,k) = dt * tz2
     $        * ( - ( u(4,i,j,k+1) * tmp1 ) ** 2
     $            + 0.50d+00 * c2
     $            * ( ( u(2,i,j,k+1) * u(2,i,j,k+1)
     $                + u(3,i,j,k+1) * u(3,i,j,k+1)
     $                + u(4,i,j,k+1) * u(4,i,j,k+1) ) * tmp2 ) )
     $        - dt * tz1 * ( - r43 * c34 * tmp2 * u(4,i,j,k+1) )
               c(4,2,i,j,k) = dt * tz2
     $             * ( - c2 * ( u(2,i,j,k+1) * tmp1 ) )
               c(4,3,i,j,k) = dt * tz2
     $             * ( - c2 * ( u(3,i,j,k+1) * tmp1 ) )
               c(4,4,i,j,k) = dt * tz2 * ( 2.0d+00 - c2 )
     $             * ( u(4,i,j,k+1) * tmp1 )
     $             - dt * tz1 * ( r43 * c34 * tmp1 )
     $             - dt * tz1 * dz4
               c(4,5,i,j,k) = dt * tz2 * c2
c
               c(5,1,i,j,k) = dt * tz2
     $     * ( ( c2 * (  u(2,i,j,k+1) * u(2,i,j,k+1)
     $                 + u(3,i,j,k+1) * u(3,i,j,k+1)
     $                 + u(4,i,j,k+1) * u(4,i,j,k+1) ) * tmp2
     $       - c1 * ( u(5,i,j,k+1) * tmp1 ) )
     $            * ( u(4,i,j,k+1) * tmp1 ) )
     $       - dt * tz1
     $       * ( - ( c34 - c1345 ) * tmp3 * (u(2,i,j,k+1)**2)
     $           - ( c34 - c1345 ) * tmp3 * (u(3,i,j,k+1)**2)
     $           - ( r43*c34 - c1345 )* tmp3 * (u(4,i,j,k+1)**2)
     $          - c1345 * tmp2 * u(5,i,j,k+1) )
               c(5,2,i,j,k) = dt * tz2
     $       * ( - c2 * ( u(2,i,j,k+1)*u(4,i,j,k+1) ) * tmp2 )
     $       - dt * tz1 * ( c34 - c1345 ) * tmp2 * u(2,i,j,k+1)
               c(5,3,i,j,k) = dt * tz2
     $       * ( - c2 * ( u(3,i,j,k+1)*u(4,i,j,k+1) ) * tmp2 )
     $       - dt * tz1 * ( c34 - c1345 ) * tmp2 * u(3,i,j,k+1)
               c(5,4,i,j,k) = dt * tz2
     $       * ( c1 * ( u(5,i,j,k+1) * tmp1 )
     $       - 0.50d+00 * c2
     $       * ( (  u(2,i,j,k+1)*u(2,i,j,k+1)
     $            + u(3,i,j,k+1)*u(3,i,j,k+1)
     $            + 3.0d+00*u(4,i,j,k+1)*u(4,i,j,k+1) ) * tmp2 ) )
     $       - dt * tz1 * ( r43*c34 - c1345 ) * tmp2 * u(4,i,j,k+1)
               c(5,5,i,j,k) = dt * tz2
     $       * ( c1 * ( u(4,i,j,k+1) * tmp1 ) )
     $       - dt * tz1 * c1345 * tmp1
     $       - dt * tz1 * dz5
c
            end do
c
         end do
c
      end do               
c
      return
      end
c
c
c
c
c
      subroutine pintgr
c
c***compute the surface integral
c
c Author: Sisira Weeratunga
c         NASA Ames Research Center
c         (10/25/90)
c
c #include 'applu.incl'  

      implicit real*8 (a-h,o-z)
c
      parameter ( c1 = 1.40d+00, c2 = 0.40d+00,
     $            c3 = 1.00d-01, c4 = 1.00d+00,
     $            c5 = 1.40d+00 )
c
c***grid
c
      common/cgcon/ nx, ny, nz,
     $              ii1, ii2, ji1, ji2, ki1, ki2, itwj,
     $              dxi, deta, dzeta,
     $              tx1, tx2, tx3,
     $              ty1, ty2, ty3,
     $              tz1, tz2, tz3
c
c***dissipation
c
      common/disp/ dx1,dx2,dx3,dx4,dx5,
     $             dy1,dy2,dy3,dy4,dy5,
     $             dz1,dz2,dz3,dz4,dz5,
     $             dssp
c
c***field variables and residuals
c
      common/cvar/ u(5,33,33,33),
     $             rsd(5,33,33,33),
     $             frct(5,33,33,33)
c
c***output control parameters
c
      common/cprcon/ ipr, iout, inorm
c
c***newton-raphson iteration control parameters
c
      common/ctscon/ itmax, invert,
     $               dt, omega, tolrsd(5),
     $               rsdnm(5), errnm(5), frc, ttotal
c
      common/cjac/ a(5,5,33,33,33),
     $             b(5,5,33,33,33),
     $             c(5,5,33,33,33),
     $             d(5,5,33,33,33)
c
c***coefficients of the exact solution
c
      common/cexact/ ce(5,13)
c



c
      dimension phi1(33,33), phi2(33,33)
c
      do j = ji1, ji2
c     
         do i = ii1, ii2
c
            phi1(i,j) = c2*(  u(5,i,j,ki1)
     $           - 0.50d+00 * (  u(2,i,j,ki1) ** 2
     $                         + u(3,i,j,ki1) ** 2
     $                         + u(4,i,j,ki1) ** 2 )
     $                        / u(1,i,j,ki1) )
c     
            phi2(i,j) = c2*(  u(5,i,j,ki2)
     $           - 0.50d+00 * (  u(2,i,j,ki2) ** 2
     $                         + u(3,i,j,ki2) ** 2
     $                         + u(4,i,j,ki2) ** 2 )
     $                        / u(1,i,j,ki2) )
c
         end do
c
      end do
c
      frc1 = 0.0d+00
c
      do j = ji1, ji2-1
c
         do i = ii1, ii2-1
c
            frc1 = frc1 + (  phi1(i,j)
     $                     + phi1(i+1,j)
     $                     + phi1(i,j+1)
     $                     + phi1(i+1,j+1)
     $                     + phi2(i,j)
     $                     + phi2(i+1,j)
     $                     + phi2(i,j+1)
     $                     + phi2(i+1,j+1) )
c
         end do
c
      end do
c
      frc1 = dxi * deta * frc1
c
      do k = ki1, ki2
c     
         do i = ii1, ii2
c
            phi1(i,k) = c2*(  u(5,i,ji1,k)
     $           - 0.50d+00 * (  u(2,i,ji1,k) ** 2
     $                         + u(3,i,ji1,k) ** 2
     $                         + u(4,i,ji1,k) ** 2 )
     $                        / u(1,i,ji1,k) )
c     
            phi2(i,k) = c2*(  u(5,i,ji2,k)
     $           - 0.50d+00 * (  u(2,i,ji2,k) ** 2
     $                         + u(3,i,ji2,k) ** 2
     $                         + u(4,i,ji2,k) ** 2 )
     $                        / u(1,i,ji2,k) )
c
         end do
c
      end do
c
      frc2 = 0.0d+00
c
      do k = ki1, ki2-1
c
         do i = ii1, ii2-1
c
            frc2 = frc2 + (  phi1(i,k)
     $                     + phi1(i+1,k)
     $                     + phi1(i,k+1)
     $                     + phi1(i+1,k+1)
     $                     + phi2(i,k)
     $                     + phi2(i+1,k)
     $                     + phi2(i,k+1)
     $                     + phi2(i+1,k+1) )
c
         end do
c
      end do
c
      frc2 = dxi * dzeta * frc2
c
      do k = ki1, ki2
c     
         do j = ji1, ji2
c
            phi1(j,k) = c2*(  u(5,ii1,j,k)
     $           - 0.50d+00 * (  u(2,ii1,j,k) ** 2
     $                         + u(3,ii1,j,k) ** 2
     $                         + u(4,ii1,j,k) ** 2 )
     $                        / u(1,ii1,j,k) )
c     
            phi2(j,k) = c2*(  u(5,ii2,j,k)
     $           - 0.50d+00 * (  u(2,ii2,j,k) ** 2
     $                         + u(3,ii2,j,k) ** 2
     $                         + u(4,ii2,j,k) ** 2 )
     $                        / u(1,ii2,j,k) )
c
         end do
c
      end do
c
      frc3 = 0.0d+00
c
      do k = ki1, ki2-1
c
         do j = ji1, ji2-1
c
            frc3 = frc3 + (  phi1(j,k)
     $                     + phi1(j+1,k)
     $                     + phi1(j,k+1)
     $                     + phi1(j+1,k+1)
     $                     + phi2(j,k)
     $                     + phi2(j+1,k)
     $                     + phi2(j,k+1)
     $                     + phi2(j+1,k+1) )
c
         end do
c
      end do
c
      frc3 = deta * dzeta * frc3
c
      frc = 0.25d+00 * ( frc1 + frc2 + frc3 )
c
      write (6,1001) frc
c
      return
c
 1001 format (//5x,'surface integral = ',1pe12.5//)
c
      end
c
c
c
c
c
      subroutine rhs
c
c***compute the right hand sides
c
c Author: Sisira Weeratunga
c         NASA Ames Research Center
c         (10/25/90)
c
c #include 'applu.incl'  

      implicit real*8 (a-h,o-z)
c
      parameter ( c1 = 1.40d+00, c2 = 0.40d+00,
     $            c3 = 1.00d-01, c4 = 1.00d+00,
     $            c5 = 1.40d+00 )
c
c***grid
c
      common/cgcon/ nx, ny, nz,
     $              ii1, ii2, ji1, ji2, ki1, ki2, itwj,
     $              dxi, deta, dzeta,
     $              tx1, tx2, tx3,
     $              ty1, ty2, ty3,
     $              tz1, tz2, tz3
c
c***dissipation
c
      common/disp/ dx1,dx2,dx3,dx4,dx5,
     $             dy1,dy2,dy3,dy4,dy5,
     $             dz1,dz2,dz3,dz4,dz5,
     $             dssp
c
c***field variables and residuals
c
      common/cvar/ u(5,33,33,33),
     $             rsd(5,33,33,33),
     $             frct(5,33,33,33)
c
c***output control parameters
c
      common/cprcon/ ipr, iout, inorm
c
c***newton-raphson iteration control parameters
c
      common/ctscon/ itmax, invert,
     $               dt, omega, tolrsd(5),
     $               rsdnm(5), errnm(5), frc, ttotal
c
      common/cjac/ a(5,5,33,33,33),
     $             b(5,5,33,33,33),
     $             c(5,5,33,33,33),
     $             d(5,5,33,33,33)
c
c***coefficients of the exact solution
c
      common/cexact/ ce(5,13)
c



c
      dimension flux(5,33)
c
      do k = 1, nz
         do j = 1, ny
            do i = 1, nx
               do m = 1, 5
                  rsd(m,i,j,k) = - frct(m,i,j,k)
               end do
            end do
         end do
      end do
c
c***xi-direction flux differences
c
      do k = 2, nz - 1
c
         do j = 2, ny - 1
c 
            do i = 1, nx
c
               flux(1,i) = u(2,i,j,k)
c
               u21 = u(2,i,j,k) / u(1,i,j,k)
c
               q = 0.50d+00 * (  u(2,i,j,k) * u(2,i,j,k)
     $                         + u(3,i,j,k) * u(3,i,j,k)
     $                         + u(4,i,j,k) * u(4,i,j,k) )
     $                      / u(1,i,j,k)
c
               flux(2,i) = u(2,i,j,k) * u21 + c2 * ( u(5,i,j,k) - q )
c
               flux(3,i) = u(3,i,j,k) * u21
c
               flux(4,i) = u(4,i,j,k) * u21
c
               flux(5,i) = ( c1 * u(5,i,j,k) - c2 * q ) * u21
c    
            end do
c
            do i = 2, nx - 1
c
               do m = 1, 5
c
                  rsd(m,i,j,k) =  rsd(m,i,j,k)
     $                       - tx2 * ( flux(m,i+1) - flux(m,i-1) )
c
               end do
c
            end do
c
            do i = 2, nx
c
               tmp = 1.0d+00 / u(1,i,j,k)
c
               u21i = tmp * u(2,i,j,k)
               u31i = tmp * u(3,i,j,k)
               u41i = tmp * u(4,i,j,k)
               u51i = tmp * u(5,i,j,k)
c
               tmp = 1.0d+00 / u(1,i-1,j,k)
c
               u21im1 = tmp * u(2,i-1,j,k)
               u31im1 = tmp * u(3,i-1,j,k)
               u41im1 = tmp * u(4,i-1,j,k)
               u51im1 = tmp * u(5,i-1,j,k)
c
               flux(2,i) = (4.0d+00/3.0d+00) * tx3 * ( u21i - u21im1 )
               flux(3,i) = tx3 * ( u31i - u31im1 )
               flux(4,i) = tx3 * ( u41i - u41im1 )
               flux(5,i) = 0.50d+00 * ( 1.0d+00 - c1*c5 )
     $              * tx3 * ( ( u21i  **2 + u31i  **2 + u41i  **2 )
     $                      - ( u21im1**2 + u31im1**2 + u41im1**2 ) )
     $              + (1.0d+00/6.0d+00)
     $              * tx3 * ( u21i**2 - u21im1**2 )
     $              + c1 * c5 * tx3 * ( u51i - u51im1 )
c
            end do
c
            do i = 2, nx - 1
c
               rsd(1,i,j,k) = rsd(1,i,j,k)
     $              + dx1 * tx1 * (            u(1,i-1,j,k)
     $                             - 2.0d+00 * u(1,i,j,k)
     $                             +           u(1,i+1,j,k) )
c
               rsd(2,i,j,k) = rsd(2,i,j,k)
     $              + tx3 * c3 * c4 * ( flux(2,i+1) - flux(2,i) )
     $              + dx2 * tx1 * (            u(2,i-1,j,k)
     $                             - 2.0d+00 * u(2,i,j,k)
     $                             +           u(2,i+1,j,k) )
c
               rsd(3,i,j,k) = rsd(3,i,j,k)
     $              + tx3 * c3 * c4 * ( flux(3,i+1) - flux(3,i) )
     $              + dx3 * tx1 * (            u(3,i-1,j,k)
     $                             - 2.0d+00 * u(3,i,j,k)
     $                             +           u(3,i+1,j,k) )
c
               rsd(4,i,j,k) = rsd(4,i,j,k)
     $              + tx3 * c3 * c4 * ( flux(4,i+1) - flux(4,i) )
     $              + dx4 * tx1 * (            u(4,i-1,j,k)
     $                             - 2.0d+00 * u(4,i,j,k)
     $                             +           u(4,i+1,j,k) )
c
               rsd(5,i,j,k) = rsd(5,i,j,k)
     $              + tx3 * c3 * c4 * ( flux(5,i+1) - flux(5,i) )
     $              + dx5 * tx1 * (            u(5,i-1,j,k)
     $                             - 2.0d+00 * u(5,i,j,k)
     $                             +           u(5,i+1,j,k) )
c
            end do
c
c***Fourth-order dissipation
c
            do m = 1, 5
c
               rsd(m,2,j,k) = rsd(m,2,j,k)
     $           - dssp * ( + 5.0d+00 * u(m,2,j,k)
     $                      - 4.0d+00 * u(m,3,j,k)
     $                      +           u(m,4,j,k) )
c
               rsd(m,3,j,k) = rsd(m,3,j,k)
     $           - dssp * ( - 4.0d+00 * u(m,2,j,k)
     $                      + 6.0d+00 * u(m,3,j,k)
     $                      - 4.0d+00 * u(m,4,j,k)
     $                      +           u(m,5,j,k) )
c
            end do
c
            do i = 4, nx - 3
c
               do m = 1, 5
c
                  rsd(m,i,j,k) = rsd(m,i,j,k)
     $              - dssp * (            u(m,i-2,j,k)
     $                        - 4.0d+00 * u(m,i-1,j,k)
     $                        + 6.0d+00 * u(m,i,j,k)
     $                        - 4.0d+00 * u(m,i+1,j,k)
     $                        +           u(m,i+2,j,k) )
c
               end do
c
            end do
c
            do m = 1, 5
c
               rsd(m,nx-2,j,k) = rsd(m,nx-2,j,k)
     $           - dssp * (             u(m,nx-4,j,k)
     $                      - 4.0d+00 * u(m,nx-3,j,k)
     $                      + 6.0d+00 * u(m,nx-2,j,k)
     $                      - 4.0d+00 * u(m,nx-1,j,k)  )
c
               rsd(m,nx-1,j,k) = rsd(m,nx-1,j,k)
     $           - dssp * (             u(m,nx-3,j,k)
     $                      - 4.0d+00 * u(m,nx-2,j,k)
     $                      + 5.0d+00 * u(m,nx-1,j,k) )
c
            end do
c
         end do
c
      end do
c
c***eta-direction flux differences
c
      do k = 2, nz - 1
c
         do i = 2, nx - 1
c 
            do j = 1, ny
c
               flux(1,j) = u(3,i,j,k)
c
               u31 = u(3,i,j,k) / u(1,i,j,k)
c
               q = 0.50d+00 * (  u(2,i,j,k) * u(2,i,j,k)
     $                         + u(3,i,j,k) * u(3,i,j,k)
     $                         + u(4,i,j,k) * u(4,i,j,k) )
     $                      / u(1,i,j,k)
c
               flux(2,j) = u(2,i,j,k) * u31 
c
               flux(3,j) = u(3,i,j,k) * u31 + c2 * ( u(5,i,j,k) - q )
c
               flux(4,j) = u(4,i,j,k) * u31
c
               flux(5,j) = ( c1 * u(5,i,j,k) - c2 * q ) * u31
c    
            end do
c
            do j = 2, ny - 1
c
               do m = 1, 5
c
                  rsd(m,i,j,k) =  rsd(m,i,j,k)
     $                       - ty2 * ( flux(m,j+1) - flux(m,j-1) )
c
               end do
c
            end do
c
            do j = 2, ny
c
               tmp = 1.0d+00 / u(1,i,j,k)
c
               u21j = tmp * u(2,i,j,k)
               u31j = tmp * u(3,i,j,k)
               u41j = tmp * u(4,i,j,k)
               u51j = tmp * u(5,i,j,k)
c
               tmp = 1.0d+00 / u(1,i,j-1,k)
c
               u21jm1 = tmp * u(2,i,j-1,k)
               u31jm1 = tmp * u(3,i,j-1,k)
               u41jm1 = tmp * u(4,i,j-1,k)
               u51jm1 = tmp * u(5,i,j-1,k)
c
               flux(2,j) = ty3 * ( u21j - u21jm1 )
               flux(3,j) = (4.0d+00/3.0d+00) * ty3 * ( u31j - u31jm1 )
               flux(4,j) = ty3 * ( u41j - u41jm1 )
               flux(5,j) = 0.50d+00 * ( 1.0d+00 - c1*c5 )
     $              * ty3 * ( ( u21j  **2 + u31j  **2 + u41j  **2 )
     $                      - ( u21jm1**2 + u31jm1**2 + u41jm1**2 ) )
     $              + (1.0d+00/6.0d+00)
     $              * ty3 * ( u31j**2 - u31jm1**2 )
     $              + c1 * c5 * ty3 * ( u51j - u51jm1 )
c
            end do
c
            do j = 2, ny - 1
c
               rsd(1,i,j,k) = rsd(1,i,j,k)
     $              + dy1 * ty1 * (            u(1,i,j-1,k)
     $                             - 2.0d+00 * u(1,i,j,k)
     $                             +           u(1,i,j+1,k) )
c
               rsd(2,i,j,k) = rsd(2,i,j,k)
     $              + ty3 * c3 * c4 * ( flux(2,j+1) - flux(2,j) )
     $              + dy2 * ty1 * (            u(2,i,j-1,k)
     $                             - 2.0d+00 * u(2,i,j,k)
     $                             +           u(2,i,j+1,k) )
c
               rsd(3,i,j,k) = rsd(3,i,j,k)
     $              + ty3 * c3 * c4 * ( flux(3,j+1) - flux(3,j) )
     $              + dy3 * ty1 * (            u(3,i,j-1,k)
     $                             - 2.0d+00 * u(3,i,j,k)
     $                             +           u(3,i,j+1,k) )
c
               rsd(4,i,j,k) = rsd(4,i,j,k)
     $              + ty3 * c3 * c4 * ( flux(4,j+1) - flux(4,j) )
     $              + dy4 * ty1 * (            u(4,i,j-1,k)
     $                             - 2.0d+00 * u(4,i,j,k)
     $                             +           u(4,i,j+1,k) )
c
               rsd(5,i,j,k) = rsd(5,i,j,k)
     $              + ty3 * c3 * c4 * ( flux(5,j+1) - flux(5,j) )
     $              + dy5 * ty1 * (            u(5,i,j-1,k)
     $                             - 2.0d+00 * u(5,i,j,k)
     $                             +           u(5,i,j+1,k) )
c
            end do
c
c***fourth-order dissipation
c
            do m = 1, 5
c
               rsd(m,i,2,k) = rsd(m,i,2,k)
     $           - dssp * ( + 5.0d+00 * u(m,i,2,k)
     $                      - 4.0d+00 * u(m,i,3,k)
     $                      +           u(m,i,4,k) )
c
               rsd(m,i,3,k) = rsd(m,i,3,k)
     $           - dssp * ( - 4.0d+00 * u(m,i,2,k)
     $                      + 6.0d+00 * u(m,i,3,k)
     $                      - 4.0d+00 * u(m,i,4,k)
     $                      +           u(m,i,5,k) )
c
            end do
c
            do j = 4, ny - 3
c
               do m = 1, 5
c
                  rsd(m,i,j,k) = rsd(m,i,j,k)
     $              - dssp * (            u(m,i,j-2,k)
     $                        - 4.0d+00 * u(m,i,j-1,k)
     $                        + 6.0d+00 * u(m,i,j,k)
     $                        - 4.0d+00 * u(m,i,j+1,k)
     $                        +           u(m,i,j+2,k) )
c
               end do
c
            end do
c
            do m = 1, 5
c
               rsd(m,i,ny-2,k) = rsd(m,i,ny-2,k)
     $           - dssp * (             u(m,i,ny-4,k)
     $                      - 4.0d+00 * u(m,i,ny-3,k)
     $                      + 6.0d+00 * u(m,i,ny-2,k)
     $                      - 4.0d+00 * u(m,i,ny-1,k)  )
c
               rsd(m,i,ny-1,k) = rsd(m,i,ny-1,k)
     $           - dssp * (             u(m,i,ny-3,k)
     $                      - 4.0d+00 * u(m,i,ny-2,k)
     $                      + 5.0d+00 * u(m,i,ny-1,k) )
c
            end do
c
         end do
c
      end do
c
c***zeta-direction flux differences
c
      do j = 2, ny - 1
c
         do i = 2, nx - 1
c
            do k = 1, nz
c
               flux(1,k) = u(4,i,j,k)
c
               u41 = u(4,i,j,k) / u(1,i,j,k)
c
               q = 0.50d+00 * (  u(2,i,j,k) * u(2,i,j,k)
     $                         + u(3,i,j,k) * u(3,i,j,k)
     $                         + u(4,i,j,k) * u(4,i,j,k) )
     $                      / u(1,i,j,k)
c
               flux(2,k) = u(2,i,j,k) * u41 
c
               flux(3,k) = u(3,i,j,k) * u41 
c
               flux(4,k) = u(4,i,j,k) * u41 + c2 * ( u(5,i,j,k) - q )
c
               flux(5,k) = ( c1 * u(5,i,j,k) - c2 * q ) * u41
c    
            end do
c
            do k = 2, nz - 1
c
               do m = 1, 5
c
                  rsd(m,i,j,k) =  rsd(m,i,j,k)
     $                       - tz2 * ( flux(m,k+1) - flux(m,k-1) )
c
               end do
c
            end do
c
            do k = 2, nz
c
               tmp = 1.0d+00 / u(1,i,j,k)
c
               u21k = tmp * u(2,i,j,k)
               u31k = tmp * u(3,i,j,k)
               u41k = tmp * u(4,i,j,k)
               u51k = tmp * u(5,i,j,k)
c
               tmp = 1.0d+00 / u(1,i,j,k-1)
c
               u21km1 = tmp * u(2,i,j,k-1)
               u31km1 = tmp * u(3,i,j,k-1)
               u41km1 = tmp * u(4,i,j,k-1)
               u51km1 = tmp * u(5,i,j,k-1)
c
               flux(2,k) = tz3 * ( u21k - u21km1 )
               flux(3,k) = tz3 * ( u31k - u31km1 )
               flux(4,k) = (4.0d+00/3.0d+00) * tz3 * ( u41k - u41km1 )
               flux(5,k) = 0.50d+00 * ( 1.0d+00 - c1*c5 )
     $              * tz3 * ( ( u21k  **2 + u31k  **2 + u41k  **2 )
     $                      - ( u21km1**2 + u31km1**2 + u41km1**2 ) )
     $              + (1.0d+00/6.0d+00)
     $              * tz3 * ( u41k**2 - u41km1**2 )
     $              + c1 * c5 * tz3 * ( u51k - u51km1 )
c
            end do
c
            do k = 2, nz - 1
c
               rsd(1,i,j,k) = rsd(1,i,j,k)
     $              + dz1 * tz1 * (            u(1,i,j,k-1)
     $                             - 2.0d+00 * u(1,i,j,k)
     $                             +           u(1,i,j,k+1) )
c
               rsd(2,i,j,k) = rsd(2,i,j,k)
     $              + tz3 * c3 * c4 * ( flux(2,k+1) - flux(2,k) )
     $              + dz2 * tz1 * (            u(2,i,j,k-1)
     $                             - 2.0d+00 * u(2,i,j,k)
     $                             +           u(2,i,j,k+1) )
c
               rsd(3,i,j,k) = rsd(3,i,j,k)
     $              + tz3 * c3 * c4 * ( flux(3,k+1) - flux(3,k) )
     $              + dz3 * tz1 * (            u(3,i,j,k-1)
     $                             - 2.0d+00 * u(3,i,j,k)
     $                             +           u(3,i,j,k+1) )
c
               rsd(4,i,j,k) = rsd(4,i,j,k)
     $              + tz3 * c3 * c4 * ( flux(4,k+1) - flux(4,k) )
     $              + dz4 * tz1 * (            u(4,i,j,k-1)
     $                             - 2.0d+00 * u(4,i,j,k)
     $                             +           u(4,i,j,k+1) )
c
               rsd(5,i,j,k) = rsd(5,i,j,k)
     $              + tz3 * c3 * c4 * ( flux(5,k+1) - flux(5,k) )
     $              + dz5 * tz1 * (            u(5,i,j,k-1)
     $                             - 2.0d+00 * u(5,i,j,k)
     $                             +           u(5,i,j,k+1) )
c
            end do
c
c***fourth-order dissipation
c
            do m = 1, 5
c
               rsd(m,i,j,2) = rsd(m,i,j,2)
     $           - dssp * ( + 5.0d+00 * u(m,i,j,2)
     $                      - 4.0d+00 * u(m,i,j,3)
     $                      +           u(m,i,j,4) )
c
               rsd(m,i,j,3) = rsd(m,i,j,3)
     $           - dssp * ( - 4.0d+00 * u(m,i,j,2)
     $                      + 6.0d+00 * u(m,i,j,3)
     $                      - 4.0d+00 * u(m,i,j,4)
     $                      +           u(m,i,j,5) )
c
            end do
c
            do k = 4, nz - 3
c
               do m = 1, 5
c
                  rsd(m,i,j,k) = rsd(m,i,j,k)
     $              - dssp * (            u(m,i,j,k-2)
     $                        - 4.0d+00 * u(m,i,j,k-1)
     $                        + 6.0d+00 * u(m,i,j,k)
     $                        - 4.0d+00 * u(m,i,j,k+1)
     $                        +           u(m,i,j,k+2) )
c
               end do
c
            end do
c
            do m = 1, 5
c
               rsd(m,i,j,nz-2) = rsd(m,i,j,nz-2)
     $           - dssp * (             u(m,i,j,nz-4)
     $                      - 4.0d+00 * u(m,i,j,nz-3)
     $                      + 6.0d+00 * u(m,i,j,nz-2)
     $                      - 4.0d+00 * u(m,i,j,nz-1)  )
c
               rsd(m,i,j,nz-1) = rsd(m,i,j,nz-1)
     $           - dssp * (             u(m,i,j,nz-3)
     $                      - 4.0d+00 * u(m,i,j,nz-2)
     $                      + 5.0d+00 * u(m,i,j,nz-1) )
c
            end do
c
         end do
c
      end do
c
      return
      end
c
c
c
c
c
      subroutine ssor
c
c***to perform pseudo-time stepping SSOR iterations
c   for five nonlinear pde's.
c
c Author: Sisira Weeratunga
c         NASA Ames Research Center
c         (10/25/90)
c
c #include 'applu.incl'  

      implicit real*8 (a-h,o-z)
c
      parameter ( c1 = 1.40d+00, c2 = 0.40d+00,
     $            c3 = 1.00d-01, c4 = 1.00d+00,
     $            c5 = 1.40d+00 )
c
c***grid
c
      common/cgcon/ nx, ny, nz,
     $              ii1, ii2, ji1, ji2, ki1, ki2, itwj,
     $              dxi, deta, dzeta,
     $              tx1, tx2, tx3,
     $              ty1, ty2, ty3,
     $              tz1, tz2, tz3
c
c***dissipation
c
      common/disp/ dx1,dx2,dx3,dx4,dx5,
     $             dy1,dy2,dy3,dy4,dy5,
     $             dz1,dz2,dz3,dz4,dz5,
     $             dssp
c
c***field variables and residuals
c
      common/cvar/ u(5,33,33,33),
     $             rsd(5,33,33,33),
     $             frct(5,33,33,33)
c
c***output control parameters
c
      common/cprcon/ ipr, iout, inorm
c
c***newton-raphson iteration control parameters
c
      common/ctscon/ itmax, invert,
     $               dt, omega, tolrsd(5),
     $               rsdnm(5), errnm(5), frc, ttotal
c
      common/cjac/ a(5,5,33,33,33),
     $             b(5,5,33,33,33),
     $             c(5,5,33,33,33),
     $             d(5,5,33,33,33)
c
c***coefficients of the exact solution
c
      common/cexact/ ce(5,13)
c



c
      dimension idmax(5), jdmax(5), kdmax(5),
     $          imax(5), jmax(5), kmax(5),
     $          delunm(5)
      parameter ( one = 1.0d+00 )
c
      lnorm = 2
c
c***begin pseudo-time stepping iterations
c
      tmp = 1.0d+00 / ( omega * ( 2.0d+00 - omega ) ) 
c
c***compute the steady-state residuals
c
      call rhs
c
c***compute the max-norms of newton iteration residuals
c
      if ( lnorm .eq. 1 ) then
c
         call maxnorm ( 33, 33, 33,
     $                   nx, ny, nz,
     $                   imax, jmax, kmax,
     $                   rsd, rsdnm )
c
         if ( ipr .eq. 1 ) then
c
            write (6,*) '          Initial residual norms'
            write (6,*)
            write (6,1003) ( rsdnm(m),
     $           imax(m), jmax(m), kmax(m), m = 1, 5 )
c
         end if
c
      else if ( lnorm .eq. 2 ) then
c
         call l2norm ( 33, 33, 33,
     $                 nx, ny, nz,
     $                 rsd, rsdnm )
c
         if ( ipr .eq. 1 ) then
c
            write (6,*) '          Initial residual norms'
            write (6,*)
            write (6,1007) ( rsdnm(m), m = 1, 5 )
c     
         end if
c     
      end if
c     
      do istep = 1, itmax
c
         if ( ( mod ( istep, inorm ) .eq. 0 ) .and.
     $        ( ipr .eq. 1 ) ) then
c
            write ( 6, 1001 ) istep
c
         end if
c
c***perform SSOR iteration
c
         do k = 2, nz - 1
            do j = 2, ny - 1
               do i = 2, nx - 1
                  do m = 1, 5
c
                     rsd(m,i,j,k) = dt * rsd(m,i,j,k)
c     
                  end do
               end do
            end do
         end do
c
c***form the lower triangular part of the jacobian matrix
c
         call jacld
c
c***perform the lower triangular solution
c
         call blts ( 33, 33, 33,
     $               nx, ny, nz,
     $               omega,
     $               rsd,
     $               a, b, c, d )
c
c***form the strictly upper triangular part of the jacobian matrix
c
         call jacu
c
c***perform the upper triangular solution
c
         call buts ( 33, 33, 33,
     $               nx, ny, nz,
     $               omega,
     $               rsd,
     $               d, a, b, c )
c
c***update the variables
c
         do k = 2, nz-1
            do j = 2, ny-1
               do i = 2, nx-1
                  do m = 1, 5
c
                     u( m, i, j, k ) = u( m, i, j, k )
     $                    + tmp * rsd( m, i, j, k )
c
                  end do
               end do
            end do
         end do
c
c***compute the max-norms of newton iteration corrections
c
         if ( mod ( istep, inorm ) .eq. 0 ) then
c
            if ( lnorm .eq. 1 ) then
c     
               call maxnorm ( 33, 33, 33,
     $                        nx, ny, nz,
     $                        idmax, jdmax, kdmax,
     $                        rsd, delunm )
c
               if ( ipr .eq. 1 ) then
c
                  write (6,1002) ( delunm(m),
     $                 idmax(m), jdmax(m), kdmax(m), m = 1, 5 )
c
               end if
c
            else if ( lnorm .eq. 2 ) then
c
               call l2norm ( 33, 33, 33,
     $                       nx, ny, nz,
     $                       rsd, delunm )
c
               if ( ipr .eq. 1 ) then
c
                  write (6,1006) ( delunm(m), m = 1, 5 )
c
               end if
c
            end if
c
         end if
c
c***compute the steady-state residuals
c
         call rhs
c
c***compute the max-norms of newton iteration residuals
c
         if ( ( mod ( istep, inorm ) .eq. 0 ) .or.
     $        ( istep .eq. itmax ) ) then
c
            if ( lnorm .eq. 1 ) then
c
               call maxnorm ( 33, 33, 33,
     $                        nx, ny, nz,
     $                        imax, jmax, kmax,
     $                        rsd, rsdnm )
c
               if ( ipr .eq. 1 ) then
c
                  write (6,1003) ( rsdnm(m),
     $                 imax(m), jmax(m), kmax(m), m = 1, 5 )
c
               end if
c
            else if ( lnorm .eq. 2 ) then
c
               call l2norm ( 33, 33, 33,
     $                       nx, ny, nz,
     $                       rsd, rsdnm )
c
               if ( ipr .eq. 1 ) then
c
                  write (6,1007) ( rsdnm(m), m = 1, 5 )
c
               end if
c
            end if
c
         end if
c
c***check the newton-iteration residuals against the tolerance levels
c
         if ( ( rsdnm(1) .lt. tolrsd(1) ) .and.
     $        ( rsdnm(2) .lt. tolrsd(2) ) .and.
     $        ( rsdnm(3) .lt. tolrsd(3) ) .and.
     $        ( rsdnm(4) .lt. tolrsd(4) ) .and.
     $        ( rsdnm(5) .lt. tolrsd(5) ) ) then
c
            write (6,1004) istep
            return
c
         end if
c
      end do
c     
      return
c     
 1001 format (1x/5x,'pseudo-time SSOR iteration no.=',i4/)
c
 1002 format (1x/1x,'max-norm of SSOR-iteration correction ',
     $ 'for first pde  = ',1pe12.5/,
     $ 55x,'(',i4,',',i4,',',i4,')'/,
     $ 1x,'max-norm of SSOR-iteration correction ',
     $ 'for second pde = ',1pe12.5/,
     $ 55x,'(',i4,',',i4,',',i4,')'/,
     $ 1x,'max-norm of SSOR-iteration correction ',
     $ 'for third pde  = ',1pe12.5/,
     $ 55x,'(',i4,',',i4,',',i4,')'/,
     $ 1x,'max-norm of SSOR-iteration correction ',
     $ 'for fourth pde = ',1pe12.5/,
     $ 55x,'(',i4,',',i4,',',i4,')'/,
     $ 1x,'max-norm of SSOR-iteration correction ',
     $ 'for fifth pde  = ',1pe12.5/,
     $ 55x,'(',i4,',',i4,',',i4,')' )
c
 1003 format (1x/1x,'max-norm of steady-state residual for ',
     $ 'first pde  = ',1pe12.5/,
     $ 51x,'(',i4,',',i4,',',i4,')'/,
     $ 1x,'max-norm of steady-state residual for ',
     $ 'second pde = ',1pe12.5/,
     $ 51x,'(',i4,',',i4,',',i4,')'/,
     $ 1x,'max-norm of steady-state residual for ',
     $ 'third pde  = ',1pe12.5/,
     $ 51x,'(',i4,',',i4,',',i4,')'/,
     $ 1x,'max-norm of steady-state residual for ',
     $ 'fourth pde = ',1pe12.5/,
     $ 51x,'(',i4,',',i4,',',i4,')'/,
     $ 1x,'max-norm of steady-state residual for ',
     $ 'fifth pde  = ',1pe12.5/,
     $ 51x,'(',i4,',',i4,',',i4,')' )
c
 1004 format (1x/1x,'convergence was achieved after ',i4,
     $   ' pseudo-time steps' )
c
 1006 format (1x/1x,'RMS-norm of SSOR-iteration correction ',
     $ 'for first pde  = ',1pe12.5/,
     $ 1x,'RMS-norm of SSOR-iteration correction ',
     $ 'for second pde = ',1pe12.5/,
     $ 1x,'RMS-norm of SSOR-iteration correction ',
     $ 'for third pde  = ',1pe12.5/,
     $ 1x,'RMS-norm of SSOR-iteration correction ',
     $ 'for fourth pde = ',1pe12.5/,
     $ 1x,'RMS-norm of SSOR-iteration correction ',
     $ 'for fifth pde  = ',1pe12.5)
c
 1007 format (1x/1x,'RMS-norm of steady-state residual for ',
     $ 'first pde  = ',1pe12.5/,
     $ 1x,'RMS-norm of steady-state residual for ',
     $ 'second pde = ',1pe12.5/,
     $ 1x,'RMS-norm of steady-state residual for ',
     $ 'third pde  = ',1pe12.5/,
     $ 1x,'RMS-norm of steady-state residual for ',
     $ 'fourth pde = ',1pe12.5/,
     $ 1x,'RMS-norm of steady-state residual for ',
     $ 'fifth pde  = ',1pe12.5)
c
      end
c
c
c
c
c
      subroutine maxnorm ( ldx, ldy, ldz,
     $                     nx, ny, nz,
     $                     imax, jmax, kmax, v, vnm )
c
c***compute the max-norm of vector v.
c
c Author: Sisira Weeratunga
c         NASA Ames Research Center
c         (10/25/90)
c
      implicit real*8 (a-h,o-z)
      dimension v(5,ldx,ldy,*),
     $          vnm(*), imax(*), jmax(*), kmax(*)
c
      do m = 1, 5
c
         vnm(m) = -1.0d+10
c
      end do
c
      do k = 1, nz
         do j = 1, ny
            do i = 1, nx
c
               do m = 1, 5
c
                  t1 = abs ( v( m, i, j, k ) )
c
                  if ( vnm(m) .lt. t1 ) then
                     vnm(m) = t1
                     imax(m) = i
                     jmax(m) = j
                     kmax(m) = k
                  end if
c
               end do
c
            end do
c
         end do
c
      end do
c
      return
      end
c
c
c
c
c
      subroutine l2norm ( ldx, ldy, ldz,
     $                    nx, ny, nz,
     $                    v, sum )
c
c***to compute the l2-norm of vector v.
c
c Author: Sisira Weeratunga
c         NASA Ames Research Center
c         (10/25/90)
c
      implicit real*8 (a-h,o-z)
      dimension v( 5, ldx, ldy, * ),
     $          sum(*)
c
      do m = 1, 5
c
         sum(m) = 0.0d+00
c
      end do
c
      do k = 2, nz-1
         do j = 2, ny-1
            do i = 2, nx-1
c
               do m = 1, 5
                  sum(m) = sum(m) + v(m,i,j,k) * v(m,i,j,k)
               end do
c
            end do
         end do
      end do
c
      do m = 1, 5
c
         sum(m) = sqrt ( sum(m) / ( (nx-2)*(ny-2)*(nz-2) ) )
c
      end do
c
      return
      end
c
c
c
c
c
      subroutine verify ( xcr, xce, xci )
c
c***verification routine
c
c Author: Sisira Weeratunga 
c         NASA Ames Research Center
c         (10/25/90)
c
c #include 'applu.incl'  

      implicit real*8 (a-h,o-z)
c
      parameter ( c1 = 1.40d+00, c2 = 0.40d+00,
     $            c3 = 1.00d-01, c4 = 1.00d+00,
     $            c5 = 1.40d+00 )
c
c***grid
c
      common/cgcon/ nx, ny, nz,
     $              ii1, ii2, ji1, ji2, ki1, ki2, itwj,
     $              dxi, deta, dzeta,
     $              tx1, tx2, tx3,
     $              ty1, ty2, ty3,
     $              tz1, tz2, tz3
c
c***dissipation
c
      common/disp/ dx1,dx2,dx3,dx4,dx5,
     $             dy1,dy2,dy3,dy4,dy5,
     $             dz1,dz2,dz3,dz4,dz5,
     $             dssp
c
c***field variables and residuals
c
      common/cvar/ u(5,33,33,33),
     $             rsd(5,33,33,33),
     $             frct(5,33,33,33)
c
c***output control parameters
c
      common/cprcon/ ipr, iout, inorm
c
c***newton-raphson iteration control parameters
c
      common/ctscon/ itmax, invert,
     $               dt, omega, tolrsd(5),
     $               rsdnm(5), errnm(5), frc, ttotal
c
      common/cjac/ a(5,5,33,33,33),
     $             b(5,5,33,33,33),
     $             c(5,5,33,33,33),
     $             d(5,5,33,33,33)
c
c***coefficients of the exact solution
c
      common/cexact/ ce(5,13)
c



c
      dimension xcr(5), xce(5),
     $          xrr(5), xre(5)
c
      if ( ( nx .eq. 12 ) .and.
     $     ( ny .eq. 12 ) .and.
     $     ( nz .eq. 12 ) ) then
c
c***tolerance level
c
         epsilon = 1.0d-08
c
c***Reference values of RMS-norms of residual, for the (12X12X12) grid,
c   after 50 time steps, with  DT = 5.0d-01
c
         xrr(1) = 1.6196343210976702d-02
         xrr(2) = 2.1976745164821318d-03
         xrr(3) = 1.5179927653399185d-03
         xrr(4) = 1.5029584435994323d-03
         xrr(5) = 3.4264073155896461d-02
c
c***Reference values of RMS-norms of solution error, for the (12X12X12) grid,
c   after 50 time steps, with  DT = 5.0d-01
c
         xre(1) = 6.4223319957960924d-04
         xre(2) = 8.4144342047347926d-05
         xre(3) = 5.8588269616485186d-05
         xre(4) = 5.8474222595157350d-05
         xre(5) = 1.3103347914111294d-03
c
c***Reference value of surface integral, for the (12X12X12) grid,
c   after 50 time steps, with DT = 5.0d-01
c
         xri = 7.8418928865937083d+00
c
c***verification test for residuals
c
         do m = 1, 5
c     
            tmp = abs ( ( xcr(m) - xrr(m) ) / xrr(m) )
c     
            if ( tmp .gt. epsilon ) then
c     
               write (6,1001) 
 1001          format(/5x,'VERIFICATION TEST FOR RESIDUALS FAILED')
c     
               go to 100
c
            end if
c     
         end do
c     
         write (6,1002) 
 1002    format (/5x,'VERIFICATION TEST FOR RESIDUALS ',
     $        'IS SUCCESSFUL')
c     
c***verification test for solution error
c
 100     continue
c     
         do m = 1, 5
c
            tmp = abs ( ( xce(m) - xre(m) ) / xre(m) )
c     
            if ( tmp .gt. epsilon ) then
c     
               write (6,1003) 
 1003          format(/5x,'VERIFICATION TEST FOR SOLUTION ',
     $              'ERRORS FAILED')
c     
               go to 200
c     
            end if
c     
         end do
c     
         write (6,1004) 
 1004    format (/5x,'VERIFICATION TEST FOR SOLUTION ERRORS ',
     $        'IS SUCCESSFUL')
c
c***verification test for surface integral
c
 200     continue
c
         tmp = abs ( ( xci - xri ) / xri )
c     
         if ( tmp .gt. epsilon ) then
c     
            write (6,1005)
 1005       format (/5x,'VERIFICATION TEST FOR SURFACE INTEGRAL FAILED')
c     
         else
c     
            write (6,1006)
 1006       format (/5x,'VERIFICATION TEST FOR SURFACE INTEGRAL ',
     $           'IS SUCCESSFUL')
c     
         end if
c     
         write (6,1007)
 1007    format(//10x,'CAUTION',
     $ //5x,'REFERENCE VALUES CURRENTLY IN THIS VERIFICATION ',
     $     'ROUTINE ',
     $ /5x,'ARE VALID ONLY FOR RUNS WITH THE FOLLOWING PARAMETER ',
     $     'VALUES:',
     $ //5x,'NX = 12;  NY = 12;  NZ = 12 ',
     $ //5x,'ITMAX = 50',
     $ //5x,'DT = 5.0d-01',
     $ //5x,'OMEGA = 1.2',
     $ //5x,'CHANGE IN ANY OF THE ABOVE VALUES RENDER THE REFERENCE ',
     $     'VALUES ',
     $ /5x,'INVALID AND CAUSES A FAILURE OF THE VERIFICATION TEST.')
c
      else if ( ( nx .eq. 64 ) .and.
     $          ( ny .eq. 64 ) .and.
     $          ( nz .eq. 64 ) ) then
c
c***tolerance level
c
         epsilon = 1.0d-08
c
c***Reference values of RMS-norms of residual, for the (64X64X64) grid,
c   after 250 time steps, with  DT = 2.0d+00
c
         xrr(1) = 7.7902107606689367d+02
         xrr(2) = 6.3402765259692870d+01
         xrr(3) = 1.9499249727292479d+02
         xrr(4) = 1.7845301160418537d+02
         xrr(5) = 1.8384760349464247d+03
c     
c***Reference values of RMS-norms of solution error, for the (64X64X64) grid,
c   after 250 time steps, with  DT = 2.0d+00
c
         xre(1) = 2.9964085685471943d+01
         xre(2) = 2.8194576365003349d+00
         xre(3) = 7.3473412698774742d+00
         xre(4) = 6.7139225687777051d+00
         xre(5) = 7.0715315688392578d+01
c
c***Reference value of surface integral, for the (64X64X64) grid,
c   after 250 time steps, with DT = 2.0d+00
c
         xri = 2.6030925604886277d+01
c
c***verification test for residuals
c
         do m = 1, 5
c     
            tmp = abs ( ( xcr(m) - xrr(m) ) / xrr(m) )
c     
            if ( tmp .gt. epsilon ) then
c     
               write (6,1001) 
               go to 400
c
            end if
c     
         end do
c     
         write (6,1002) 
c     
c***verification test for solution error
c
 400     continue
c     
         do m = 1, 5
c
            tmp = abs ( ( xce(m) - xre(m) ) / xre(m) )
c     
            if ( tmp .gt. epsilon ) then
c     
               write (6,1003) 
               go to 500
c     
            end if
c     
         end do
c     
         write (6,1004) 
c
c***verification test for surface integral
c
 500     continue
c
         tmp = abs ( ( xci - xri ) / xri )
c     
         if ( tmp .gt. epsilon ) then
c     
            write (6,1005)
c     
         else
c     
            write (6,1006)
c     
         end if
c     
         write (6,1008)
 1008    format(//10x,'CAUTION',
     $ //5x,'REFERENCE VALUES CURRENTLY IN THIS VERIFICATION ',
     $     'ROUTINE ',
     $ /5x,'ARE VALID ONLY FOR RUNS WITH THE FOLLOWING PARAMETER ',
     $     'VALUES:',
     $ //5x,'NX = 64;  NY = 64;  NZ = 64 ',
     $ //5x,'ITMAX = 250',
     $ //5x,'DT = 2.0d+00',
     $ //5x,'OMEGA = 1.2',
     $ //5x,'CHANGE IN ANY OF THE ABOVE VALUES RENDER THE REFERENCE ',
     $     'VALUES ',
     $ /5x,'INVALID AND CAUSES A FAILURE OF THE VERIFICATION TEST.')
c
      else if ( ( nx .eq. 102 ) .and.
     $          ( ny .eq. 102 ) .and.
     $          ( nz .eq. 102 ) ) then
c
c***tolerance level
c
         epsilon = 1.0d-08
c
c***Reference values of RMS-norms of residual, for the (102X102X102) grid,
c   after 250 time steps, with  DT = 2.0d+00
c
         xrr(1) = 3.5532672969982736d+03
         xrr(2) = 2.6214750795310692d+02
         xrr(3) = 8.8333721850952190d+02
         xrr(4) = 7.7812774739425265d+02
         xrr(5) = 7.3087969592545314d+03
c     
c***Reference values of RMS-norms of solution error, for the (102X102X102) 
c   grid, after 250 time steps, with  DT = 2.0d+00
c
         xre(1) = 1.1401176380212709d+02
         xre(2) = 8.1098963655421574d+00
         xre(3) = 2.8480597317698308d+01
         xre(4) = 2.5905394567832939d+01
         xre(5) = 2.6054907504857413d+02
c
c***Reference value of surface integral, for the (102X102X102) grid,
c   after 250 time steps, with DT = 2.0d+00
c
         xri = 4.7887162703308227d+01
c
c***verification test for residuals
c
         do m = 1, 5
c     
            tmp = abs ( ( xcr(m) - xrr(m) ) / xrr(m) )
c     
            if ( tmp .gt. epsilon ) then
c     
               write (6,1001) 
               go to 600
c
            end if
c     
         end do
c     
         write (6,1002) 
c     
c***verification test for solution error
c
 600     continue
c     
         do m = 1, 5
c
            tmp = abs ( ( xce(m) - xre(m) ) / xre(m) )
c     
            if ( tmp .gt. epsilon ) then
c     
               write (6,1003) 
               go to 700
c     
            end if
c     
         end do
c     
         write (6,1004) 
c
c***verification test for surface integral
c
 700     continue
c
         tmp = abs ( ( xci - xri ) / xri )
c     
         if ( tmp .gt. epsilon ) then
c     
            write (6,1005)
c     
         else
c     
            write (6,1006)
c     
         end if
c     
         write (6,1009)
 1009    format(//10x,'CAUTION',
     $ //5x,'REFERENCE VALUES CURRENTLY IN THIS VERIFICATION ',
     $     'ROUTINE ',
     $ /5x,'ARE VALID ONLY FOR RUNS WITH THE FOLLOWING PARAMETER ',
     $     'VALUES:',
     $ //5x,'NX = 102;  NY = 102;  NZ = 102 ',
     $ //5x,'ITMAX = 250',
     $ //5x,'DT = 2.0d+00',
     $ //5x,'OMEGA = 1.2',
     $ //5x,'CHANGE IN ANY OF THE ABOVE VALUES RENDER THE REFERENCE ',
     $     'VALUES ',
     $ /5x,'INVALID AND CAUSES A FAILURE OF THE VERIFICATION TEST.')
c
      else
c
         write (6,1010) 
 1010    format (//1x,'FOR THE PROBLEM PARAMETERS IN USE ',
     $        'NO REFERENCE VALUES ARE PROVIDED'
     $        /1x,'IN THE CURRENT VERIFICATION ROUTINE - ',
     $        'NO VERIFIACTION TEST WAS PERFORMED')
c
      end if
c
      return
      end
