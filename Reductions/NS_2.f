       program NS
       parameter (nvar=3,nxm=2000,nym=2000)
       real phi1(nvar,nxm,nym)

        nx=101
        ny=101

! Should be parallelized with an OpenMP reduction:
        presmoy=0.
        do i=1,nx
           do j=1,ny
              presmoy=presmoy+phi1(3,i,j)
           enddo
        enddo
        presmoy=presmoy/(nx*ny)
! Should be parallelized on the "do i" and j should be privatized:
        do i=1,nx
           do j=1,ny
              phi1(3,i,j)=phi1(3,i,j)-presmoy
           enddo
        enddo

        end
