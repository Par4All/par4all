       program NS
       parameter (nvar=3,nxm=2000,nym=2000)
       real phi1(nvar,nxm,nym)
       real xcoef(nxm,nym)

       nx=101
       ny=101

! This loop nest should be parallelized on j
! xco, px and py should be privatized
       do j=2,ny-1
          do i=2,nx-1
             xco=xcoef(i,j)
             px=(phi1(3,i+1,j)-phi1(3,i-1,j))*h1p2
             py=(phi1(3,i,j+1)-phi1(3,i,j-1))*h2p2
             phi1(1,i,j)=phi1(1,i,j)-dt*px*xco
             phi1(2,i,j)=phi1(2,i,j)-dt*py*xco
          enddo
       enddo
       end
