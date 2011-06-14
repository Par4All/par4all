       program NS
       parameter (nvar=3,nxm=2000,nym=2000)
       real phi(nvar,nxm,nym),phi1(nvar,nxm,nym)
       real phides(nvar,nym)
       real dist(nxm,nym),xnor(2,nxm,nym),sgn(nxm,nym)
       real xcoef(nxm,nym),xpt(nxm),ypt(nxm)

         do j=2,ny-1
         do i=2,nx-1
            xco=xcoef(i,j)
            px=(phi1(3,i+1,j)-phi1(3,i-1,j))*h1p2
            py=(phi1(3,i,j+1)-phi1(3,i,j-1))*h2p2
            phi1(1,i,j)=phi1(1,i,j)-dt*px*xco
            phi1(2,i,j)=phi1(2,i,j)-dt*py*xco
         enddo
         enddo
         call a()
         call b()
           end

      subroutine a
      parameter (nvar=3,nxm=2000,nym=2000)
      real b(nxm)
      do i = 1, nxm
         b(i) = i
      enddo
      end

      subroutine b
      parameter (nvar=3,nxm=2000,nym=2000)
      real c(nxm)
      do i = 1, nxm
         c(i) = 0
      enddo
      end
