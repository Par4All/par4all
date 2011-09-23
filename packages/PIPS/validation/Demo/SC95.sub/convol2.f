      program image_processing
c     goal: show effect of cloning, partial evaluation and loop unrolling
c     and reduction parallelization for a Power architecture

c     kernel_size must be even
      parameter (image_size=512, kernel_size=3, nsteps=20)
      real image(image_size,image_size)
      real new_image(image_size,image_size)
      real kernel(kernel_size, kernel_size)

      do i = 1, kernel_size
         do j = 1, kernel_size
            kernel(i,j) = 1.
         enddo
      enddo

c     read *, image
      do i = 1, image_size
         do j = 1, image_size
            image(i,j) = 1.
         enddo
      enddo

      do n = 1, nsteps
         call convol(new_image, image, image_size, image_size, 
     &        kernel, kernel_size, kernel_size)
      enddo

c     print *, new_image
      print *, new_image (image_size/2, image_size/2)

      end

      subroutine convol(new_image, image, isi, isj, kernel, ksi, ksj)
c     The convolution kernel is not applied on the outer part
c     of the image
      real image(isi,isj)
      real new_image(isi,isj)
      real kernel(ksi,ksj)


      do i = 1, isi
         do j = 1, isj
            new_image(i,j) = image(i,j)
         enddo
      enddo
      

      do 400 i = 1 + ksi/2, isi - ksi/2
         do 300 j = 1 + ksj/2, isj - ksj/2
            new_image(i,j) = 0.
            do 200 ki = 1, ksi
               do 100 kj = 1, ksj
                  new_image(i,j) = new_image(i,j) + 
     &                 image(i+ki-ksi/2-1,j+kj-ksj/2-1)* 
     &                 kernel(ki,kj)
 100           continue
 200        continue
            new_image(i,j) = new_image(i,j)/(ksi*ksj)
 300     continue
 400  continue

      end
