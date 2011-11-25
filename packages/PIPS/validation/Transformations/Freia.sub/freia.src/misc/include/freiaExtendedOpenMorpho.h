/*
   * \defgroup freia_ecipo_morpho  Mathematical morphology extended complex operations
   * \ingroup freia_ecipo
   * @{
   */


/**

   \brief inf of closing in all directions of the connexity grid

     The given connexity :
    - 8-connexity : 3x3 square
    - 6-connexity : hexagon of radius 1
    - 4-connexity : rhombus (inscribed in a square 3x3)



    \param[out] imout destination image 
    \param[in] imin source image
    \param[in] connexity 4,6 or 8 connexity
    \param[in] size size the structuring element
    \return error code

*/
freia_status freia_ecipo_inf_close(freia_data2d *imOut, freia_data2d *imIn, int32_t connexity, int32_t size);

/**
   \brief \brief sup of opening in all directions of the connexity grid

   \copydoc freia_ecipo_inf_close()
*/
freia_status freia_ecipo_sup_open(freia_data2d *imOut, freia_data2d *imIn, int32_t connexity, int32_t size);



/**
    \brief Erode and image using a specific kernel, connexity and a specific size
    
     The given connexity :
    - 8-connexity : 3x3 square
    - 6-connexity : hexagon of radius 1
    - 4-connexity : rhombus (inscribed in a square 3x3)
    

    The kernel definition is refering to the one defined in \see freia_aipo_erode


    Size represents the radius of structuring element. In practice,
    the size parameters is used to repeat 'size' time the erosion
    operation with a unitary structuring element.

    The operation could be done in-place 

    \param[out] imout destination image 
    \param[in] imin source image
    \param[in] kernel array of neighbors
    \param[in] connexity 4,6 or 8 connexity
    \param[in] size size the structuring element
    \return error code
*/
freia_status freia_ecipo_erode(freia_data2d *imout, freia_data2d *imin, const int32_t *neighbor, int32_t connexity, uint32_t size);

/** 
    \brief  Dilate and image using a specific kernel, connexity and a specific size
    
    \copydoc freia_ecipo_erode()

*/
freia_status freia_ecipo_dilate(freia_data2d *imout, freia_data2d *imin, const int32_t *neighbor, int32_t connexity, uint32_t size);

/** 
    \brief  Close and image using a specific kernel, connexity and a specific size
    
    \copydoc freia_ecipo_erode()

*/
freia_status freia_ecipo_close(freia_data2d *imout, freia_data2d *imin, const int32_t *neighbor, int32_t connexity, uint32_t size);

/** 
    \brief  Open and image using a specific kernel, connexity and a specific size
    
    \copydoc freia_ecipo_erode()

*/
freia_status freia_ecipo_open(freia_data2d *imout, freia_data2d *imin, const int32_t *neighbor, int32_t connexity, uint32_t size);


 /*!@}*/


void freia_ecipo_distance(freia_data2d *imOut, freia_data2d *imIn, const int32_t connexity);

