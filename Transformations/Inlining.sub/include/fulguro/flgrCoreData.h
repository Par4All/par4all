/****************************************************************
 * Fulguro
 * Copyright (C) 2004 Christophe Clienti
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the
 * Free Software Foundation, Inc., 59 Temple Place - Suite 330,
 * Boston, MA 02111-1307, USA.
 ***************************************************************/

#ifdef __cplusplus
extern "C" {
#endif


#ifndef __FLGR_CORE_DATA_H
#define __FLGR_CORE_DATA_H

#include <flgrCoreDll.h>
#include <flgrCoreErrors.h>
#include <flgrCoreTypes.h>

  /*! \mainpage Fulguro Documentation
   *
   *
   * \section intro_sec Introduction
   *
   * Fulgoro is library for image processing with real-time constraints. It use SIMD optimizations when available
   * and give the opportunity to use automatic smart threads to speed-up processing.
   * Python or Ruby extension could be used to easily "script" applications while preserving the computation speed
   * <BR>
   * <BR>
   * <HR>
   *
   * \section wrapper_doc Wrapper Documentation
   * - <A href="wrapper/python.html">Python Fulguro Module Documentation</A><BR>
   * - <A href="wrapper/ruby.html">Ruby Fulguro Module Documentation</A><BR>
   * 
   * <HR>
   *
   * \section install_sec Installation
   * 
   * <B> * NEW * </B> CVS repository is deprecated and no longer updated, please use SVN repository
   *
   * <A href="http://fulguro.svn.sourceforge.net/viewvc/fulguro/fulguro/trunk.tar.gz?view=tar">Download svn fulguro/trunk Snapshot</A>
   * or retreive source project from SVN :<BR>
   * svn co https://fulguro.svn.sourceforge.net/svnroot/fulguro fulguro
   * <BR>
   * Check the root Makefile.include in the trunk directory to specify correct installation and dependences path.<BR>
   * <BR> To compile fulguro :
   * - Linux :
   *   - Check "Makefile.include" and modify it to your needs
   *   - Make
   *   - Make install
   *   - Make tests (try it multiple times because of fft wisdom file creation)
   *   - If LD_LIBRARY_PATH is correctly set to install directory, you can try test script in the swig directory or in the apps directory<BR><BR>
   * - Windows (mingw32 gcc 3.4.5) :
   *     - Open "fulguro_workspace.workspace" with <A href="http://www.codeblocks.org/">Code::Blocks</A>
   *     - Rebuild workspace fulguro_xxxx
   *     - Rebuild workspace winswig_python
   *
   * A CMake project is being prepared
   *
   * You should update your PYTHONPATH variable to /<install_lib_path>/pythonFulguro
   *
   * \subsection tools_subsec Tools required for compilation:
   * - python 2.5 devel
   * - ruby 1.8 devel
   * - libpng devel
   * - libjpeg devel
   * - libtiff4 devel
   * - swig (1.3.31 or better)
   * - fftw3 devel
   * - SDL devel (needed for Realtime Display Module and Threads management)
   * - Video4Linux devel (needed only for Realtime Capture Module)
   * - Doxygen (needed for documentation generation)
   * - Graphviz (needed for documentation generation)
   *
   * \subsection using_fulguro Using Fulguro Python or Ruby Scripts:
   * To run correctly python or ruby scripts, you will need <A href="http://nxv.sourceforge.net/">NxV</A>
   * (a multiplatform image viewer).
   *
   * <HR>
   *
   * \section project_website Sourceforge Project Website
   * \sa Project Website : http://sourceforge.net/projects/fulguro
   * \sa SVN Browse Website : http://fulguro.svn.sourceforge.net
   *
   * <HR>
   *
   * \section project_needs What is missing in Fulguro, in your opinion?
   * - <A href="php/poll.html">Vote now</A>
   * - <A href="php/poll_result.php">See results</A>
   *
   * <HR>
   *
   * \section bench Some benchmarks
   * - <A href="bench_IntelR_PentiumR_4_CPU_3.00GHz____Cache_2048_KB.html">Pentium IV Hyper-Threading</A><BR>
   * - <A href="bench_Dual_Core_AMD_Opterontm_Processor_280____Cache_1024_KB.html">Dual-Core Opteron 280</A><BR>
   * - <A href="bench_IntelR_XeonTM_CPU_3.20GHz____Cache_1024_KB.html">Xeon Hyper-Threading</A><BR>
   * - <A href="bench_Genuine_IntelR_CPU___________T2300____1.66GHz____Cache_2048_KB.html">Centrino Duo T2300</A><BR>
   * - <A href="bench_AMD_Athlontm_64_Processor_3200_____Cache_512_KB.html">Athlon 64 3200+</A><BR>
   * <BR><BR>
   *
   * <HR>
   *
   * \section Cost Estimation
   * <A href="cost_estimation.html">Cost estimation to develop fulguro</A><BR>
   * <BR>
   * <BR>
   * <HR>
   *
   * \section copyright Copyright and License
   * GNU LESSER GENERAL PUBLIC LICENSE
   *                      Version 3, 29 June 2007
   *
   *  Copyright (C) 2007 Free Software Foundation, Inc. <http://fsf.org/>
   *  Everyone is permitted to copy and distribute verbatim copies
   *  of this license document, but changing it is not allowed.
   * 
   * 
   *   This version of the GNU Lesser General Public License incorporates
   * the terms and conditions of version 3 of the GNU General Public
   * License, supplemented by the additional permissions listed below.
   * 
   *   0. Additional Definitions. 
   * 
   *   As used herein, "this License" refers to version 3 of the GNU Lesser
   * General Public License, and the "GNU GPL" refers to version 3 of the GNU
   * General Public License.
   * 
   *   "The Library" refers to a covered work governed by this License,
   * other than an Application or a Combined Work as defined below.
   * 
   *   An "Application" is any work that makes use of an interface provided
   * by the Library, but which is not otherwise based on the Library.
   * Defining a subclass of a class defined by the Library is deemed a mode
   * of using an interface provided by the Library.
   * 
   *   A "Combined Work" is a work produced by combining or linking an
   * Application with the Library.  The particular version of the Library
   * with which the Combined Work was made is also called the "Linked
   * Version".
   * 
   *   The "Minimal Corresponding Source" for a Combined Work means the
   * Corresponding Source for the Combined Work, excluding any source code
   * for portions of the Combined Work that, considered in isolation, are
   * based on the Application, and not on the Linked Version.
   * 
   *   The "Corresponding Application Code" for a Combined Work means the
   * object code and/or source code for the Application, including any data
   * and utility programs needed for reproducing the Combined Work from the
   * Application, but excluding the System Libraries of the Combined Work.
   * 
   *   1. Exception to Section 3 of the GNU GPL.
   * 
   *   You may convey a covered work under sections 3 and 4 of this License
   * without being bound by section 3 of the GNU GPL.
   * 
   *   2. Conveying Modified Versions.
   * 
   *   If you modify a copy of the Library, and, in your modifications, a
   * facility refers to a function or data to be supplied by an Application
   * that uses the facility (other than as an argument passed when the
   * facility is invoked), then you may convey a copy of the modified
   * version:
   * 
   *    a) under this License, provided that you make a good faith effort to
   *    ensure that, in the event an Application does not supply the
   *    function or data, the facility still operates, and performs
   *    whatever part of its purpose remains meaningful, or
   * 
   *    b) under the GNU GPL, with none of the additional permissions of
   *    this License applicable to that copy.
   * 
   *   3. Object Code Incorporating Material from Library Header Files.
   * 
   *   The object code form of an Application may incorporate material from
   * a header file that is part of the Library.  You may convey such object
   * code under terms of your choice, provided that, if the incorporated
   * material is not limited to numerical parameters, data structure
   * layouts and accessors, or small macros, inline functions and templates
   * (ten or fewer lines in length), you do both of the following:
   * 
   *    a) Give prominent notice with each copy of the object code that the
   *    Library is used in it and that the Library and its use are
   *    covered by this License.
   * 
   *    b) Accompany the object code with a copy of the GNU GPL and this license
   *    document.
   * 
   *   4. Combined Works.
   * 
   *   You may convey a Combined Work under terms of your choice that,
   * taken together, effectively do not restrict modification of the
   * portions of the Library contained in the Combined Work and reverse
   * engineering for debugging such modifications, if you also do each of
   * the following:
   * 
   *    a) Give prominent notice with each copy of the Combined Work that
   *    the Library is used in it and that the Library and its use are
   *    covered by this License.
   * 
   *    b) Accompany the Combined Work with a copy of the GNU GPL and this license
   *    document.
   * 
   *    c) For a Combined Work that displays copyright notices during
   *    execution, include the copyright notice for the Library among
   *    these notices, as well as a reference directing the user to the
   *    copies of the GNU GPL and this license document.
   * 
   *    d) Do one of the following:
   * 
   *        0) Convey the Minimal Corresponding Source under the terms of this
   *        License, and the Corresponding Application Code in a form
   *        suitable for, and under terms that permit, the user to
   *        recombine or relink the Application with a modified version of
   *        the Linked Version to produce a modified Combined Work, in the
   *        manner specified by section 6 of the GNU GPL for conveying
   *        Corresponding Source.
   * 
   *        1) Use a suitable shared library mechanism for linking with the
   *        Library.  A suitable mechanism is one that (a) uses at run time
   *        a copy of the Library already present on the user's computer
   *        system, and (b) will operate properly with a modified version
   *        of the Library that is interface-compatible with the Linked
   *        Version. 
   * 
   *    e) Provide Installation Information, but only if you would otherwise
   *    be required to provide such information under section 6 of the
   *    GNU GPL, and only to the extent that such information is
   *    necessary to install and execute a modified version of the
   *    Combined Work produced by recombining or relinking the
   *    Application with a modified version of the Linked Version. (If
   *    you use option 4d0, the Installation Information must accompany
   *    the Minimal Corresponding Source and Corresponding Application
   *    Code. If you use option 4d1, you must provide the Installation
   *    Information in the manner specified by section 6 of the GNU GPL
   *    for conveying Corresponding Source.)
   * 
   *   5. Combined Libraries.
   * 
   *   You may place library facilities that are a work based on the
   * Library side by side in a single library together with other library
   * facilities that are not Applications and are not covered by this
   * License, and convey such a combined library under terms of your
   * choice, if you do both of the following:
   * 
   *    a) Accompany the combined library with a copy of the same work based
   *    on the Library, uncombined with any other library facilities,
   *    conveyed under the terms of this License.
   * 
   *    b) Give prominent notice with the combined library that part of it
   *    is a work based on the Library, and explaining where to find the
   *    accompanying uncombined form of the same work.
   * 
   *   6. Revised Versions of the GNU Lesser General Public License.
   * 
   *   The Free Software Foundation may publish revised and/or new versions
   * of the GNU Lesser General Public License from time to time. Such new
   * versions will be similar in spirit to the present version, but may
   * differ in detail to address new problems or concerns.
   * 
   *   Each version is given a distinguishing version number. If the
   * Library as you received it specifies that a certain numbered version
   * of the GNU Lesser General Public License "or any later version"
   * applies to it, you have the option of following the terms and
   * conditions either of that published version or of any later version
   * published by the Free Software Foundation. If the Library as you
   * received it does not specify a version number of the GNU Lesser
   * General Public License, you may choose any version of the GNU Lesser
   * General Public License ever published by the Free Software Foundation.
   * 
   *   If the Library as you received it specifies that a proxy can decide
   * whether future versions of the GNU Lesser General Public License shall
   * apply, that proxy's public statement of acceptance of any version is
   * permanent authorization for you to choose that version for the
   * Library.
   *
   * <BR><BR>
   *
   */

  /*! 
   *  Data 1D array structure
   */
  typedef struct {
    int dim;                  /*!< Dimension */
    int size_struct;          /*!< Size of the structure  */
    int bps;                  /*!< Number of bits per sample  */
    int spp;                  /*!< Number of samples per pixel */
    int ref2d;                /*!< Value will be != -1 if the array correspond to a specific FLGR_Data2D's row */
    FLGR_Type type;           /*!< Type of a sample*/
    FLGR_Shape shape;         /*!< Shape if applicable*/
    int length;               /*!< length of the array */
    void *array;              /*!< Virtual Start of row elements */
    void *array_phantom;      /*!< Physical Start of row elements (array = array_phantom+32)*/
  }FLGR_Data1D;

  /*! 
   *  Data2D array structure
   */
  typedef struct {
    int dim;                  /*!< Dimension */
    int size_struct;          /*!< Size of the structure  */
    int link_overlap;         /*!< Set sup>=0 if rows are linked to another FLGR_Data2D, else set to -1 */
    int link_position;        /*!< which part of the image is used as link from the original image, else -1*/
    int link_number;          /*!< number ofy position where the link starts in the source image, else -1 */
    int bps;                  /*!< Number of bits per sample  */
    int spp;                  /*!< Number of samples per pixel */
    int ref3d;                /*!< Value will be != -1 if the array correspond to a specific FLGR_Data3D's plan */
    FLGR_Type type;           /*!< Type of a sample */
    FLGR_Shape shape;         /*!< Shape if applicable drawed in the matrix*/
    FLGR_Connexity connexity; /*!< Connexity if applicable of the matrix*/
    int size_y;               /*!< Number of line */
    int size_x;               /*!< Number of column */
    FLGR_Data1D **row;        /*!< FLGR_Data1D row pointer array*/
    void **array;             /*!< fast access to 2d array values */
  }FLGR_Data2D;


  EXPORT_LIB int flgr_normalize_coordinate(int axis_coord, int axis_length);




  /****************************************************************
   ********************* 1D Functions *****************************
   ****************************************************************/
  EXPORT_LIB FLGR_Data1D *flgr1d_create(int length, int spp, FLGR_Type type, FLGR_Shape shape);

  EXPORT_LIB FLGR_Data1D *flgr1d_create_from(FLGR_Data1D *dat);

  EXPORT_LIB FLGR_Data1D *flgr1d_create_signal(int length, int spp, FLGR_Type type);

  EXPORT_LIB FLGR_Data1D *flgr1d_create_signal_from(FLGR_Data1D *dat);

  EXPORT_LIB FLGR_Data1D *flgr1d_create_neighborhood(int length, int spp, FLGR_Type type, FLGR_Shape shape);

  EXPORT_LIB FLGR_Data1D *flgr1d_create_neighborhood_from(FLGR_Data1D *dat);

  EXPORT_LIB FLGR_Data1D *flgr1d_create_fgBIT(int length, int spp, FLGR_Shape shape);

  EXPORT_LIB FLGR_Data1D *flgr1d_create_fgUINT8(int length, int spp, FLGR_Shape shape);

  EXPORT_LIB FLGR_Data1D *flgr1d_create_fgUINT16(int length, int spp, FLGR_Shape shape);

  EXPORT_LIB FLGR_Data1D *flgr1d_create_fgUINT32(int length, int spp, FLGR_Shape shape);

  EXPORT_LIB FLGR_Data1D *flgr1d_create_fgINT8(int length, int spp, FLGR_Shape shape);

  EXPORT_LIB FLGR_Data1D *flgr1d_create_fgINT16(int length, int spp, FLGR_Shape shape);

  EXPORT_LIB FLGR_Data1D *flgr1d_create_fgINT32(int length, int spp, FLGR_Shape shape);

  EXPORT_LIB FLGR_Data1D *flgr1d_create_fgFLOAT32(int length, int spp, FLGR_Shape shape);

  EXPORT_LIB FLGR_Data1D *flgr1d_create_fgFLOAT64(int length, int spp, FLGR_Shape shape);

  EXPORT_LIB FLGR_Ret flgr1d_destroy(FLGR_Data1D *dat);

  EXPORT_LIB FLGR_Ret flgr1d_is_data_same_length(FLGR_Data1D *dat1, FLGR_Data1D *dat2);

  EXPORT_LIB FLGR_Ret flgr1d_is_data_same_type(FLGR_Data1D *dat1, FLGR_Data1D *dat2);

  EXPORT_LIB FLGR_Ret flgr1d_is_data_same_spp(FLGR_Data1D *dat1, FLGR_Data1D *dat2);

  EXPORT_LIB FLGR_Ret flgr1d_is_data_same_attributes(FLGR_Data1D *data1, FLGR_Data1D *data2, 
						     const char *callingFunction);

  EXPORT_LIB int flgr1d_data_is_type_fgBIT(FLGR_Data1D *data);

  EXPORT_LIB int flgr1d_data_is_type_fgUINT8(FLGR_Data1D *data);

  EXPORT_LIB int flgr1d_data_is_type_fgUINT16(FLGR_Data1D *data);

  EXPORT_LIB int flgr1d_data_is_type_fgUINT32(FLGR_Data1D *data);

  EXPORT_LIB int flgr1d_data_is_type_fgINT8(FLGR_Data1D *data);

  EXPORT_LIB int flgr1d_data_is_type_fgINT16(FLGR_Data1D *data);

  EXPORT_LIB int flgr1d_data_is_type_fgINT32(FLGR_Data1D *data);

  EXPORT_LIB int flgr1d_data_is_type_fgFLOAT32(FLGR_Data1D *data);

  EXPORT_LIB int flgr1d_data_is_type_fgFLOAT64(FLGR_Data1D *data);

  EXPORT_LIB int flgr1d_data_is_type(FLGR_Data1D *data, FLGR_Type type);

  EXPORT_LIB int flgr1d_data_is_shape(FLGR_Data1D *data, FLGR_Shape shape);

  EXPORT_LIB FLGR_Ret flgr1d_data_set_shape(FLGR_Data1D *dat, FLGR_Shape shape);

  EXPORT_LIB FLGR_Ret flgr1d_clear_all(FLGR_Data1D *data);

  /****************************************************************
   ********************* 2D Functions *****************************
   ****************************************************************/

  EXPORT_LIB FLGR_Data2D *flgr2d_create(int size_y, int size_x, int spp, FLGR_Type type, FLGR_Shape shape, FLGR_Connexity connexity);

  EXPORT_LIB FLGR_Data2D *flgr2d_create_from(FLGR_Data2D *datsrc);

  EXPORT_LIB FLGR_Data2D *flgr2d_create_pixmap(int size_y, int size_x, int spp, FLGR_Type type);

  EXPORT_LIB FLGR_Data2D *flgr2d_create_pixmap_from(FLGR_Data2D *imgsrc);

  EXPORT_LIB FLGR_Data2D *flgr2d_create_neighborhood(int size_y, int size_x, int spp, FLGR_Type type, 
						 FLGR_Shape shape, FLGR_Connexity connexity);
 
  EXPORT_LIB FLGR_Data2D *flgr2d_create_neighborhood_from(FLGR_Data2D *nhbsrc);

  EXPORT_LIB FLGR_Data2D *flgr2d_create_neighborhood_from_connexity( int spp, FLGR_Type type, FLGR_Connexity connexity);

  EXPORT_LIB FLGR_Data2D *flgr2d_create_fgBIT(int size_y, int size_x, int spp, FLGR_Shape shape, FLGR_Connexity connexity);

  EXPORT_LIB FLGR_Data2D *flgr2d_create_fgUINT8(int size_y, int size_x, int spp, FLGR_Shape shape, FLGR_Connexity connexity);

  EXPORT_LIB FLGR_Data2D *flgr2d_create_fgUINT16(int size_y, int size_x, int spp, FLGR_Shape shape, FLGR_Connexity connexity);

  EXPORT_LIB FLGR_Data2D *flgr2d_create_fgUINT32(int size_y, int size_x, int spp, FLGR_Shape shape, FLGR_Connexity connexity);

  EXPORT_LIB FLGR_Data2D *flgr2d_create_fgINT8(int size_y, int size_x, int spp, FLGR_Shape shape, FLGR_Connexity connexity);

  EXPORT_LIB FLGR_Data2D *flgr2d_create_fgINT16(int size_y, int size_x, int spp, FLGR_Shape shape, FLGR_Connexity connexity);

  EXPORT_LIB FLGR_Data2D *flgr2d_create_fgINT32(int size_y, int size_x, int spp, FLGR_Shape shape, FLGR_Connexity connexity);

  EXPORT_LIB FLGR_Data2D *flgr2d_create_fgFLOAT32(int size_y, int size_x, int spp, FLGR_Shape shape, FLGR_Connexity connexity);

  EXPORT_LIB FLGR_Data2D *flgr2d_create_fgFLOAT64(int size_y, int size_x, int spp, FLGR_Shape shape, FLGR_Connexity connexity);

  EXPORT_LIB FLGR_Ret flgr2d_destroy(FLGR_Data2D *dat);

  EXPORT_LIB FLGR_Data2D *flgr2d_create_pixmap_link(FLGR_Data2D *datain, int partsNumber, int partIndex, int overlapSize);

  EXPORT_LIB FLGR_Ret flgr2d_destroy_link(FLGR_Data2D *dat);



  EXPORT_LIB FLGR_Ret flgr2d_is_data_same_attributes(FLGR_Data2D *dat1, FLGR_Data2D *dat2, 
						     const char *callingFunction);

  EXPORT_LIB FLGR_Ret flgr2d_is_data_same_type(FLGR_Data2D *dat1, FLGR_Data2D *dat2);

  EXPORT_LIB FLGR_Ret flgr2d_is_data_same_spp(FLGR_Data2D *dat1, FLGR_Data2D *dat2);

  EXPORT_LIB FLGR_Ret flgr2d_is_data_same_size(FLGR_Data2D *dat1, FLGR_Data2D *dat2);

  EXPORT_LIB int flgr2d_data_is_type_fgBIT(FLGR_Data2D *data);

  EXPORT_LIB int flgr2d_data_is_type_fgUINT8(FLGR_Data2D *data);

  EXPORT_LIB int flgr2d_data_is_type_fgUINT16(FLGR_Data2D *data);

  EXPORT_LIB int flgr2d_data_is_type_fgUINT32(FLGR_Data2D *data);

  EXPORT_LIB int flgr2d_data_is_type_fgINT8(FLGR_Data2D *data);

  EXPORT_LIB int flgr2d_data_is_type_fgINT16(FLGR_Data2D *data);

  EXPORT_LIB int flgr2d_data_is_type_fgINT32(FLGR_Data2D *data);

  EXPORT_LIB int flgr2d_data_is_type_fgFLOAT32(FLGR_Data2D *data);

  EXPORT_LIB int flgr2d_data_is_type_fgFLOAT64(FLGR_Data2D *data);

  EXPORT_LIB int flgr2d_data_is_type(FLGR_Data2D *data, FLGR_Type type);

  EXPORT_LIB int flgr2d_data_is_shape(FLGR_Data2D *data, FLGR_Shape shape);

  EXPORT_LIB int flgr2d_data_is_connexity(FLGR_Data2D *data, FLGR_Connexity connexity);

  EXPORT_LIB FLGR_Ret flgr2d_data_set_shape(FLGR_Data2D *dat, FLGR_Shape shape);

  EXPORT_LIB FLGR_Ret flgr2d_data_set_connexity(FLGR_Data2D *dat, FLGR_Connexity connexity);

  EXPORT_LIB FLGR_Ret flgr2d_clear_all(FLGR_Data2D *data);


#endif

#ifdef __cplusplus
}
#endif
