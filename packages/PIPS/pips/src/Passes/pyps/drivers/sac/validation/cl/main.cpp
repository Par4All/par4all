/* ============================================================

Copyright (c) 2009 Advanced Micro Devices, Inc.  All rights reserved.
 
Redistribution and use of this material is permitted under the following 
conditions:
 
Redistributions must retain the above copyright notice and all terms of this 
license.
 
In no event shall anyone redistributing or accessing or using this material 
commence or participate in any arbitration or legal action relating to this 
material against Advanced Micro Devices, Inc. or any copyright holders or 
contributors. The foregoing shall survive any expiration or termination of 
this license or any agreement or access or use related to this material. 

ANY BREACH OF ANY TERM OF THIS LICENSE SHALL RESULT IN THE IMMEDIATE REVOCATION 
OF ALL RIGHTS TO REDISTRIBUTE, ACCESS OR USE THIS MATERIAL.

THIS MATERIAL IS PROVIDED BY ADVANCED MICRO DEVICES, INC. AND ANY COPYRIGHT 
HOLDERS AND CONTRIBUTORS "AS IS" IN ITS CURRENT CONDITION AND WITHOUT ANY 
REPRESENTATIONS, GUARANTEE, OR WARRANTY OF ANY KIND OR IN ANY WAY RELATED TO 
SUPPORT, INDEMNITY, ERROR FREE OR UNINTERRUPTED OPERA TION, OR THAT IT IS FREE 
FROM DEFECTS OR VIRUSES.  ALL OBLIGATIONS ARE HEREBY DISCLAIMED - WHETHER 
EXPRESS, IMPLIED, OR STATUTORY - INCLUDING, BUT NOT LIMITED TO, ANY IMPLIED 
WARRANTIES OF TITLE, MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, 
ACCURACY, COMPLETENESS, OPERABILITY, QUALITY OF SERVICE, OR NON-INFRINGEMENT. 
IN NO EVENT SHALL ADVANCED MICRO DEVICES, INC. OR ANY COPYRIGHT HOLDERS OR 
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, PUNITIVE,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT
OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, REVENUE, DATA, OR PROFITS; OR 
BUSINESS INTERRUPTION) HOWEVER CAUSED OR BASED ON ANY THEORY OF LIABILITY 
ARISING IN ANY WAY RELATED TO THIS MATERIAL, EVEN IF ADVISED OF THE POSSIBILITY 
OF SUCH DAMAGE. THE ENTIRE AND AGGREGATE LIABILITY OF ADVANCED MICRO DEVICES, 
INC. AND ANY COPYRIGHT HOLDERS AND CONTRIBUTORS SHALL NOT EXCEED TEN DOLLARS 
(US $10.00). ANYONE REDISTRIBUTING OR ACCESSING OR USING THIS MATERIAL ACCEPTS 
THIS ALLOCATION OF RISK AND AGREES TO RELEASE ADVANCED MICRO DEVICES, INC. AND 
ANY COPYRIGHT HOLDERS AND CONTRIBUTORS FROM ANY AND ALL LIABILITIES, 
OBLIGATIONS, CLAIMS, OR DEMANDS IN EXCESS OF TEN DOLLARS (US $10.00). THE 
FOREGOING ARE ESSENTIAL TERMS OF THIS LICENSE AND, IF ANY OF THESE TERMS ARE 
CONSTRUED AS UNENFORCEABLE, FAIL IN ESSENTIAL PURPOSE, OR BECOME VOID OR 
DETRIMENTAL TO ADVANCED MICRO DEVICES, INC. OR ANY COPYRIGHT HOLDERS OR 
CONTRIBUTORS FOR ANY REASON, THEN ALL RIGHTS TO REDISTRIBUTE, ACCESS OR USE 
THIS MATERIAL SHALL TERMINATE IMMEDIATELY. MOREOVER, THE FOREGOING SHALL 
SURVIVE ANY EXPIRATION OR TERMINATION OF THIS LICENSE OR ANY AGREEMENT OR 
ACCESS OR USE RELATED TO THIS MATERIAL.

NOTICE IS HEREBY PROVIDED, AND BY REDISTRIBUTING OR ACCESSING OR USING THIS 
MATERIAL SUCH NOTICE IS ACKNOWLEDGED, THAT THIS MATERIAL MAY BE SUBJECT TO 
RESTRICTIONS UNDER THE LAWS AND REGULATIONS OF THE UNITED STATES OR OTHER 
COUNTRIES, WHICH INCLUDE BUT ARE NOT LIMITED TO, U.S. EXPORT CONTROL LAWS SUCH 
AS THE EXPORT ADMINISTRATION REGULATIONS AND NATIONAL SECURITY CONTROLS AS 
DEFINED THEREUNDER, AS WELL AS STATE DEPARTMENT CONTROLS UNDER THE U.S. 
MUNITIONS LIST. THIS MATERIAL MAY NOT BE USED, RELEASED, TRANSFERRED, IMPORTED,
EXPORTED AND/OR RE-EXPORTED IN ANY MANNER PROHIBITED UNDER ANY APPLICABLE LAWS, 
INCLUDING U.S. EXPORT CONTROL LAWS REGARDING SPECIFICALLY DESIGNATED PERSONS, 
COUNTRIES AND NATIONALS OF COUNTRIES SUBJECT TO NATIONAL SECURITY CONTROLS. 
MOREOVER, THE FOREGOING SHALL SURVIVE ANY EXPIRATION OR TERMINATION OF ANY 
LICENSE OR AGREEMENT OR ACCESS OR USE RELATED TO THIS MATERIAL.

NOTICE REGARDING THE U.S. GOVERNMENT AND DOD AGENCIES: This material is 
provided with "RESTRICTED RIGHTS" and/or "LIMITED RIGHTS" as applicable to 
computer software and technical data, respectively. Use, duplication, 
distribution or disclosure by the U.S. Government and/or DOD agencies is 
subject to the full extent of restrictions in all applicable regulations, 
including those found at FAR52.227 and DFARS252.227 et seq. and any successor 
regulations thereof. Use of this material by the U.S. Government and/or DOD 
agencies is acknowledgment of the proprietary rights of any copyright holders 
and contributors, including those of Advanced Micro Devices, Inc., as well as 
the provisions of FAR52.227-14 through 23 regarding privately developed and/or 
commercial computer software.

This license forms the entire agreement regarding the subject matter hereof and 
supersedes all proposals and prior discussions and writings between the parties 
with respect thereto. This license does not affect any ownership, rights, title,
or interest in, or relating to, this material. No terms of this license can be 
modified or waived, and no breach of this license can be excused, unless done 
so in a writing signed by all affected parties. Each term of this license is 
separately enforceable. If any term of this license is determined to be or 
becomes unenforceable or illegal, such term shall be reformed to the minimum 
extent necessary in order for this license to remain in effect in accordance 
with its terms as modified by such reformation. This license shall be governed 
by and construed in accordance with the laws of the State of Texas without 
regard to rules on conflicts of law of any state or jurisdiction or the United 
Nations Convention on the International Sale of Goods. All disputes arising out 
of this license shall be subject to the jurisdiction of the federal and state 
courts in Austin, Texas, and all defenses are hereby waived concerning personal 
jurisdiction and venue of these courts.

============================================================ */

#define __CL_ENABLE_EXCEPTIONS

#include <iostream>
#include <iomanip>
#include <string>
#include <fstream>
#include <iostream>
#include <vector>
#include <math.h>
#include <omp.h>
#include <string.h>
#include <CL/cl.h>
#include <sys/time.h>
#include "Timer.h"
#include "Params.h"

using std::cout;
using std::cerr;
using std::endl;
using std::setw;
using std::string;
using std::ifstream;

/////////////////////////////////////////////////////////////////
// Macros 
/////////////////////////////////////////////////////////////////

#define FREE(ptr, free_val) \
    if (ptr != free_val)    \
    {                       \
        free(ptr);          \
        ptr = free_val;     \
    }

/////////////////////////////////////////////////////////////////
// Globals
/////////////////////////////////////////////////////////////////

struct hostBufferStruct
{
    float * pInput;
    float * pFilter;
    float * pOutputCL;
    float * pOutputCPU;
} hostBuffers;

struct oclBufferStruct
{
    cl_mem  inputCL;
    cl_mem  filterCL;
    cl_mem  outputCL;
} oclBuffers;

struct oclHandleStruct
{
    cl_context              context;
    cl_device_id            *devices;
    cl_command_queue        queue;
    cl_program              program;
    std::vector<cl_kernel>  kernel;
} oclHandles;

struct timerStruct
{
    double dCpuTime;
    double dDeviceTotal;
    double dDeviceKernel;
    CPerfCounter counter;
} timers;

bool bCPUResultsReady = false;

/////////////////////////////////////////////////////////////////
// Host buffers
/////////////////////////////////////////////////////////////////

void InitHostBuffers()
{
    hostBuffers.pInput  = NULL;
    hostBuffers.pOutputCL = NULL;
    hostBuffers.pOutputCPU = NULL;

    /////////////////////////////////////////////////////////////////
    // Allocate and initialize memory used by host 
    /////////////////////////////////////////////////////////////////
    int sizeInBytes = params.nInWidth * params.nInHeight * sizeof(float);
    hostBuffers.pInput = (float *) malloc(sizeInBytes);
    if (!hostBuffers.pInput) 
        throw(string("InitHostBuffers()::Could not allocate memory"));

    int sizeOutBytes = params.nWidth * params.nHeight * sizeof(float);
    hostBuffers.pOutputCL = (float *) malloc(sizeOutBytes);
    if (!hostBuffers.pOutputCL) 
        throw(string("InitHostBuffers()::Could not allocate memory"));
    hostBuffers.pOutputCPU = (float *) malloc(sizeOutBytes);
    if (!hostBuffers.pOutputCPU) 
        throw(string("InitHostBuffers()::Could not allocate memory"));

    int filterSizeBytes = params.nFilterWidth * params.nFilterWidth * sizeof(float);
    hostBuffers.pFilter = (float *) malloc(filterSizeBytes);
    if (!hostBuffers.pFilter) 
        throw(string("InitHostBuffers()::Could not allocate memory"));

    srand(0);
    #pragma omp parallel for num_threads(DEFAULT_NUM_THREADS)
    for (int i = 0; i < params.nInWidth * params.nInHeight; i++)
    {
        hostBuffers.pInput[i] = float(rand());
    }

    double dFilterSum = 0;
    int nFilterSize = params.nFilterWidth*params.nFilterWidth;
    for (int i = 0; i < nFilterSize; i++)
    {
        hostBuffers.pFilter[i] = float(rand());
        dFilterSum += hostBuffers.pFilter[i];
    }
    for (int i = 0; i < nFilterSize; i++)
        hostBuffers.pFilter[i] /= dFilterSum;
}

void ClearBuffer(float * pBuf)
{
    #pragma omp parallel for num_threads(DEFAULT_NUM_THREADS)
    for (int i = 0; i < params.nWidth*params.nHeight; i++)
    {
        pBuf[i] = -999.999f;
    }
}

void ReleaseHostBuffers()
{
    FREE(hostBuffers.pInput, NULL);
    FREE(hostBuffers.pOutputCL, NULL);
    FREE(hostBuffers.pOutputCPU, NULL);
    FREE(hostBuffers.pFilter, NULL);
}

/////////////////////////////////////////////////////////////////
// Utils
/////////////////////////////////////////////////////////////////

/*
 * Converts the contents of a file into a string
 */
string FileToString(const string fileName)
{
    ifstream f(fileName.c_str(), ifstream::in | ifstream::binary);

    try
    {
        size_t size;
        char*  str;
        string s;

        if(f.is_open())
        {
            size_t fileSize;
            f.seekg(0, ifstream::end);
            size = fileSize = f.tellg();
            f.seekg(0, ifstream::beg);

            str = new char[size+1];
            if (!str) throw(string("Could not allocate memory"));

            f.read(str, fileSize);
            f.close();
            str[size] = '\0';
        
            s = str;
            delete [] str;
            return s;
        }
    }
    catch(std::string msg)
    {
        cerr << "Exception caught in FileToString(): " << msg << endl;
        if(f.is_open())
            f.close();
    }
    catch(...)
    {
        cerr << "Exception caught in FileToString()" << endl;
        if(f.is_open())
            f.close();
    }
    string errorMsg = "FileToString()::Error: Unable to open file "
                            + fileName;
    throw(errorMsg);
}

/////////////////////////////////////////////////////////////////
// CL Buffers
/////////////////////////////////////////////////////////////////

void InitCLBuffers()
{
    cl_int resultCL;

    oclBuffers.inputCL = NULL;
    oclBuffers.outputCL = NULL;
    oclBuffers.filterCL = NULL;

    /////////////////////////////////////////////////////////////////
    // Create OpenCL memory buffers
    /////////////////////////////////////////////////////////////////
    oclBuffers.inputCL = clCreateBuffer(oclHandles.context, 
                                    CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                                    sizeof(cl_float) * params.nInWidth*params.nInHeight,
                                    hostBuffers.pInput, 
                                    &resultCL);

    if ((resultCL != CL_SUCCESS) || (oclBuffers.inputCL == NULL))
        throw(string("InitCLBuffers()::Error: clCreateBuffer (oclBuffers.inputCL)"));

    oclBuffers.outputCL = clCreateBuffer(oclHandles.context, 
                                    CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR,
                                    sizeof(cl_float) * params.nWidth*params.nHeight,
                                    hostBuffers.pOutputCL, 
                                    &resultCL);

    if ((resultCL != CL_SUCCESS) || (oclBuffers.outputCL == NULL))
        throw(string("InitCLBuffers()::Error: clCreateBuffer (oclBuffers.outputCL)"));

    oclBuffers.filterCL = clCreateBuffer(oclHandles.context, 
                                    CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                                    sizeof(cl_float) * params.nFilterWidth*params.nFilterWidth,
                                    hostBuffers.pFilter, 
                                    &resultCL);

    if ((resultCL != CL_SUCCESS) || (oclBuffers.filterCL == NULL))
        throw(string("InitCLBuffers()::Error: clCreateBuffer (oclBuffers.filterCL)"));
}

void ReleaseCLBuffers()
{
    bool errorFlag = false;

    if (oclBuffers.inputCL != NULL)
    {
        cl_int resultCL = clReleaseMemObject(oclBuffers.inputCL);
        if (resultCL != CL_SUCCESS)
        {
            cerr << "ReleaseCLBuffers()::Error: In clReleaseMemObject (inputBuffer)" << endl;
            errorFlag = true;
        }
        oclBuffers.inputCL = NULL;
    }

    if (oclBuffers.outputCL != NULL)
    {
        cl_int resultCL = clReleaseMemObject(oclBuffers.outputCL);
        if (resultCL != CL_SUCCESS)
        {
            cerr << "ReleaseCLBuffers()::Error: In clReleaseMemObject (outputBuffer)" << endl;
            errorFlag = true;
        }
        oclBuffers.outputCL = NULL;
    }

    if (oclBuffers.filterCL != NULL)
    {
        cl_int resultCL = clReleaseMemObject(oclBuffers.filterCL);
        if (resultCL != CL_SUCCESS)
        {
            cerr << "ReleaseCLBuffers()::Error: In clReleaseMemObject (filterBuffer)" << endl;
            errorFlag = true;
        }
        oclBuffers.filterCL = NULL;
    }

    if (errorFlag)
        throw(string("ReleaseCLBuffers()::Error encountered."));
}

/////////////////////////////////////////////////////////////////
// Initialize and shutdown CL 
/////////////////////////////////////////////////////////////////

/*
 * Create Context, Device list, Command Queue
 * Load CL file, compile, link CL source 
 * Build program and kernel objects
 */
void InitCL()
{
    cl_int resultCL;
    
    oclHandles.context = NULL;
    oclHandles.devices = NULL;
    oclHandles.queue = NULL;
    oclHandles.program = NULL;

    size_t deviceListSize;

    /////////////////////////////////////////////////////////////////
    // Find the available platforms and select one
    /////////////////////////////////////////////////////////////////
    cl_uint numPlatforms;
    cl_platform_id targetPlatform = NULL;

    resultCL = clGetPlatformIDs(0, NULL, &numPlatforms);
    if (resultCL != CL_SUCCESS)
        throw (string("InitCL()::Error: Getting number of platforms (clGetPlatformIDs)"));
    
    if (!(numPlatforms > 0))
        throw (string("InitCL()::Error: No platforms found (clGetPlatformIDs)"));

    cl_platform_id* allPlatforms = (cl_platform_id*) malloc(numPlatforms * sizeof(cl_platform_id));

    resultCL = clGetPlatformIDs(numPlatforms, allPlatforms, NULL);
    if (resultCL != CL_SUCCESS)
        throw (string("InitCL()::Error: Getting platform ids (clGetPlatformIDs)"));

    /* Select the target platform. Default: first platform */
    targetPlatform = allPlatforms[0];
    for (int i = 0; i < numPlatforms; i++)
    {
        char pbuff[128];
        resultCL = clGetPlatformInfo( allPlatforms[i],
                                        CL_PLATFORM_VENDOR,
                                        sizeof(pbuff),
                                        pbuff,
                                        NULL);
        if (resultCL != CL_SUCCESS)
            throw (string("InitCL()::Error: Getting platform info (clGetPlatformInfo)"));

        if(!strcmp(pbuff, "Advanced Micro Devices, Inc."))
        {
            targetPlatform = allPlatforms[i];
            break;
        }
    }

    FREE(allPlatforms, NULL);

    /////////////////////////////////////////////////////////////////
    // Create an OpenCL context
    /////////////////////////////////////////////////////////////////
    cl_context_properties cprops[3] = { CL_CONTEXT_PLATFORM, (cl_context_properties)targetPlatform, 0 };
    oclHandles.context = clCreateContextFromType(cprops, 
                                                CL_DEVICE_TYPE_CPU, 
                                                NULL, 
                                                NULL, 
                                                &resultCL);

    if ((resultCL != CL_SUCCESS) || (oclHandles.context == NULL))
        throw (string("InitCL()::Error: Creating Context (clCreateContextFromType)"));
        
    /////////////////////////////////////////////////////////////////
    // Detect OpenCL devices
    /////////////////////////////////////////////////////////////////
    /* First, get the size of device list */
    resultCL = clGetContextInfo(oclHandles.context, 
                                CL_CONTEXT_DEVICES, 
                                0, 
                                NULL, 
                                &deviceListSize);

    if (resultCL != CL_SUCCESS)
        throw(string("InitCL()::Error:  Getting Context Info."));
    if (deviceListSize == 0)
        throw(string("InitCL()::Error: No devices found."));

    /* Now, allocate the device list */
    oclHandles.devices = (cl_device_id *)malloc(deviceListSize);

    if (oclHandles.devices == 0)
        throw(string("InitCL()::Error: Could not allocate memory."));

    /* Next, get the device list data */
    resultCL = clGetContextInfo(oclHandles.context, 
                                CL_CONTEXT_DEVICES, 
                                deviceListSize, 
                                oclHandles.devices, 
                                NULL);

    if (resultCL != CL_SUCCESS)
        throw(string("InitCL()::Error: Getting Context Info. (device list, clGetContextInfo)"));

    /////////////////////////////////////////////////////////////////
    // Create an OpenCL command queue
    /////////////////////////////////////////////////////////////////
    oclHandles.queue = clCreateCommandQueue(oclHandles.context, 
                                            oclHandles.devices[0], 
                                            0, 
                                            &resultCL);

    if ((resultCL != CL_SUCCESS) || (oclHandles.queue == NULL))
        throw(string("InitCL()::Creating Command Queue. (clCreateCommandQueue)"));

    /////////////////////////////////////////////////////////////////
    // Load CL file, build CL program object, create CL kernel object
    /////////////////////////////////////////////////////////////////
    std::string  sourceStr = FileToString(params.kernelFile);
    const char * source    = sourceStr.c_str();
    size_t sourceSize[]    = { strlen(source) };

    oclHandles.program = clCreateProgramWithSource(oclHandles.context, 
                                                    1, 
                                                    &source,
                                                    sourceSize,
                                                    &resultCL);

    if ((resultCL != CL_SUCCESS) || (oclHandles.program == NULL))
        throw(string("InitCL()::Error: Loading Binary into cl_program. (clCreateProgramWithBinary)"));    

    resultCL = clBuildProgram(oclHandles.program, 1, oclHandles.devices,
		    "-w", NULL, NULL);

    if ((resultCL != CL_SUCCESS) || (oclHandles.program == NULL))
    {
        cerr << "InitCL()::Error: In clBuildProgram" << endl;

		size_t length;
        resultCL = clGetProgramBuildInfo(oclHandles.program, 
                                        oclHandles.devices[0], 
                                        CL_PROGRAM_BUILD_LOG, 
                                        0, 
                                        NULL, 
                                        &length);
        if(resultCL != CL_SUCCESS) 
            throw(string("InitCL()::Error: Getting Program build info(clGetProgramBuildInfo)"));

		char* buffer = (char*)malloc(length);
        resultCL = clGetProgramBuildInfo(oclHandles.program, 
                                        oclHandles.devices[0], 
                                        CL_PROGRAM_BUILD_LOG, 
                                        length, 
                                        buffer, 
                                        NULL);
        if(resultCL != CL_SUCCESS) 
            throw(string("InitCL()::Error: Getting Program build info(clGetProgramBuildInfo)"));

		cerr << buffer << endl;
        free(buffer);

        throw(string("InitCL()::Error: Building Program (clBuildProgram)"));
    }

    for (int nKernel = 0; nKernel < params.nTotalKernels; nKernel++)
    {
        /* get a kernel object handle for a kernel with the given name */
        cl_kernel kernel = clCreateKernel(oclHandles.program,
                                            params.kernelNames[nKernel].c_str(),
                                            &resultCL);

        if ((resultCL != CL_SUCCESS) || (kernel == NULL))
        {
            string errorMsg = "InitCL()::Error: Creating Kernel (clCreateKernel) \"" + params.kernelNames[nKernel] + "\"";
            throw(errorMsg);
        }

        oclHandles.kernel.push_back(kernel);
    }
}

/*
 * Release OpenCL resources 
 */
void ReleaseCL()
{
    bool errorFlag = false;

    for (int nKernel = 0; nKernel < oclHandles.kernel.size(); nKernel++)
    {
        if (oclHandles.kernel[nKernel] != NULL)
        {
            cl_int resultCL = clReleaseKernel(oclHandles.kernel[nKernel]);
            if (resultCL != CL_SUCCESS)
            {
                cerr << "ReleaseCL()::Error: In clReleaseKernel" << endl;
                errorFlag = true;
            }
            oclHandles.kernel[nKernel] = NULL;
        }
        oclHandles.kernel.clear();
    }

    if (oclHandles.program != NULL)
    {
        cl_int resultCL = clReleaseProgram(oclHandles.program);
        if (resultCL != CL_SUCCESS)
        {
            cerr << "ReleaseCL()::Error: In clReleaseProgram" << endl;
            errorFlag = true;
        }
        oclHandles.program = NULL;
    }

    if (oclHandles.queue != NULL)
    {
        cl_int resultCL = clReleaseCommandQueue(oclHandles.queue);
        if (resultCL != CL_SUCCESS)
        {
            cerr << "ReleaseCL()::Error: In clReleaseCommandQueue" << endl;
            errorFlag = true;
        }
        oclHandles.queue = NULL;
    }

    FREE(oclHandles.devices, NULL);

    if (oclHandles.context != NULL)
    {
        cl_int resultCL = clReleaseContext(oclHandles.context);
        if (resultCL != CL_SUCCESS)
        {
            cerr << "ReleaseCL()::Error: In clReleaseContext" << endl;
            errorFlag = true;
        }
        oclHandles.context = NULL;
    }

    if (errorFlag) throw(string("ReleaseCL()::Error encountered."));
}

/////////////////////////////////////////////////////////////////
// Convolution on CL device 
/////////////////////////////////////////////////////////////////

/*          
 *        Bind variables and CL buffers to kernel arguments 
 */
void SetKernelArgs(int nKernel)
{
    cl_int   resultCL;
    
    /* input image */
    resultCL = clSetKernelArg(oclHandles.kernel[nKernel], 
                                0, 
                                sizeof(cl_mem), 
                                (void *)&oclBuffers.inputCL);

    if (resultCL != CL_SUCCESS) 
        throw(string("SetKernelArgs()::Error: Setting kernel argument."));

    /* filter */
    resultCL = clSetKernelArg(oclHandles.kernel[nKernel], 
                                1, 
                                sizeof(cl_mem), 
                                (void *)&oclBuffers.filterCL);

    if (resultCL != CL_SUCCESS) 
        throw(string("SetKernelArgs()::Error: Setting kernel argument."));

    /* output image */
    resultCL = clSetKernelArg(oclHandles.kernel[nKernel], 
                                2, 
                                sizeof(cl_mem), 
                                (void *)&oclBuffers.outputCL);

    if (resultCL != CL_SUCCESS) 
        throw(string("SetKernelArgs()::Error: Setting kernel argument."));

    /* input image width*/
    resultCL = clSetKernelArg(oclHandles.kernel[nKernel], 
                                3, 
                                sizeof(int), 
                                (void *)&params.nInWidth);

    if (resultCL != CL_SUCCESS) 
        throw(string("SetKernelArgs()::Error: Setting kernel argument."));

    /* filter width*/
    resultCL = clSetKernelArg(oclHandles.kernel[nKernel], 
                                4, 
                                sizeof(int), 
                                (void *)&params.nFilterWidth);

    if (resultCL != CL_SUCCESS)
        throw(string("SetKernelArgs()::Error: Setting kernel argument."));
}

void EnqueueKernel(int nKernel, bool bBlocking = false)
{    
    cl_event events[1];
    cl_int   resultCL;
    
    resultCL = clEnqueueNDRangeKernel(oclHandles.queue,
                                     oclHandles.kernel[nKernel],
                                     params.WORK_DIM,
                                     NULL,
                                     params.globalNDWorkSize,
                                     params.localNDWorkSize,
                                     0, NULL,
                                     &events[0]);

    if (resultCL != CL_SUCCESS)
        throw(string("CallKernel()::Error: Enqueueing kernel onto command queue. (clEnqueueNDRangeKernel)"));

    /* wait for the kernel call to finish execution */
    if (bBlocking)
    {
        resultCL = clWaitForEvents(1, &events[0]);
        if (resultCL != CL_SUCCESS)
            throw(string("CallKernel()::Error: Waiting for kernel run to finish. (clWaitForEvents)"));
    }
}

static int cmpdouble(const void *p1, const void *p2)
{
	return *((const double*)p1) - *((const double*)p2);
}

void RunCL(int nKernel)
{
	if (params.bDeviceKernelTiming)
	{
		cout << "\n********    Kernel " 
			<< params.kernelNames[nKernel] << "    ********" << endl;

		SetKernelArgs(nKernel);

		clFinish(oclHandles.queue);

		double times[params.nIterations];
		timeval timestart,timeend;
		for (int i = 0; i < params.nIterations; i++)
		{
			gettimeofday(&timestart,NULL);
			EnqueueKernel(nKernel);
			clFinish(oclHandles.queue);
			gettimeofday(&timeend,NULL);
			times[i] = (double)(timeend.tv_sec-timestart.tv_sec)*1000.0 + (double)(timeend.tv_usec-timestart.tv_usec)/1000.0;
			cout << "Iteration " << i << " took " << times[i] << " milliseconds." << endl;
		}
		qsort(&times, params.nIterations, sizeof(double), cmpdouble);
		timers.dDeviceKernel = times[params.nIterations/2];
	}
}

/////////////////////////////////////////////////////////////////
// Convolution on CPU 
/////////////////////////////////////////////////////////////////

void Convolve(float * pInput, float * pFilter, float * pOutput,
          const int nInWidth, const int nWidth, const int nHeight,
          const int nFilterWidth, const int nNumThreads)
{
    #pragma omp parallel for num_threads(nNumThreads)
    for (int yOut = 0; yOut < nHeight; yOut++)
    {
        const int yInTopLeft = yOut;

        for (int xOut = 0; xOut < nWidth; xOut++)
        {
            const int xInTopLeft = xOut;

            float sum = 0;
            for (int r = 0; r < nFilterWidth; r++)
            {
                const int idxFtmp = r * nFilterWidth;

                const int yIn = yInTopLeft + r;
                const int idxIntmp = yIn * nInWidth + xInTopLeft;

                for (int c = 0; c < nFilterWidth; c++)
                {
                    const int idxF  = idxFtmp  + c;
                    const int idxIn = idxIntmp + c;    
                    sum += pFilter[idxF]*pInput[idxIn];
                }
            } //for (int r = 0...

            const int idxOut = yOut * nWidth + xOut;
            pOutput[idxOut] = sum;

        } //for (int xOut = 0...
    } //for (int yOut = 0...
}

void RunCPU(int run)
{
    if (params.bCPUTiming)
    {
        cout << "\n********    Starting CPU (" << params.ompThreads[run]
             << "-threads) run    ********" << endl;

        timers.counter.Reset();
        timers.counter.Start();

        for (int i = 0; i < params.nIterations; i++)
            Convolve(hostBuffers.pInput, hostBuffers.pFilter, hostBuffers.pOutputCPU,
                        params.nInWidth,
                        params.nWidth, params.nHeight, 
                        params.nFilterWidth,
                        params.ompThreads[run]);

        timers.counter.Stop();
        timers.dCpuTime = timers.counter.GetElapsedTime()/double(params.nIterations);

        bCPUResultsReady = true;
    }
}

/////////////////////////////////////////////////////////////////
// Verification of CL output 
/////////////////////////////////////////////////////////////////

inline bool Compare(const float val0, const float val1)
{
    const float epsilon = (float)1e-4;
    float diff = (val1 - val0);
    if (fabs(val1) > epsilon)
        diff /= val0;            
    return (fabs(diff) > epsilon);
}

int CompareBuffers(const float* buf0, const float* buf1, const unsigned int size)
{
    int nequal = 0;
    #pragma omp parallel for reduction(+:nequal) num_threads(DEFAULT_NUM_THREADS)
    for (int i = 0; i < size; ++i)
    {
        if ((Compare(buf0[i], buf1[i])))
            nequal++;
    }
    return nequal;    
}

void VerifyOutput()
{
    if (!params.bVerify)
        return;

    cout << "\n********    Starting Verification    ********" << endl;

    if (!bCPUResultsReady)
    {
        Convolve(hostBuffers.pInput, hostBuffers.pFilter, hostBuffers.pOutputCPU,
                    params.nInWidth,
                    params.nWidth, params.nHeight, 
                    params.nFilterWidth, DEFAULT_NUM_THREADS);
        bCPUResultsReady = true;
    }

    int nErrors = CompareBuffers(hostBuffers.pOutputCL, hostBuffers.pOutputCPU, params.nWidth*params.nHeight);

    if (nErrors == 0)
        cout << "\n********    Passed!    ********\n" << endl;
    else
        cout << "\n********    FAILED!!! With " << nErrors << " errors!    ********\n" << endl;
}

/////////////////////////////////////////////////////////////////
// Print info, timing 
/////////////////////////////////////////////////////////////////

void PrintInfo()
{
    cout << endl;
    cout << "Width:          " << params.nWidth << endl;
    cout << "Height:         " << params.nHeight << endl;
    cout << "Filter Size:    " << params.nFilterWidth << " x "
                                << params.nFilterWidth << endl;
    cout << "Iterations:     " << params.nIterations << endl;
    cout << "Workgroup:      " << params.localNDWorkSize[0] << " x " 
                                << params.localNDWorkSize[1] << endl;
    cout << "Verify:         " << (params.bVerify ? "True":"False") << endl;
    cout << "CPU Timing:     " << (params.bCPUTiming ? "True":"False") << endl;
    cout << "Device (kernel):" << (params.bDeviceKernelTiming ? "True":"False") << endl;
    cout << "Testing:        ";

    if (params.bCPUTiming)
        for (int run = 0; run < params.nOmpRuns; run++)
            cout << "CPU (" << params.ompThreads[run] << "-threads) , ";

    for (int nKernel = 0; nKernel < params.nTotalKernels; nKernel++)
        if (params.bDeviceKernelTiming)
            cout << params.kernelNames[nKernel] 
                    << " , ";
    cout << endl << endl;
}

void PrintCPUTime(int run)
{
    if (params.bCPUTiming)
        cout << "CPU (" << params.ompThreads[run] << "-threads): " << timers.dCpuTime << endl;
}

void PrintKernelTime(int nKernel)
{
    if (params.bDeviceKernelTiming)
        cout << "Kernel: " << params.kernelNames[nKernel] 
                << " : " << timers.dDeviceKernel << endl;
    cout << endl;
}

/////////////////////////////////////////////////////////////////
// Main 
/////////////////////////////////////////////////////////////////

int main(int argc, char * argv[])
{
    try
    {
        InitParams(argc, argv);
        PrintInfo();

        InitHostBuffers();
        InitCL();
        InitCLBuffers();

        for (int run = 0; run < params.nOmpRuns; run++)
        {
            ClearBuffer(hostBuffers.pOutputCPU);
            RunCPU(run);
            PrintCPUTime(run);
        }

        for (int nKernel = 0; nKernel < params.nTotalKernels; nKernel++)
        {
            ClearBuffer(hostBuffers.pOutputCL);
            RunCL(nKernel);
            VerifyOutput();
            PrintKernelTime(nKernel);
        }

        ReleaseCLBuffers();
        ReleaseCL();
        ReleaseHostBuffers();
    }
    catch(std::string msg)
    {
        cerr << "Exception caught in main(): " << msg << endl;
        ReleaseCLBuffers();
        ReleaseCL();
        ReleaseHostBuffers();
    }
    catch(...)
    {
        cerr << "Exception caught in main()" << endl;
        ReleaseCLBuffers();
        ReleaseCL();
        ReleaseHostBuffers();
    }

    return 0;
}
