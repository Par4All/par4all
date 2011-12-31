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


#ifndef PARAMS_H_
#define PARAMS_H_

#define DEFAULT_NUM_THREADS 4

struct paramStruct
{
    int nWidth;         //Output image width
    int nHeight;        //Output image height
    int nInWidth;       //Input  image width
    int nInHeight;      //Input  image height
    int nFilterWidth;   //Filter size is nFilterWidth X nFilterWidth
    int nIterations;    //Run timing loop for nIterations

    enum {WORK_DIM=2};
    size_t globalNDWorkSize[WORK_DIM];  //Total number of work items
    size_t localNDWorkSize[WORK_DIM];   //Work items in each work-group

    std::string kernelFile; //File that contains all the kernels
    //Vector to store all kernels
    std::vector<std::string> kernelNames;
    int nTotalKernels;  //kernelNames.size()

    //Test CPU performance with 1,4,8 etc. OpenMP threads
    std::vector<int> ompThreads;
    int nOmpRuns;       //ompThreads.size()

    bool bCPUTiming;            //Time CPU performance
    bool bDeviceKernelTiming;   //Time OpenCL performance

    bool bVerify;       //Crosscheck the OpenCL results with the CPU results
} params;

void Usage(char *name);
void ParseCommandLine(int argc, char* argv[]);

void InitParams(int argc, char* argv[])
{
    params.nWidth = 1024*8;
    params.nHeight = 1024*8;
    params.nFilterWidth = 50;
    params.nIterations = 3;

    params.bCPUTiming = true;
    params.bDeviceKernelTiming = true;
    params.bVerify = true;

    params.kernelFile = "convol.cl";
    ParseCommandLine(argc, argv);

    params.nInWidth = params.nWidth + (params.nFilterWidth-1);
    params.nInHeight = params.nHeight + (params.nFilterWidth-1);

    params.globalNDWorkSize[0] = params.nWidth;
    params.globalNDWorkSize[1] = params.nHeight;
    params.localNDWorkSize[0]  = 32;
    params.localNDWorkSize[1]  = 32;

    params.kernelNames.push_back("Convolve");
    params.nTotalKernels = params.kernelNames.size();

    params.ompThreads.push_back(4);
    //params.ompThreads.push_back(1);
    //params.ompThreads.push_back(8);
    params.nOmpRuns = params.ompThreads.size();

    if (!params.bDeviceKernelTiming)
        params.bVerify = false;
}

void ParseCommandLine(int argc, char* argv[])
{
    for (int i = 1; i < argc; ++i)
    {
        switch (argv[i][1])
        {
        case 'c':
            params.bCPUTiming = false;
            break;
        case 'k':
            params.bDeviceKernelTiming = false;
            break;
        case 'e':
            params.bVerify = false;
            break;
        case 'f':
            if (++i < argc)
            {
                sscanf(argv[i], "%u", &params.nFilterWidth);
            }
            else
            {
                std::cerr << "Could not read argument after option " << argv[i-1] << std::endl;
                Usage(argv[0]);
                throw;
            }
            break;
        case 'i':
            if (++i < argc)
            {
                sscanf(argv[i], "%u", &params.nIterations);
            }
            else
            {
                std::cerr << "Could not read argument after option " << argv[i-1] << std::endl;
                Usage(argv[0]);
                throw;
            }
            break;
        case 'x':
            if (++i < argc)
            {
                sscanf(argv[i], "%u", &params.nWidth);
            }
            else
            {
                std::cerr << "Could not read argument after option " << argv[i-1] << std::endl;
                Usage(argv[0]);
                throw;
            }
            break;
        case 'y':
            if (++i < argc)
            {
                sscanf(argv[i], "%u", &params.nHeight);
            }
            else
            {
                std::cerr << "Could not read argument after option " << argv[i-1] << std::endl;
                Usage(argv[0]);
                throw;
            }
            break;
	case 'o':
            if (++i < argc)
            {
		params.kernelFile = argv[i];
            }
            else
            {
                std::cerr << "Could not read argument after option " << argv[i-1] << std::endl;
                Usage(argv[0]);
                throw;
            }
	    break;
        case 'h':
            Usage(argv[0]);
            exit(1);
        default:
            std::cerr << "Invalid argument " << argv[i] << std::endl;
            Usage(argv[0]);
            throw(std::string("Invalid argument"));
        }
    }
}

void Usage(char *name)
{
    printf("\tUsage: %s [-h] [-c] [-k] [-e] [-f <int>] [-i <int>] [-x <int>] [-y <int>] [-o kernel]\n", name);
    printf("   -h          Print this help menu.\n");
    printf("   -c          Supress CPU timing run.\n");
    printf("   -k          Supress Device (kernel only) timing run.\n");
    printf("   -e          Supress output verification.\n");
    printf("   -f <int>    Sets the filter width.\n");
    printf("   -i <int>    Number of iterations.\n");
    printf("   -x <int>    Sets the image width.\n");
    printf("   -y <int>    Sets the image height.\n");
    printf("   -o <kernel> Sets the OpenCL kernel to use.\n");
}


#endif    //#ifndef PARAMS_H_
