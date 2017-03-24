/* Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions
* are met:
*  * Redistributions of source code must retain the above copyright
*    notice, this list of conditions and the following disclaimer.
*  * Redistributions in binary form must reproduce the above copyright
*    notice, this list of conditions and the following disclaimer in the
*    documentation and/or other materials provided with the distribution.
*  * Neither the name of NVIDIA CORPORATION nor the names of its
*    contributors may be used to endorse or promote products derived
*    from this software without specific prior written permission.
*
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
* EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
* IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
* PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
* CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
* EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
* PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
* PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
* OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
* (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
* OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include <stdio.h>
#include <stdint.h>
#include <unistd.h>
#include <assert.h>
#include <string>
#include <map>
#include <fstream>
#include <sys/time.h>
#include <cupti.h>

#include <sassi/sassi-core.hpp>
#include <sassi/sassi-regs.hpp>
#include <sassi/sassi-memory.hpp>
#include "sassi/sassi-opcodes.h"
#include "sassi_intrinsics.h"
#include "sassi_dictionary.hpp"
#include "sassi_lazyallocator.hpp"

#include "error_injector.h"

std::map<std::string, int> knameCount;

std::ofstream ofs; 

#if TIMING
struct timeval start, end;
float mTotalTime = 0;
#endif

// This function will be called before every SASS instruction gets executed 
__device__ void sassi_before_handler(SASSIBeforeParams* bp, SASSIMemoryParams *mp, SASSIRegisterParams *rp) {
#if EMPTY_HANDLER
	return;
#endif

  if (bp->GetInstrWillExecute()) {
		profile_instructions(bp, mp, rp);
	} else {
		profile_will_not_execute_instructions();
	}

}

// This function will be exected before a kernel is launced
static void onKernelEntry(const CUpti_CallbackData *cbInfo) {
	reset_profiling_counters(); // reset profiling counters

#if TIMING 
	gettimeofday(&start, NULL);
#endif
} 

// This function will be exected after the kernel exits 
static void onKernelExit(const CUpti_CallbackData *cbInfo) {

	cudaError_t * error = (cudaError_t*) cbInfo->functionReturnValue; 
	if ( (*error) != cudaSuccess ) {
		printf("Kernel Exit Error: %d", (*error));
	}

	// print per thread counters
	std::string kName = cbInfo->symbolName; // name of kernel
	if (knameCount.find(kName) == knameCount.end()) {
		knameCount[kName] = 0;
	} else {
		knameCount[kName] += 1;				
	}

	char numstr[21]; // enough to hold all numbers up to 64-bits
	sprintf(numstr, "%d", knameCount[kName]); // convert int to string

	if (INJ_DEBUG_LIGHT) {
		printf("%s: count=%d\n", kName.c_str(), knameCount[kName]);
	}

	ofs << kName << ":" << numstr;
	for (int i=0; i<NUM_INST_TYPES; i++) {
		ofs << ":" << injCountersInstType[i] ;
	}

	ofs << ":" << opWillNotExecuteCount; // print the number of operations that will not execute
	for (int i=0; i<SASSI_NUM_OPCODES; i++) {
		ofs << ":" << opCounters[i] ;
	}
	ofs << "\n";

#if TIMING
	gettimeofday(&end, NULL);

	long seconds, useconds;    
	seconds  = end.tv_sec  - start.tv_sec;
	useconds = end.tv_usec - start.tv_usec;
	float mTime = ((seconds) * 1000 + useconds/1000.0);
	printf("\nTime for %s:  %f ms\n", cbInfo->symbolName, mTime);
	mTotalTime += mTime;
#endif
} 

static void sassi_init() 
{
	if (INJ_DEBUG_LIGHT)
		printf("Writing to filename:%s\n", profileFilename.c_str());
	ofs.open(profileFilename.c_str(), std::ofstream::out);

	ofs << get_profile_format();
}

static void sassi_finalize(sassi::lazy_allocator::device_reset_reason reason)
{
	ofs.close();

#if TIMING
	printf("\nTotal kernel time: %f ms\n", mTotalTime);
#endif
}

static sassi::lazy_allocator profilerInit(sassi_init, sassi_finalize, onKernelEntry, onKernelExit); 
