Compile the SASSIFI profiler and injector libraries: 
	"make" in ${SASSI_HOME}/instlibs/src/err_injector directory

Please check for these parameters before running SASSIFI:
	in Makefile:
		DUMMY_INJECTION: To test the handler. If this is set to zero then the error
		will not be injected, but the rest of the code will run. This is a great
		flag to use if you want to test/debug the injection functions.  

		INJ_DEBUG_LIGHT, INJ_DEBUG_HEAVY: Use only if you want to debug SASSIFI
		handlers. Set these variables to 1 to see more information during
		injection/profiling runs. Setting either of these to 1 will print debug
		information on stdout, which will cause an SDC. 

	in injector.cu
		injInputFilename is hard-coded to "sassifi-inst-counts.txt"

It's a good idea to compile with DUMMY_INJECTION=1 for the first time and
ensure that error injection jobs return "Masked"

