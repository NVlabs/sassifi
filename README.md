# SASSIFI: An Architecture-level Fault Injection Tool for GPU Application Resilience Evaluations

SASSIFI provides an automated framework to perform error injection campaigns for GPU application resilience evaluation.  SASSIFI builds on top of [SASSI](https://github.com/NVlabs/SASSI), which is a low-level assembly-language instrumentation tool that provides the ability to instrument instructions in the low-level GPU assembly language (SASS).  SASSIFI can be used to perform many types of resilience evaluation studies. Our [ISPASS 2017 paper](https://research.nvidia.com/publication/2017-04_SASSIFI%3A-An-Architecture-level) explains the tool in detail and presents a few case studies. 

Please see the sassifi-user-guide.pdf file in the documentation directory for details on how to use the use SASSIFI (some basic information can be found below). Portions of the documentation are from the IPASS 2017 paper, which is copyrighted by IEEE.

# Modes of operations and supported error models

SASSIFI can be operated in three modes based on the type of study one wants to perform:

* IOV (Instruction Output Value) mode: In this mode, we inject errors into the destination register values by instrumenting instructions after they are executed. 
* IOA (Instruction Output Address) mode: In this mode, we inject errors into destination register indices and store addresses. 
* RF (Register File) mode: In this mode, we inject bit-flips in the RF, randomly spread across time and space (among allocated registers). 

## Where can SASSIFI inject errors?

For the IOA mode, SASSIFI can inject errors in the outputs of randomly selected instructions. SASSIFI allows us to select different types of instructions to study how error in them will propagate to the application output. As of now (3/10/2017), SASSIFI supports selecting the following instruction groups. 
 * Instructions that write to general purpose registers (GPR) 
 * Instructions that write to condition code (CC) 
 * Instructions that write to a predicate register (PR) 
 * Store instruction (ST)
 * Integer add and multiply instructions (IADD-IMAD-OP)
 * Single precision floating point add and multiply instructions (FADD-FMUL-OP)
 * Double precision floating point add and multiply instructions (DFADD-DFMUL-OP)
 * Integer fused multiply and add (MAD) instructions (MAD-OP)
 * Single precision floating point fused multiply and add (FMA) instructions (FMA-OP)
 * Double precision floating point fused multiply and add (DFMA) instructions (DFMA-OP)
 * Instructions that compare source registers and set a predicate register (SETP-OP)
 * Loads from shared memory (LDS-OP)
 * Load instructions, excluding LDS instructions (LD-OP)

SASSIFI can be extended to include custom instruction groups (see sassifi-user-guide.pdf for more details).

In the IOA mode, SASSIFI supports selecting the following two instruction groups -- GPR and ST. At these instructions, SASSIFI injects errors into the address (register index or store address) based on the following defined bit-flip models. 

For the RF mode (injections to measure RF AVF), SASSIFI selects a dynamic instruction randomly from a program and injects an error in a randomly selected register among the allocated registers.  Results obtained from these injections will quantify the probability with which a particle strike in an allocated register can manifest in the application output.  These results need to be further derated by the fraction of physical registers that are unallocated to obtain AVF of the register file for a specific device (instructions to obtain the derating factor are provided in sassifi-user-guide.pdf). 

##  What errors can SASSIFI inject?

For the IOV mode, SASSIFI can inject the error in a destination register based on the different Bit Flip Models (BFM).  In the current release, the following BFMs are implemented. 

 * Single bit-flip: one bit-flip in one register in one thread
 * Double bit-flip: bit-flips in two adjacent bits in one register in one thread
 * Random value: random value in one register in one thread
 * Zero value: zero out the value of one register in one thread
 * Warp wide single bit-flip: one bit-flip in one register in all the threads in a warp
 * Warp wide double bit-flip: bit-flips in two adjacent bits in one register in all the threads in a warp
 * Warp wide random value: random value in one register in all the threads in a warp
 * Warp wide zero value: zero out the value of one register in all the threads in a warp

In the current implementation, we can only inject single bit-flip in one register in one thread (first bit-flip model) for the CC and PR injections. For the SETP-OP instruction group, we can inject only single bit-flip and warp wide single bit-flip (first and fifth bit-flip models, respectively).  

For the IOA and RF modes, SASSIFI considers the following two bit-flip models.  
 * Single bit-flip 
 * Double bit-flip 

These BFMs can be extended to include different bit-flip pattern. To add a new bit-flip model err\_injector.h and injector.cu files in err\_injector directory and common\_params.py and specific\_params.py files in the scripts directory need to be modified. 

# Getting started

## Prerequisites

* A Linux-based system with an x86 64-bit host, a Fermi-, Kepler-, or Maxwell-based GPU. SASSIFI has been tested on Ubuntu (12) and CentOS (6).

* Python 2.7 is needed to run the scripts provided in the package. These scripts are used to generate injection sites, launch injection campaign, and parse the results.	

  * The [lockfile](https://pypi.python.org/pypi/lockfile) module is needed to run the injection jobs in parallel either on a multi-gpu system or a cluster of nodes with shared file-system. This module is not needed if you want to run injection jobs sequentially on a single-gpu system. 

  * (Optional) The final results can be parsed into an xlsx file using the [xlsxwriter](http://xlsxwriter.readthedocs.io/getting_started.html) module.  The results will be parsed into multiple text files, if you do not have xlswriter. These can be copied into an excel file to plot and visualize the results.

* [SASSI](https://github.com/NVlabs/SASSI): Please follow the steps provided in the SASSI documentation to install SASSI.  SASSIFI is tested using the latest commit (5523d984ad047a272297c1a3ff8c63f55c0ad026).  SASSIFI provides code  that needs to be compiled using the SASSI framework.  This code includes SASSI handlers that execute code before and after instructions for profiling and error injections. 

## Setting up and running SASSIFI

Follow these steps to setup and run SASSIFI. We provide a sample script (test.sh) that automates several of these steps.

1. Set the following environment variables:
 * SASSIFI\_HOME: Path to the SASSIFI package (e.g., /home/username/sassifi\_package/)
 * SASSI\_SRC: Path to the SASSI source package (e.g., /home/username/sassi/)
 * INST\_LIB\_DIR: Path to the SASSI libraries (e.g., $SASSI\_SRC/instlibs/lib/)
 * CCDIR: Path to the gcc version 4.8.4 or newer (e.g., /usr/local/gcc-4.8.4/)
 * CUDA\_BASE\_DIR: Path to SASSI installation (e.g., /usr/local/sassi7/)
 * LD\_LIBRARY\_PATH should include the cuda libraries (e.g., $CUDA\_BASE\_DIR/lib64/ and CUDA\_BASE\_DIR/extras/CUPTI/lib64/)
 * Ensure that the GENCODE variable is correctly set for the target GPU in $SASSI\_SRC/instlibs/env.mk and application makefiles (e.g., $SASSIFI\_HOME/suites/example/simple\_add/Makefile). 

2. Copy the SASSI Fault Injection (SASSIFI) handlers into the SASSI package:
 * We provide err\_injector/copy\_handler.sh script to perform this step. Simply run it from any directory. This script creates a new directory named err\_injector in the SASSI\_SRC/instlibs/src directory and creates soft-links for the files provided in the err\_injector directory to avoid keeping multiple copies of the SASSI handler files.

3. Compile the SASSIFI handlers:
 * Simply type `make` in $SASSI\_SRC/instlibs/src/err\_injector.  This should create four libraries. The first one is for profiling the application. The remaining three are for injecting errors during an application run (one each for the three injection modes).

4. Prepare applications:

 1. Record fault-free outputs: Record golden output file (as golden.txt) and golden stdout (as golden\_stdout.txt) and golden stderr (as golden\_stderr.txt) in the workload directory (e.g., $SASSIFI\_HOME/suites/example/simple\_add/).

 2. Create application-specific scripts: Create sassifi\_run.sh and sdc\_check.sh scripts in run/ directory.  These are workload specific and have to be manually created. Instead of using absolute paths, please use environment variables for paths such as BIN\_DIR, APP\_DIR, DATA\_SET\_DIR, and RUN\_SCRIPT\_DIR. These variables are set by run\_one\_injection.py script before launching error injections. See the bash scripts in the run/example/simple\_add/run/ directory for examples. You can also add an application specific check here. 

 3. Prepare applications to compile with the SASSIFI handlers: This might require some work. Follow instructions in the SASSI documentation on how to compile your application with a SASSI handler.
  * Tip: Prepare them such that you can type "make OPTION=profiler" to generate binaries to do the profiling step (step 4) and "make OPTION=inst\_value\_injector"  or "make OPTION=inst\_address\_injector" or "make OPTION=rf\_injector" to generate binaries for error injection campaigns for the three injection modes. See makefile in suites/example/simple\_add/ for an example. This makefile installs different versions of the binaries to $SASSIFI\_HOME/bin/$OPTIONS/ directories. 

5. Profile the application: Compile the application with "OPTION=profiler" and run it once with the same inputs that is specified in the sassifi\_run.sh script. A new file named sassifi-inst-counts.txt will be generated in the directory where the application was run. This file contains the instruction counts for all the instruction groups defined in err\_injector/error\_injector.h and all the opcodes defined in sassi-opcodes.h for all the CUDA kernels. One line is created per dynamic kernel invocation and the format in which the data is printed is shown in the first line in the sassifi-inst-counts.txt file. 

6. Build the applications for error injection runs: Simply run "make OPTION=inst\_value\_injector", "make OPTION=inst\_address\_injector" and/or "make OPTION=rf\_injector"

7. Generate injection sites:

 1. Ensure that the parameters are set correctly in specific\_params.py and common\_params.py files.  Some of the parameters that need user attention are: 

	* Setting maximum number of error injections to perform per instruction group and bit-flip model combination. See NUM\_INJECTION and THRESHOLD\_JOBS in specific\_params.py file. 

	* Selecting instruction groups and bit-flip models. See rf\_bfm\_list,  inst\_value\_igid\_bfm\_map, and inst\_address\_igid\_bfm\_map in specific\_params.py for the list of supported instruction groups (IGIDs) and bit-flip models (BFMs). Simply uncomment the lines to include the IGID and the associated BFMs. User can also select only a subset of the supported BFMs per IGID for targeted error injection studies.

	* Listing the applications, benchmark suite name, application binary file name, and the expected runtime on the system where the injection job will be run. See the apps dictionary in specific\_params.py for an example. The dictionary and the strings defined here are used by other scripts to identify the directory structure in the suites directory and the application binary name.  The expected runtime defined here is used later to determine when to timeout injection runs (based on the TIMEOUT\_THRESHOLD defined in common\_params.py).

	* Setting paths for the suites, logs, bin, and run directories if the user decides to use a different directory structure. If the directory structure for the new benchmark suite that you plan to use is different, please update the app\_dir[app] and app\_data\_dir[app] variables accordingly. 

	* Setting the number of allocated registers per static kernel per application. When an application is compiled using `-Xptxas -v` flags, the number of registers allocated for each static kernel in the application are printed on the standard error (stderr).  User needs to parse the stderr and update the num\_regs dictionary in the specific\_params.py file. Obtain the number of allocated registers without SASSI instrumentation.  If num\_regs dictionary is incorrect (missing/extra kernel names, fewer/more registers per kernel), then the results will also be incorrect because the number of error injections are chosen based on num\_regs.  We provide the process\_kernel\_regcount.py script that parses the stderr from an input file and creates a dictionary per application which is stored in a pickle file. This pickle file can be loaded directly by the specific\_params.py (see set\_num\_regs() for an example). We process the stderr generated by compiling the simple\_add program using this script in test.sh.  The num\_regs dictionary is needed for the RF mode injections. If you do not plan to perform RF mode injections, you can ignore this part.  

 2. Run generate\_injection\_list.py script to generate a file that contains what errors to inject. Instructions are selected randomly for across the entire application for the RF mode and across the instructions from the specified instruction group in the IOV and IOA modes. See sassi-user-guide.pdf for details about what information is collected per injection and how it is used by the next step. 

8. Run injections: Run the run\_injections.py script to launch the error injection campaign.  This script will run one injection after the other in the standalone mode.  If you plan to run multiple jobs in parallel, special care must be taken to ensure that the output file is not clobbered. As of now, we support running multiple jobs on a multi-GPU system. Please see sassi-user-guide.pdf for more details. 

 * Tip: Perform a few dummy injections before proceeding with full injection campaign. Go to step 3 and look for DUMMY\_INJECTION flag in the makefile. Setting this flag will allow you to go through most of the SASSI handler code, but skip the error injection. This is to ensure that you are not seeing crashes/SDCs that you should not see.

9. Parse results: Use the parse\_results.py script to parse the results. This script generates an excel workbook with three sheets, if the xlsxwriter python module is found in the system. If not, three text files are created. The first sheet/text file shows the fraction of executed instructions for different instruction groups and opcodes. The second sheet/text file shows the outcomes of the error injections.  Refer to the documentation to see how we categorize error outcomes.  The third sheet/text file shows the average runtime for the injection runs for different applications, instruction groups, and bit-flip models. Based on how you want to visualize the results, you may want to modify the script or write your own. 

In the current setup, steps 1, 2, 3, 4.2, 4.3, and 7.1 have to be done manually.  Once this is done, the remaining steps can be automated and we provide an example script (test.sh) to run these steps using a single command (`./test.sh` from SASSIFI\_HOME). 

