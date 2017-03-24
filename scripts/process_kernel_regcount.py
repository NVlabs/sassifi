#########################################################################
# Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#########################################################################

import sys, cPickle, os

#helper function, if you do not want to use the picke function
def print_dictionary(d):
	print "{"
	for key in d:
		print "\t'" + key + "': " + d[key] + ","
	print "}"

def main():
	if len(sys.argv) != 4:
		print "Usage: python extract_reg_numbers.py <application name> <sm version> <stderr file name>"
		print "Example: python extract_reg_numbers.py simple_add sm_35 stderr"
		print "It is prefered that you run this script from the application directory and store the pickle file there."
		sys.exit(-1)

	app = sys.argv[1]
	sm_version = sys.argv[2]
	input_fname = sys.argv[3]

	f = open(input_fname, "r")

	# dictionary to store the number of allocated registers per static
	kernel_reg = {}

	kname = "" # temporary variable to store the kname
	check_for_regcount = False 
	
	# process the input file created by capturing the stderr while compiling the
	# application using -Xptxas -v options 
	for line in f: # for each line in the file
		if "Compiling entry function" in line:   # if line has this string
			kname = line.split("'")[1].strip() # extract kernel name 
			check_for_regcount = True if sm_version in line else False
		if check_for_regcount and ": Used" in line and "registers, " in line:
			reg_num = line.split(':')[1].split()[1] # extract register number
			if kname not in kernel_reg:
				kernel_reg[kname] = int(reg_num.strip()) # associate the extracted register number with the kernel name
			else:
				print "Warning: " + kname + " exists in the kernel_reg dictionary. Skipping this regcount."
		
	# print the recorded kernel_reg dictionary
	pickle_filename = app+"_kernel_regcount.p"
	cPickle.dump(kernel_reg, open(pickle_filename, "wb"))
	print "Created the pickle file: " + os.getcwd() + "/" + pickle_filename
	print "Load it from the specific_params.py file" 

	#print_dictionary(kernel_reg)

if __name__ == "__main__":
	main()

