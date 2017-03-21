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


import os, sys, cPickle
import common_params as p

#########################################################################
# Number of injections per app, per instruction group (IGID), per bit-flip
# model (BFM)
# 
# According to http://www.surveysystem.com/sscalc.htm:
# - Confdience interval at 99% condence level is at most 5% with 644 injections
# - Confdience interval at 95% condence level is at most 5% with 384 injections
# - Confdience interval at 95% condence level is at most 3.1% with 1000 injections
#########################################################################

# Specify the number of injection sites to create before starting the injection
# campaign. This is essentially the maximum number of injections one can run
# per instruction group (IGID) and bit-flip model (BFM).
# 
NUM_INJECTIONS = 644

# Specify how many injections you want to perform per IGID and BFM combination. 
# Only the first THRESHOLD_JOBS will be selected from the generated NUM_INJECTIONS.
THRESHOLD_JOBS = 10 # test

# THRESHOLD_JOBS sould be <= NUM_INJECTIONS
assert THRESHOLD_JOBS <= NUM_INJECTIONS

#########################################################################
# Error model: Plese refer to the SASSIFI user guide to see a description of 
# where and what errors SASSIFI can inject for the two modes (register file 
# and instruction output-level injections). 
# Acronyms: 
#    bfm: bit-flip model
#    igid: instruction group ID
#########################################################################
rf_bfm_list = [p.FLIP_SINGLE_BIT, p.FLIP_TWO_BITS]

# Used for instruction output-level value injection runs 
inst_value_igid_bfm_map = {
	p.GPR: [p.FLIP_SINGLE_BIT, p.FLIP_TWO_BITS],

#  Supported models
# 	p.GPR: [p.FLIP_SINGLE_BIT, p.FLIP_TWO_BITS, p.RANDOM_VALUE, p.ZERO_VALUE, p.WARP_FLIP_SINGLE_BIT, p.WARP_FLIP_TWO_BITS, p.WARP_RANDOM_VALUE, p.WARP_ZERO_VALUE],
# 	p.CC: [p.FLIP_SINGLE_BIT],
# 	p.PR: [p.FLIP_SINGLE_BIT],
# 	p.STORE_OP: [p.FLIP_SINGLE_BIT, p.FLIP_TWO_BITS, p.RANDOM_VALUE, p.ZERO_VALUE, p.WARP_FLIP_SINGLE_BIT, p.WARP_FLIP_TWO_BITS, p.WARP_RANDOM_VALUE, p.WARP_ZERO_VALUE],
# 	p.IADD_IMUL_OP: [p.FLIP_SINGLE_BIT, p.FLIP_TWO_BITS, p.RANDOM_VALUE, p.ZERO_VALUE, p.WARP_FLIP_SINGLE_BIT, p.WARP_FLIP_TWO_BITS, p.WARP_RANDOM_VALUE, p.WARP_ZERO_VALUE],
# 	p.FADD_FMUL_OP: [p.FLIP_SINGLE_BIT, p.FLIP_TWO_BITS, p.RANDOM_VALUE, p.ZERO_VALUE, p.WARP_FLIP_SINGLE_BIT, p.WARP_FLIP_TWO_BITS, p.WARP_RANDOM_VALUE, p.WARP_ZERO_VALUE],
# 	p.DADD_DMUL_OP: [p.FLIP_SINGLE_BIT, p.FLIP_TWO_BITS, p.RANDOM_VALUE, p.ZERO_VALUE, p.WARP_FLIP_SINGLE_BIT, p.WARP_FLIP_TWO_BITS, p.WARP_RANDOM_VALUE, p.WARP_ZERO_VALUE],
# 	p.MAD_OP: [p.FLIP_SINGLE_BIT, p.FLIP_TWO_BITS, p.RANDOM_VALUE, p.ZERO_VALUE, p.WARP_FLIP_SINGLE_BIT, p.WARP_FLIP_TWO_BITS, p.WARP_RANDOM_VALUE, p.WARP_ZERO_VALUE],
# 	p.FFMA_OP: [p.FLIP_SINGLE_BIT, p.FLIP_TWO_BITS, p.RANDOM_VALUE, p.ZERO_VALUE, p.WARP_FLIP_SINGLE_BIT, p.WARP_FLIP_TWO_BITS, p.WARP_RANDOM_VALUE, p.WARP_ZERO_VALUE],
# 	p.DFMA_OP: [p.FLIP_SINGLE_BIT, p.FLIP_TWO_BITS, p.RANDOM_VALUE, p.ZERO_VALUE, p.WARP_FLIP_SINGLE_BIT, p.WARP_FLIP_TWO_BITS, p.WARP_RANDOM_VALUE, p.WARP_ZERO_VALUE],
# 	p.SETP_OP: [p.FLIP_SINGLE_BIT, p.WARP_FLIP_SINGLE_BIT],
# 	p.LDS_OP: [p.FLIP_SINGLE_BIT, p.FLIP_TWO_BITS, p.RANDOM_VALUE, p.ZERO_VALUE, p.WARP_FLIP_SINGLE_BIT, p.WARP_FLIP_TWO_BITS, p.WARP_RANDOM_VALUE, p.WARP_ZERO_VALUE],
# 	p.LD_OP: [p.FLIP_SINGLE_BIT, p.FLIP_TWO_BITS, p.RANDOM_VALUE, p.ZERO_VALUE, p.WARP_FLIP_SINGLE_BIT, p.WARP_FLIP_TWO_BITS, p.WARP_RANDOM_VALUE, p.WARP_ZERO_VALUE],

}

# Used for instruction output-level address injection runs 
inst_address_igid_bfm_map = {
	p.GPR: [p.FLIP_SINGLE_BIT, p.FLIP_TWO_BITS],

#  Supported models
# 	p.GPR: [p.FLIP_SINGLE_BIT, p.FLIP_TWO_BITS],
# 	p.STORE_OP: [p.FLIP_SINGLE_BIT, p.FLIP_TWO_BITS],
}



#########################################################################
# List of apps 
# app_name: [suite name, binary name, expected runtime in secs on the target PC]
#########################################################################

apps = {
	'simple_add': ['example', 'simple_add', 2],
}

#########################################################################
# Separate list of apps and error models for parsing because one may want to
# parse results from a differt set of applications and error models 
#########################################################################
parse_rf_bfm_list = rf_bfm_list
parse_inst_value_igid_bfm_map = inst_value_igid_bfm_map
parse_inst_address_igid_bfm_map = inst_address_igid_bfm_map
parse_apps = apps

#########################################################################
# Set paths for application binary, run script, etc. 
#########################################################################
if 'SASSIFI_HOME' not in os.environ:
	print "Error: Please set SASSIFI_HOME environment variable" 
	sys.exit(-1)

SASSIFI_HOME = os.environ['SASSIFI_HOME']
suites_base_dir = SASSIFI_HOME + "/suites/"
logs_base_dir = SASSIFI_HOME + "/logs/"
run_script_base_dir = SASSIFI_HOME + "/run/"
bin_base_dir = SASSIFI_HOME + "/bin/"

app_log_dir = {} 
script_dir = {} 
bin_dir = {}
app_dir = {}
app_data_dir = {}
def set_paths(): 
	merged_apps = apps # merge the two dictionaries 
	merged_apps.update(parse_apps) 
	
	for app in merged_apps:
		suite_name = merged_apps[app][0]

		app_log_dir[app] = logs_base_dir + suite_name + "/" + app + "/"
		script_dir[app] = run_script_base_dir + suite_name + "/" + app + "/"
		bin_dir[app] = bin_base_dir + p.rf_inst + "_injector/" + suite_name + "/"
		app_dir[app] = suites_base_dir +  suite_name + "/" + app + "/"
		app_data_dir[app] = suites_base_dir +  suite_name + "/data/" # without the app name here!

set_paths()

#########################################################################
# Max number of registers per kernel per application, as reported by the
# compiler using "-Xptxas -v" option
#########################################################################

num_regs = {
	'simple_add': {
		'_Z10simple_addi': 6, 
		}
}

# update dictionaries for different applications here
def set_num_regs(): 
	# update the path to the kerne_regcount.p file if it is stored in a different location
	app = "simple_add"
	num_regs[app] = cPickle.load(open(suites_base_dir + apps[app][0] + "/" + app + "/" + app + "_kernel_regcount.p", "rb"))

set_num_regs()

#########################################################################
# Parameterizing file names
#########################################################################
run_script = "sassifi_run.sh"
clean_logs_script = "sassifi_clean_logs.sh"
injection_seeds = "sassifi-injection-seeds.txt"
stdout_file = "stdout.txt"
stderr_file = "stderr.txt"
output_diff_log = "diff.log"
stdout_diff_log = "stdout_diff.log"
stderr_diff_log = "stderr_diff.log"
special_sdc_check_log = "special_check.log"

#########################################################################
# Number of gpus to use for error injection runs
#########################################################################
NUM_GPUS = 2

