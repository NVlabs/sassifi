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

import os, sys, re, string, operator, math, datetime, subprocess, time, multiprocessing, pkgutil
import common_params as cp
import specific_params as sp

###############################################################################
# Basic functions and parameters
###############################################################################
before = -1

def print_usage():
	print "Usage: \n run_injections.py rf/inst_address/inst_value standalone/multigpu/cluster <clean>"
	print "Example1: \"run_injections.py rf standalone\" to run jobs on the current system"
	print "Example1: \"run_injections.py inst_value multigpu\" to run jobs on the current system using multiple gpus"
	print "Example2: \"run_injections.py inst_value cluster clean\" to launch jobs on cluster and clean all previous logs/results"

############################################################################
# Print progress every 10 minutes for jobs submitted to the cluster
############################################################################
def print_heart_beat(nj):
	global before
	if before == -1:
		before = datetime.datetime.now()
	if (datetime.datetime.now()-before).seconds >= 10*60:
		print "Jobs so far: %d" %nj
		before = datetime.datetime.now()

def get_log_name(app, inj_mode, igid, bfm):
	return sp.app_log_dir[app] + "results-mode" + str(inj_mode) + "-igid" + str(igid) + ".bfm" + str(bfm) + "." + str(sp.NUM_INJECTIONS) + ".txt"

############################################################################
# Clear log conent. Default is to append, but if the user requests to clear
# old logs, use this function.
############################################################################
def clear_results_file(app):
	for bfm in sp.rf_bfm_list: 
		open(get_log_name(app, cp.RF_MODE, "rf", bfm)).close()
	for igid in sp.inst_value_igid_bfm_map:
		for bfm in sp.inst_value_igid_bfm_map[igid]:
			open(get_log_name(app, cp.INST_VALUE_MODE, igid, bfm)).close()
	for igid in sp.inst_address_igid_bfm_map:
		for bfm in sp.inst_address_igid_bfm_map[igid]:
			open(get_log_name(app, cp.INST_ADDRESS_MODE, igid, bfm)).close()

############################################################################
# count how many jobs are done
############################################################################
def count_done(fname):
	return sum(1 for line in open(fname)) # count line in fname 


############################################################################
# check queue and launch multiple jobs on a cluster 
# This feature is not implemented.
############################################################################
def check_and_submit_cluster(cmd):
		print "This feature is not implement. Please write code here to submit jobs to your cluster.\n"
		sys.exit(-1)

############################################################################
# check queue and launch multiple jobs on the multigpu system 
############################################################################
jobs_list = []
pool = multiprocessing.Pool(sp.NUM_GPUS) # create a pool

def check_and_submit_multigpu(cmd):
	jobs_list.append("CUDA_VISIBLE_DEVICES=" + str(len(jobs_list)) + " " + cmd)
	if len(jobs_list) == sp.NUM_GPUS:
		pool.map(os.system, jobs_list) # launch jobs in parallel
		del jobs_list[:] # clear the list


###############################################################################
# Run Multiple injection experiments
###############################################################################
def run_multiple_injections_igid(app, inj_mode, igid, where_to_run):
	bfm_list = [] 
	if inj_mode == cp.RF_MODE: 
		bfm_list = sp.rf_bfm_list 
	if inj_mode == cp.INST_VALUE_MODE:
		bfm_list = sp.inst_value_igid_bfm_map[igid]
	if inj_mode == cp.INST_ADDRESS_MODE:
		bfm_list = sp.inst_address_igid_bfm_map[igid]
		
	for bfm in bfm_list:
		#print "App: %s, IGID: %s, EM: %s" %(app, cp.IGID_STR[igid], cp.EM_STR[bfm])
		total_jobs = 0
		inj_list_filenmae = sp.app_log_dir[app] + "/injection-list/mode" + inj_mode + "-igid" + str(igid) + ".bfm" + str(bfm) + "." + str(sp.NUM_INJECTIONS) + ".txt"
		inf = open(inj_list_filenmae, "r")
		for line in inf: # for each injection site 
			total_jobs += 1
			if total_jobs > sp.THRESHOLD_JOBS: 
				break; # no need to run more jobs

			#_Z24bpnn_adjust_weights_cudaPfiS_iS_S_ 0 1297034 0.877316323856 0.214340876321
			if len(line.split()) >= 5: 
				[kname, kcount, iid, opid, bid] = line.split() # obtains params for this injection
				if cp.verbose: print "\n%d: app=%s, Kernel=%s, kcount=%s, igid=%d, bfm=%d, instID=%s, opID=%s, bitLocation=%s" %(total_jobs, app, kname, kcount, igid, bfm, iid, opid, bid)
				cmd = "%s %s/scripts/run_one_injection.py %s %s %s %s %s %s %s %s %s" %(cp.PYTHON_P, sp.SASSIFI_HOME, inj_mode, str(igid), str(bfm), app, kname, kcount, iid, opid, bid)
				if where_to_run == "cluster":
					check_and_submit_cluster(cmd)
				elif where_to_run == "multigpu":
					check_and_submit_multigpu(cmd)
				else:
					os.system(cmd)
				if cp.verbose: print "done injection run "
			else:
				print "Line doesn't have enough params:%s" %line
			print_heart_beat(total_jobs)


###############################################################################
# wrapper function to call either RF injections or instruction level injections
###############################################################################
def run_multiple_injections(app, inj_mode, where_to_run):
	if inj_mode == cp.RF_MODE:
		run_multiple_injections_igid(app, inj_mode, "rf", where_to_run)
	elif inj_mode == cp.INST_VALUE_MODE:
		for igid in sp.inst_value_igid_bfm_map:
			run_multiple_injections_igid(app, inj_mode, igid, where_to_run)
	elif inj_mode == cp.INST_ADDRESS_MODE:
		for igid in sp.inst_address_igid_bfm_map:
			run_multiple_injections_igid(app, inj_mode, igid, where_to_run)

###############################################################################
# Starting point of the execution
###############################################################################
def main(): 
	if len(sys.argv) >= 3: 
		where_to_run = sys.argv[2]
	
		if where_to_run != "standalone":
			if pkgutil.find_loader('lockfile') is None:
				print "lockfile module not found. This python module is needed to run injection experiments in parallel." 
				sys.exit(-1)
	
		sorted_apps = [app for app, value in sorted(sp.apps.items(), key=lambda e: e[1][2])] # sort apps according to expected runtimes
		for app in sorted_apps: 
		 	print app
			if not os.path.isdir(sp.app_log_dir[app]): os.system("mkdir -p " + sp.app_log_dir[app]) # create directory to store summary
			if len(sys.argv) == 4: 
				if sys.argv[3] == "clean":
					clear_results_file(app) # clean log files only if asked for
	
		 	run_multiple_injections(app, sys.argv[1], where_to_run)
	
	else:
		print_usage()

if __name__ == "__main__":
    main()
