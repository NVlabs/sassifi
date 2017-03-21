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


import os, sys, re, string, operator, math, datetime, time, signal, subprocess, shutil, glob, pkgutil
import common_params as cp
import specific_params as sp

###############################################################################
# Basic functions and parameters
###############################################################################

def print_usage():
	print "Usage: run_one_injection.py <inj_mode, igid, bfm, app, kernel_name, kcount, instID, opID, bitID>"

def get_seconds(td):
	return (td.microseconds + (td.seconds + td.days * 24 * 3600) * 10**6) / float(10**6)

###############################################################################
# Set enviroment variables for run_script and clean_logs_script
###############################################################################
[stdout_fname, stderr_fname, injection_seeds_file, new_directory] = ["", "", "", ""]
def set_env_variables(inj_mode, igid, bfm, app, kname, kcount, iid, opid, bid): # Set directory paths 
	if inj_mode == cp.RF_MODE:
		cp.rf_inst = "rf"
	elif inj_mode == cp.INST_VALUE_MODE:
		cp.rf_inst = "inst_value"
	else:
		cp.rf_inst = "inst_address"

	sp.set_paths() # update paths 
	global stdout_fname, stderr_fname, injection_seeds_file, new_directory
	kname_truncated = kname[:100] if len(kname) > 100 else kname
	new_directory = sp.logs_base_dir + sp.apps[app][0] + "/" + app + "-igid" +  igid + "-bfm" + bfm + "-" + kname_truncated + "-" + kcount + "-" + iid + "-" + opid + "-" + bid
	stdout_fname = new_directory + "/" + sp.stdout_file 
	stderr_fname = new_directory + "/" + sp.stderr_file
	injection_seeds_file = new_directory + "/" + sp.injection_seeds

	# Make sure that you use the same ENV variables in the run scripts
	os.environ['RUNSCRIPT_DIR'] = sp.bin_dir[app]
	os.environ['BIN_DIR'] = sp.bin_dir[app]
	os.environ['APP_DIR'] = sp.app_dir[app]
	os.environ['DATASET_DIR'] = sp.app_data_dir[app]

###############################################################################
# Record result in to a common file. This function uses file-locking such that
# mutliple parallel jobs can run safely write results to the file
###############################################################################
def record_result(inj_mode, igid, bfm, app, kname, kcount, iid, opid, bid, cat, pc, inst_type, tid, injBID, runtime, dmesg, value_str):
	res_fname = sp.app_log_dir[app] + "/results-mode" + inj_mode + "-igid" + str(igid) + ".bfm" + str(bfm) + "." + str(sp.NUM_INJECTIONS) + ".txt"
	result_str = kname + "-" + kcount + "-" + iid + "-" + opid 
	result_str += "-" + bid + ":" + str(pc) + ":" + str(inst_type) + ":" +  str(tid) 
	result_str += ":" + str(injBID) + ":" + str(runtime) + ":" + str(cat) + ":" + dmesg 
	result_str += ":" + value_str + "\n"
	if cp.verbose:
		print result_str

	has_filelock = False
	if pkgutil.find_loader('lockfile') is not None:
		from lockfile import FileLock
		has_filelock = True
	
	if has_filelock:
		lock = FileLock(res_fname)
		lock.acquire() #acquire lock

	rf = open(res_fname, "a")
	rf.write(result_str)
	rf.close()

	if has_filelock:
		lock.release() # release lock

	# Record the outputs if 
	if cat == cp.OUT_DIFF or cat == cp.STDOUT_ONLY_DIFF or cat == cp.APP_SPECIFIC_CHECK_FAIL:
		if not os.path.isdir(sp.app_log_dir[app] + "/sdcs"): os.system("mkdir -p " + sp.app_log_dir[app] + "/sdcs") # create directory to store sdcs 
		full_sdc_dir = sp.app_log_dir[app] + "/sdcs/sdc-" + app + "-igid" +  igid + "-bfm" + bfm + "-" + kname + "-" + kcount + "-" + iid + "-" + opid + "-" + bid 
		os.system("mkdir -p " + full_sdc_dir) # create directory to store sdc
		map((lambda x: shutil.copy(x, full_sdc_dir)), [stdout_fname, stderr_fname, injection_seeds_file, new_directory + "/" + sp.output_diff_log]) # copy stdout, stderr injection seeds, output diff
		shutil.make_archive(full_sdc_dir, 'gztar', full_sdc_dir) # archieve the outputs
		shutil.rmtree(full_sdc_dir, True) # remove the directory

###############################################################################
# Create params file. The contents of this file will be read by the injection run.
###############################################################################
def create_p_file(p_filename, igid, bfm, kname, kcount, iid, opid, bid):
	outf = open(p_filename, "w")
	if igid == "rf":
		outf.write(bfm + "\n" + kname + "\n" + kcount + "\n" + iid + "\n" + opid + "\n" + bid)
	else:
		outf.write(igid + "\n" + bfm + "\n" + kname + "\n" + kcount + "\n" + iid + "\n" + opid + "\n" + bid)
	outf.close()

###############################################################################
# Parse stadout file and get the injection information. 
###############################################################################
def get_inj_info():
	[pc, inst_type, tid, injBID] = ["", "", -1, -1]
	if os.path.isfile(stdout_fname): 
		logf = open(stdout_fname, "r")
		for line in logf:
			#":::Injecting: opcode=%s tid=%d instCount=%lld instType=GPR injOpID=%d injBIDSeed=%f:::", SASSIInstrOpcodeStrings[ap->GetOpcode()], get_flat_tid(), injInstID, injOpID, injBIDSeed);
			matchObj = re.match( r'.*:::Injecting: pc=(\S+) opcode=(\S+) tid=(\d+) .* injBID=(\d+):::.*', line, re.M) #:::Injecting: opcode=IADD tid=583564 instCount=1284359 instType=CC injBID=0:::
			if matchObj:
				[pc, inst_type, tid, injBID] = [matchObj.group(1), matchObj.group(2), matchObj.group(3), matchObj.group(4)]
				break 
		logf.close()
	return [pc, inst_type, tid, injBID];

###############################################################################
# Parse stadout file and obtain values before and after the injection 
###############################################################################
def get_inj_value_str():
	value_str = ""
	if os.path.isfile(stdout_fname): 
		logf = open(stdout_fname, "r")
		for line in logf:
			matchObj = re.match( r'.*:::value_before(.*)value_before_end:::.*', line, re.M)
			if matchObj:
				value_str += "value_before" + matchObj.group(1)
			matchObj = re.match( r'.*:::value_after(.*)value_after_end:::.*', line, re.M)
			if matchObj:
				value_str += ":value_after" + matchObj.group(1)
		logf.close()
	return value_str

###############################################################################
# Classify error injection result based on stdout, stderr, application output,
# exit status, etc.
###############################################################################
def classify_injection(app, igid, kname, kcount, iid, opid, bid, retcode, dmesg_delta):
	[found_line, found_error, found_skip] = [False, False, False]

	if retcode != 0:
		return cp.NON_ZERO_EC

	stdout_str = "" 
	if os.path.isfile(stdout_fname): 
		stdout_str = open(stdout_fname).read()
	
	if "Masked: Write before read" in stdout_str:
		return cp.MASKED_WRITTEN
	if "Masked: Error was never read" in stdout_str:
		return cp.MASKED_NOT_READ
	if "Skipped injection" in stdout_str:
		if cp.verbose: print "Skipped injection "
		return cp.OTHERS
	if ":::Injecting: " not in stdout_str: 
		if cp.verbose: print "Error Not Injected: %s, %s, %s, %s, %s, %s" %(igid, kname, kcount, iid, opid, bid)
		return cp.OTHERS
	if "Error: misaligned address" in stdout_str: 
		return cp.STDOUT_ERROR_MESSAGE
	if "Error: an illegal memory access was encountered" in stdout_str: 
		return cp.STDOUT_ERROR_MESSAGE
	if "Error: misaligned address" in open(stderr_fname).read(): # if error is found in the log standard err 
		return cp.STDOUT_ERROR_MESSAGE

	os.system(sp.script_dir[app] + "/sdc_check.sh") # perform SDC check

	if os.path.isfile(sp.output_diff_log) and os.path.isfile(sp.stdout_diff_log) and os.path.isfile(sp.stderr_diff_log):
		if os.path.getsize(sp.output_diff_log) == 0 and os.path.getsize(sp.stdout_diff_log) == 0 and os.path.getsize(sp.stderr_diff_log) == 0: # no diff is observed
			if "Kernel Exit Error:" in stdout_str: 
				return cp.KERNEL_ERROR # masked_kernel_error
			else:
				return cp.MASKED_OTHER
		elif os.path.isfile(sp.special_sdc_check_log):
			if os.path.getsize(sp.special_sdc_check_log) != 0:
				if "Xid" in dmesg_delta:
					return cp.DMESG_APP_SPECIFIC_CHECK_FAIL
				elif "Kernel Exit Error:" in stdout_str: 
					return cp.KERNEL_ERROR # sdc_kernel_error
				else:
					return cp.APP_SPECIFIC_CHECK_FAIL

		if os.path.getsize(sp.output_diff_log) != 0:
			if "Xid" in dmesg_delta:
				return cp.DMESG_OUT_DIFF 
			elif "Kernel Exit Error:" in stdout_str: 
				return cp.SDC_KERNEL_ERROR
			else:
				return cp.OUT_DIFF 
		elif os.path.getsize(sp.stdout_diff_log) != 0 and os.path.getsize(sp.stderr_diff_log) == 0:
			if "Xid" in dmesg_delta:
				return cp.DMESG_STDOUT_ONLY_DIFF
			elif "Kernel Exit Error:" in stdout_str: 
				return cp.SDC_KERNEL_ERROR
			else:
				return cp.STDOUT_ONLY_DIFF
		elif os.path.getsize(sp.stderr_diff_log) != 0 and os.path.getsize(sp.stdout_diff_log) == 0:
			if "Xid" in dmesg_delta:
				return cp.DMESG_STDERR_ONLY_DIFF
			elif "Kernel Exit Error:" in stdout_str: 
				return cp.SDC_KERNEL_ERROR
			else:
				return cp.STDERR_ONLY_DIFF
		else:
			if cp.verbose: 
				print "Other from here"
			return cp.OTHERS
	else: # one of the files is not found, strange
		print "%s, %s, %s not found" %(sp.output_diff_log, sp.stdout_diff_log, sp.stderr_diff_log)
		return cp.OTHERS

def cmdline(command):
	process = subprocess.Popen(args=command, stdout=subprocess.PIPE, shell=True)
	return process.communicate()[0]

###############################################################################
# Check for timeout and kill the job if it has passed the threshold
###############################################################################
def is_timeout(app, pr): # check if the process is active every 'factor' sec for timeout threshold 
	factor = 0.25
	retcode = None
	tt = cp.TIMEOUT_THRESHOLD * sp.apps[app][2] # sp.apps[app][2] = expected runtime
	if tt < 10: tt = 10

	to_th = tt / factor
	while to_th > 0:
		retcode = pr.poll()
		if retcode is not None:
			break
		to_th -= 1
		time.sleep(factor)

	if to_th == 0:
		os.killpg(pr.pid, signal.SIGKILL) # pr.kill()
		print "timeout"
		return [True, pr.poll()]
	else:
		return [False, retcode]

###############################################################################
# Run the actual injection run 
###############################################################################
def run_one_injection_job(inj_mode, igid, bfm, app, kname, kcount, iid, opid, bid):
	start = datetime.datetime.now() # current time
	[pc, inst_type, tid, injBID, ret_vat] = ["", "", -1, -1, -1]

	shutil.rmtree(new_directory, True)
	os.system("mkdir -p " + new_directory) # create directory to store temp_results
	create_p_file(injection_seeds_file, igid, bfm, kname, kcount, iid, opid, bid)

	dmesg_before = cmdline("dmesg")

	if cp.verbose: print "%s: %s" %(new_directory, sp.script_dir[app] + "/" + sp.run_script)
	cwd = os.getcwd()
	os.chdir(new_directory) # go to app dir
	if cp.verbose: start_main = datetime.datetime.now() # current time
	pr = subprocess.Popen(sp.script_dir[app] + "/" + sp.run_script, shell=True, executable='/bin/bash', preexec_fn=os.setsid) # run the injection job

	[timeout_flag, retcode] = is_timeout(app, pr)
	if cp.verbose: print "App runtime: " + str(get_seconds(datetime.datetime.now() - start_main))

	# Record kernel error messages (dmesg)
	dmesg_after = cmdline("dmesg")
	dmesg_delta = dmesg_after[len(dmesg_before):].replace("\n", "; ").replace(":", "-")

	if cp.verbose: os.system("cat " + sp.stdout_file + " " + sp.stderr_file)
	
	value_str = ""
	if timeout_flag:
		ret_cat = cp.TIMEOUT 
	else:
		ret_cat = classify_injection(app, igid, kname, kcount, iid, opid, bid, retcode, dmesg_delta)
		[pc, inst_type, tid, injBID] = get_inj_info()
		value_str = get_inj_value_str()
	
	os.chdir(cwd) # return to the main dir
	# print ret_cat

	elapsed = datetime.datetime.now() - start
	record_result(inj_mode, igid, bfm, app, kname, kcount, iid, opid, bid, ret_cat, pc, inst_type, tid, injBID, get_seconds(elapsed), dmesg_delta, value_str)

	if get_seconds(elapsed) < 0.5: time.sleep(0.5)
	shutil.rmtree(new_directory, True) # remove the directory once injection job is done

	return ret_cat

###############################################################################
# Starting point of the execution
###############################################################################
def main(): 
	# check if paths exit
	if not os.path.isdir(sp.SASSIFI_HOME): print "Error: Regression dir not found!"
	if not os.path.isdir(sp.logs_base_dir + "/results"): os.system("mkdir -p " + sp.logs_base_dir + "/results") # create directory to store summary

	if len(sys.argv) == 10:
		start= datetime.datetime.now()
		[inj_mode, igid, bfm, app, kname, kcount, iid, opid, bid] = [sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], str(sys.argv[5]), str(sys.argv[6]), str(sys.argv[7]), str(sys.argv[8]), str(sys.argv[9])]
		set_env_variables(inj_mode, igid, bfm, app, kname, kcount, iid, opid, bid) 
		
		err_cat = run_one_injection_job(inj_mode, igid, bfm, app, kname, kcount, iid, opid, bid) 
	
		elapsed = datetime.datetime.now() - start
		print "App=%s, Mode=%s, IGID=%s, EM=%s, Time=%f, Outcome: %s" %(app, inj_mode, igid, bfm, get_seconds(elapsed), cp.CAT_STR[err_cat-1])
	else:
		print_usage()

if __name__ == "__main__":
    main()
