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

import sys, re, string, os, operator, math, datetime, random
import common_params as cp

# parse the file with inst count info per thread and create a tid->inst_count map
def read_inst_counts(app_dir, app):
	countList = []
	fName = app_dir + "/sassifi-inst-counts.txt"
	if not os.path.exists(fName):
		print "%s file not found!" %fName 
		return countList

	f = open(fName, "r")
	next(f) # Skip first line. This line has just the format.
	for line in f:
		line = line.rstrip()
		countList.append(line.split(":"))
		#countList.append([kName, int(kCount), int(storeCount), int(loadCount), int(gpCount), int(ccCount), int(prCount)])
	f.close()

	return countList 

# Sample format from sassifi-inst-counts.txt - kName:kernelCount:GPR:CC:PR:STORE_OP:IADD_IMUL_OP:FADD_FMUL_OP:DADD_DMUL_OP:MAD_OP:FFMA_OP:DFMA_OP:SETP_OP:LDS_OP:LD_OP:MISC_OP:opWillNotExecuteCount::ATOMS:B2R:BAR:BFE:BFI:BPT:BRA:BRK:BRX:CAL:CAS:CCTL:CCTLL:CCTLT:CONT:CS2R:CSET:CSETP:DADD:DEPBAR:DFMA:DMNMX:DMUL:DSET:DSETP:EXIT:F2F:F2I:FADD:FADD32I:FCHK:FCMP:FFMA:FFMA32I:FLO:FMNMX:FMUL:FMUL32I:FSET:FSETP:FSWZ:FSWZADD:I2F:I2I:IADD:IADD3:IADD32I:ICMP:IMAD:IMAD32I:IMADSP:IMNMX:IMUL:IMUL32I:ISAD:ISCADD:ISCADD32I:ISET:ISETP:JCAL:JMX:LD:LDC:LDG:LDL:LDLK:LDS:LDSLK:LDS_LDU:LDU:LD_LDU:LEA:LEPC:LONGJMP:LOP:LOP3:LOP32I:MEMBAR:MOV:MUFU:NOP:P2R:PBK:PCNT:PEXIT:PLONGJMP:POPC:PRET:PRMT:PSET:PSETP:R2B:R2P:RED:RET:RRO:S2R:SEL:SHF:SHFL:SHL:SHR:SSY:ST:STG:STL:STS:STSCUL:STSUL:STUL:SUATOM:SUBFM:SUCLAMP:SUEAU:SULD:SULDGA:SULEA:SUQ:SURED:SUST:SUSTGA:SYNC:TEX:TEXDEPBAR:TEXS:TLD:TLD4:TLD4S:TLDS:TXQ:UNMAPPED:USER_DEFINED:VMNMX:VOTE:XMAD
def get_inst_count_format():
	ret_str = "kName:kernelCount"
	for s in cp.IGID_STR:
		ret_str += ":" + s
	ret_str += ":WILL_NOT_EXECUTE"

	ret_str += ":ATOM:ATOMS:B2R:BAR:BFE:BFI:BPT:BRA:BRK:BRX:CAL:CAS:CCTL:CCTLL:CCTLT:CONT:CS2R:CSET:CSETP:DADD:DEPBAR:DFMA:DMNMX:DMUL:DSET:DSETP:EXIT:F2F:F2I:FADD:FADD32I:FCHK:FCMP:FFMA:FFMA32I:FLO:FMNMX:FMUL:FMUL32I:FSET:FSETP:FSWZ:FSWZADD:I2F:I2I:IADD:IADD3:IADD32I:ICMP:IMAD:IMAD32I:IMADSP:IMNMX:IMUL:IMUL32I:ISAD:ISCADD:ISCADD32I:ISET:ISETP:JCAL:JMX:LD:LDC:LDG:LDL:LDLK:LDS:LDSLK:LDS_LDU:LDU:LD_LDU:LEA:LEPC:LONGJMP:LOP:LOP3:LOP32I:MEMBAR:MOV:MUFU:NOP:P2R:PBK:PCNT:PEXIT:PLONGJMP:POPC:PRET:PRMT:PSET:PSETP:R2B:R2P:RED:RET:RRO:S2R:SEL:SHF:SHFL:SHL:SHR:SSY:ST:STG:STL:STS:STSCUL:STSUL:STUL:SUATOM:SUBFM:SUCLAMP:SUEAU:SULD:SULDGA:SULEA:SUQ:SURED:SUST:SUSTGA:SYNC:TEX:TEXDEPBAR:TEXS:TLD:TLD4:TLD4S:TLDS:TXQ:UNMAPPED:USER_DEFINED:VMNMX:VOTE:XMAD"

	return ret_str

#return total number of instructions of each type or opcode
def get_total_counts(countList):
	length = get_inst_count_format().count(':')-1
	total_icounts = [0] * length
	for l in countList:
		for i in range(length):
			total_icounts[i] += int(l[2+i])
	return total_icounts

# return total number of instructions in the countList 
def get_total_insts(countList, with_will_not_execute):
	total = 0
	for l in countList:
		# 3: 1 for kname, 1 for kcount and 1 for WILL NOT EXECUTE instruction count
		# 2: 1 for kname, 1 for kcount 
		start = cp.NUM_INST_TYPES+2 if with_will_not_execute else cp.NUM_INST_TYPES+3
		for i in range(start, len(countList[0])): 
			total += int(l[i])
	return total

def get_rf_injection_site_info(countList, inj_num, with_will_not_execute):
	start = 0
	for item in countList:
		st = cp.NUM_INST_TYPES+2 if with_will_not_execute else cp.NUM_INST_TYPES+3
		total = sum(int(item[i]) for i in range(st, len(countList[0]))) # total number of instructions in this kernel
		if start <= inj_num < start + total:
			return [item[0], item[1], inj_num-start] # return [kname, kcount, inj_num in this kernel]
		start += total
	return ["", -1, -1]

def get_injection_site_info(countList, inj_num, igid):
	start = 0
	idx = igid + 2
	for item in countList:
		if start <= inj_num < start + int(item[idx]):
			return [item[0], item[1], inj_num-start] # return [kname, kcount, inj_num in this kernel]
		start += int(item[idx])
	return ["", -1, -1]
