#!/bin/bash
: ${SASSI_SRC:?"Need to set SASSI_SRC (e.g., export SASSI_SRC=/home/user/sassi/; do not set this to the instlibs/src directory)"}
: ${SASSIFI_HOME:?"Need to set SASSIFI_HOME (e.g., export SASSI_SRC=/home/user/sassifi/)"}

if [ ! -d $SASSI_SRC ] || [ ! -d $SASSIFI_HOME ] ; then
	printf "Either SASSI_SRC or SASSIFI_HOME do not point to a directory!\n"
	exit -1;
fi

NEW_HANDLER_DIR=$SASSI_SRC/instlibs/src/err_injector
mkdir $NEW_HANDLER_DIR

if [ $? -ne 0 ] ; then
	printf "Cannot create directory $NEW_HANDLER_DIR\n"
	exit -1;
else
	printf "Created directory $NEW_HANDLER_DIR\n"
fi

ln -s $SASSIFI_HOME/err_injector/Makefile $NEW_HANDLER_DIR/Makefile
ln -s $SASSIFI_HOME/err_injector/error_injector.h $NEW_HANDLER_DIR/error_injector.h
ln -s $SASSIFI_HOME/err_injector/injector.cu $NEW_HANDLER_DIR/injector.cu
ln -s $SASSIFI_HOME/err_injector/profiler.cu $NEW_HANDLER_DIR/profiler.cu
printf "Created soft-links\n"

printf "All done. Please cross-check links in the $NEW_HANDLER_DIR directory.\n"

