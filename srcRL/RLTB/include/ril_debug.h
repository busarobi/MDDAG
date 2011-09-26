// Copyright (C) 2003
// Gerhard Neumann (gneumann@gmx.net)
// Stephan Neumann (sneumann@gmx.net) 
//                
// This file is part of RL Toolbox.
// http://www.igi.tugraz.at/ril_toolbox
//
// All rights reserved.
// 
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
// 1. Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
// 3. The name of the author may not be used to endorse or promote products
//    derived from this software without specific prior written permission.
// 
// THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
// IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
// OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
// IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT,
// INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
// NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
// THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

/** @file ril_debug.h
This File contains the debugging routines for the RL Toolbox.
They where inspired by 'nachos'
Copyright (c) 1992-1993 The Regents of the University of California.

The debugging routines allow the user to turn on selected
debugging messages and write then into different files.
The debug files can also be directly accessed. Enablig and
disabling of debug flags can easily be done by console or during
runtime.
You are encouraged to add your own
debugging flags. \n
The predefined debugging flags are: \n\n
	'+' -- turn on all debug messages \n
	'd' -- dynamic programming package \n
	'e' -- etraces \n
	'f' -- feature functions \n
	'p' -- policies \n
	'q' -- qfunctions \n
	't' -- tdleaners \n
	'v' -- torchfunction (gradient descent updates) \n
	\n
*/

#ifndef RIL_DEBUG_H
#define RIL_DEBUG_H
 
#include <stdio.h>
#include <iostream>
#include <string>
#include "general.h"


#define double double
#define MAX_EXP 400
#define MIN_EXP -400

/*#define double float
#define MAX_EXP 80
#define MIN_EXP -80*/

/// the debug mode sets the file caching behavior of the debug system
/** In most cases speed of debugging is most important, in other cases
it is most important, that all debug data is saved in case of a program crash.
The debug mode allows the user to configure the debug system to his/her needs
*/



enum DebugMode
{
    /// Slowest mode; all debug files are closed immedeately after writing
	DebugModeCloseAlways,
	/// all debug files are flushed immedeately after writing (default)
	DebugModeFlushAlways,
	/// debug data is left cached in memory -- don't forget to close all debug files (DebugDisable) before finishing your program
	DebugModeLeaveCached
};

/// set the debug mode
extern void DebugSetMode(DebugMode mode);

/// get the actual debug mode
extern DebugMode DebugGetMode();

/// initialize a debug file,  enable printing debug messages
/**
all debug flags in 'flags' will be written into file 'filename'
if flgAppend = false and the file already exists it will be deleted
*/
extern void DebugInit(char *filename, char *flags, bool flgAppend = true);

/// disable (all) debug flags
/**
all debug flags in 'flags' will be removed and their debug files closed
removes all flags if 'flags' = NULL
*/
extern void DebugDisable(char *flags = NULL);

/// Checks whether a debug flag is enabled
extern bool DebugIsEnabled(char flag = 0);

/// get a handle to the debug file
/** if the flag is disabled, the return value is NULL
the File has to be closed with DebugDisposeFileHandle(char flag) if no longer used
*/
extern FILE *DebugGetFileHandle(char flag); 

/// disposes of a file handle accquired by DebugGetFileHandle(char flag)
extern void DebugDisposeFileHandle(char flag); 

/// Print debug message into corresponding debug file if flag is enabled, else do nothing
extern void DebugPrint(char flag, char* format, ...);  	

#endif // RIL_DEBUG_H
