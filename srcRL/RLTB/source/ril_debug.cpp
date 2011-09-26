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

#include "ril_debug.h"

// this seems to be dependent on how the compiler is configured.
// if you have problems with va_start, try both of these alternatives
//#ifdef HOST_SNAKE
//#include <stdarg.h>
//#else
//#ifdef HOST_SPARC
//#include <stdarg.h>
//#else
//#include "/usr/include/stdarg.h"
//#endif
//#endif
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <memory.h>
#include <string.h>
#include <stdexcept>
#include <map>
#include <iostream>
#include <sstream>
#include <string>
using namespace std;

static DebugMode debugMode = DebugModeFlushAlways;

static std::map<char, std::pair<char*, FILE *>*>*debugFileMap = new std::map<char, std::pair<char*, FILE*>*>();

void DebugSetMode(DebugMode mode)
{
	debugMode = mode;
}

DebugMode DebugGetMode()
{
	return debugMode;
}

void DebugInit(char* fileName, char *flagList, bool flgAppend)
{
	char* buffer;
	std::map<char, std::pair<char*, FILE *>*>::iterator it;
	std::pair<char*, FILE *>* temp;
	const char *mode = (flgAppend ? "a" : "w");
	FILE* f = fopen(fileName, mode);
    if (!f) 
	{
		throw new std::runtime_error("RL Toolbox DEBUG subsystem: could not open debug file!");
	}
	else
	{
		if (debugMode == DebugModeCloseAlways)
		{
			fclose(f);
			f = NULL;
		}
        for (unsigned int i = 0; i < strlen(flagList); i++)
		{
			it = debugFileMap->find(flagList[i]);
			if (it != debugFileMap->end())
			{
				if ((*it).second->second != NULL)
					fclose((*it).second->second);
				delete (*it).second->first;
				delete (*it).second;
			}
            buffer = new char[strlen(fileName) + 2];
			strcpy(buffer, fileName);
			temp = new std::pair<char*, FILE *>(buffer, f);
			(*debugFileMap)[flagList[i]] = temp;
		}		
	}
}

void DebugDisable(char* flagList)
{
	std::map<char, std::pair<char*, FILE *>*>::iterator it;
	if (flagList == NULL)
	{
		while (debugFileMap->size() > 0)
		{
			if ((*(debugFileMap->begin())).second->second != NULL)
				fclose((*(debugFileMap->begin())).second->second);
			delete (*(debugFileMap->begin())).second->first;
			delete (*(debugFileMap->begin())).second;
			debugFileMap->erase(debugFileMap->begin());
		}
	}
	else
	{
        for (unsigned int i = 0; i < strlen(flagList); i++)
		{
			it = debugFileMap->find(flagList[i]);
			if (it != debugFileMap->end())
			{
				if ((*it).second->second != NULL)
					fclose((*it).second->second);
				delete (*it).second->first;
				delete (*it).second;
				debugFileMap->erase(it);
			}
		}
	}
}

bool DebugIsEnabled(char flag)
{
    if (debugFileMap->size() > 0)
		if (flag == 0)
		{
			return true;
		}
		else
		{
			if (debugFileMap->find(flag) == debugFileMap->end())
				return debugFileMap->find('+') != debugFileMap->end();
			else
				return true;
		}
	else
		return false;
}

void DebugPrint(char flag, char *format, ...)
{
    char *fileName;
	std::map<char, std::pair<char*, FILE *>*>::iterator it;

	if (debugFileMap->size() > 0)
	{
		it = debugFileMap->find(flag);
		if (it == debugFileMap->end())
		{
			if (flag == '+')
			{
				it = debugFileMap->begin();
			}
			else
			{
				it = debugFileMap->find('+');
				if (it == debugFileMap->end())
				{
					return;
				}
			}
		}
		fileName = (*it).second->first;
		FILE* f = (*it).second->second;
		if (f == NULL) f = fopen(fileName, "a");
		va_list ap;
		// You will get an unused variable message here -- ignore it.
		va_start(ap, format);
		if (!f) 
		{
			throw new std::runtime_error("RL Toolbox DEBUG subsystem: could not open debug file for writing!");
		}
		else
		{
			vfprintf(f, format, ap);
			va_end(ap);
			if (debugMode == DebugModeFlushAlways) 
				fflush(f);
			else if (debugMode == DebugModeCloseAlways)
			{
				fclose(f);
				(*it).second->second = NULL;
			}
		}
    }
}

FILE *DebugGetFileHandle(char flag)
{
    char *fileName;
	std::map<char, std::pair<char*, FILE *> *>::iterator it;

	if (debugFileMap->size() > 0)
	{
		it = debugFileMap->find(flag);
		if (it == debugFileMap->end())
		{
			if (flag == '+')
			{
				it = debugFileMap->begin();
			}
			else
			{
				return DebugGetFileHandle('+');
			}
		}
		fileName = (*it).second->first;
		FILE* f = (*it).second->second;
		if (f == NULL) 
		{
			f = fopen(fileName, "a");
			(*it).second->second = f;
		}
		if (!f) 
		{
			throw new std::runtime_error("RL Toolbox DEBUG subsystem: could not open debug file for writing!");
		}
		else
		{
			return f;
		}
    }
	return NULL;
}

void DebugDisposeFileHandle(char flag)
{
	std::map<char, std::pair<char*, FILE *> *>::iterator it;
	if (debugFileMap->size() > 0 && debugMode != DebugModeLeaveCached)
	{
		it = debugFileMap->find(flag);
		if (it == debugFileMap->end())
		{
			it = debugFileMap->find('+');
			if (it == debugFileMap->end())
				return;
		}
		FILE* f = (*it).second->second;
		if (f != NULL) 
		{
			if (debugMode == DebugModeFlushAlways) 
				fflush(f);
			else if (debugMode == DebugModeCloseAlways)
			{
				fclose(f);
				(*it).second->second = NULL;
			}
		}
    }
}

