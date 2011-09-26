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

/************************************************************************/
/*Error Numbers:														*/
/* 101, "DivergentVFunction"											*/
/* 701, "WrongTestScriptFormat"											*/
/************************************************************************/

#ifndef C__MYEXCEPTION__H
#define C__MYEXCEPTION__H

#include <stdio.h>
#include <iostream>
#include <string>

#include "ril_debug.h"

using namespace std;

/// Interface for all Exception classes used by the toolbox
/**
Each exception has an errornumber and an exception name. Additionally each subclass has to provide the function getInnerErrorMessage, which has to return a more detailed error message. The function getErrorMsg formats the inner errormessage with the errornumber and the exceptionname.
Following error numbers and exceptions exists:
- 101: "DivergentVFunction", see class CDivergentVFunctionException
- 1301, "WrongTestScriptFormat", not documented

The error system definitely needs some reworking!!!
*/
class CMyException 
{
protected:
	int errorNum;
	string exceptionName;

	virtual string getInnerErrorMsg() = 0;
public:
	CMyException(int errorNum, string exceptionName);
	virtual ~CMyException() {};
	virtual string getErrorMsg();
};

#endif

