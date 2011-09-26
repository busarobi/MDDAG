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

#ifndef CERRORLISTENER_H
#define CERRORLISTENER_H

#include "cparameters.h"

class CState;
class CStateCollection;
class CAction;
class CActionData;
class CAbstractVFunction;
class CAbstractQFunction;

class CErrorListener : virtual public CParameterObject
{
public:
	virtual void receiveError(double error, CStateCollection *state, CAction *action, CActionData *data = NULL) = 0;
};

class CErrorSender
{
protected:
	std::list<CErrorListener *> *errorListeners;
public:
	CErrorSender();
	virtual ~CErrorSender();

	void sendErrorToListeners(double error, CStateCollection *state, CAction *action, CActionData *data = NULL);

	void addErrorListener(CErrorListener *listener);
	void removeErrorListener(CErrorListener *listener);
};

class CStateErrorLearner : public CErrorListener
{
protected:
	CAbstractVFunction *stateErrors;
public:
	CStateErrorLearner(CAbstractVFunction *stateErrors);

	virtual void receiveError(double error, CStateCollection *state, CAction *action, CActionData *data = NULL);
};

class CStateActionErrorLearner : public CErrorListener
{
protected:
	CAbstractQFunction *stateActionErrors;
public:
	CStateActionErrorLearner(CAbstractQFunction *stateActionErrors);

	virtual void receiveError(double error, CStateCollection *state, CAction *action, CActionData *data = NULL);

};

#endif
