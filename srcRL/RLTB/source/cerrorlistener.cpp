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
#include "cerrorlistener.h"
#include "caction.h"
#include "cstate.h"
#include "cvfunction.h"
#include "cqfunction.h"


CErrorSender::CErrorSender()
{
	errorListeners = new std::list<CErrorListener *>();
}

CErrorSender::~CErrorSender()
{
	delete errorListeners;
}

void CErrorSender::sendErrorToListeners(double error, CStateCollection *state, CAction *action, CActionData *data )
{
	std::list<CErrorListener *>::iterator it = errorListeners->begin();

	for (; it != errorListeners->end(); it ++)
	{
		(*it)->receiveError(error, state, action, data);
	}
}


void CErrorSender::addErrorListener(CErrorListener *listener)
{
	errorListeners->push_back(listener);
}

void CErrorSender::removeErrorListener(CErrorListener *listener)
{
	errorListeners->remove(listener);
}


CStateErrorLearner::CStateErrorLearner(CAbstractVFunction *stateErrors)
{
	this->stateErrors = stateErrors;
	addParameter("ErrorLearningRate", 0.5);
}

void CStateErrorLearner::receiveError(double error, CStateCollection *state, CAction *, CActionData * )
{
	double alpha = getParameter("ErrorLearningRate");
	double oldError = stateErrors->getValue(state);
	stateErrors->updateValue(state, alpha * (fabs(error) - oldError));
}

CStateActionErrorLearner::CStateActionErrorLearner(CAbstractQFunction *stateActionErrors)
{
	this->stateActionErrors = stateActionErrors;
	addParameter("ErrorLearningRate", 0.5);
}

void CStateActionErrorLearner::receiveError(double error, CStateCollection *state, CAction *action, CActionData *data )
{
	double alpha = getParameter("ErrorLearningRate");
	double oldError = stateActionErrors->getValue(state, action, data);
	stateActionErrors->updateValue(state, action, alpha * (fabs(error) - oldError), data);
}
