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
#include "cvfunctionfromqfunction.h"

#include "cpolicies.h"
#include "caction.h"
#include "cstatecollection.h"
#include "cstate.h"
#include "cstateproperties.h"
#include "cqfunction.h"
#include "cvetraces.h"
#include "ril_debug.h"

COptimalVFunctionFromQFunction::COptimalVFunctionFromQFunction(CAbstractQFunction *qfunction, CStateProperties *properties) : CAbstractVFunction(properties)
{
	this->qFunction = qfunction;
	availableActions = new CActionSet();
}

COptimalVFunctionFromQFunction::~COptimalVFunctionFromQFunction()
{
	delete availableActions;
}

double COptimalVFunctionFromQFunction::getValue(CStateCollection *state)
{
	qFunction->getActions()->getAvailableActions(availableActions, state);
	double value = qFunction->getMaxValue(state, availableActions);

	DebugPrint('v', "Optimal V-Function Value: %f\n", value);

	return value;
}

double COptimalVFunctionFromQFunction::getValue(CState *state)
{
	qFunction->getActions()->getAvailableActions(availableActions, state);
	return qFunction->getMaxValue(state,  availableActions);
}

CAbstractVETraces *COptimalVFunctionFromQFunction::getStandardETraces()
{
	return NULL;
}

CVFunctionFromQFunction::CVFunctionFromQFunction(CAbstractQFunction *qfunction, CStochasticPolicy *stochPolicy, CStateProperties *properties) : COptimalVFunctionFromQFunction(qfunction, properties)
{
	this->stochPolicy = stochPolicy;

	actionValues = new double[stochPolicy->getActions()->size()];
}

CVFunctionFromQFunction::~CVFunctionFromQFunction()
{
	delete actionValues;
}


CStochasticPolicy *CVFunctionFromQFunction::getPolicy()
{
	return this->stochPolicy;
}

void CVFunctionFromQFunction::setPolicy(CStochasticPolicy *policy)
{
	this->stochPolicy = policy;
}

double CVFunctionFromQFunction::getValue(CState *state)
{
	double qValue = 0;
	stochPolicy->getActions()->getAvailableActions(availableActions, state);

	if (availableActions->size() == 0)
	{
		return 0;
	}
	CActionSet::iterator it = availableActions->begin();
	stochPolicy->getActionProbabilities(state, availableActions, actionValues);

	for (int i = 0; it != availableActions->end(); it++, i++)
	{
        qValue += actionValues[i] * ((CAbstractQFunction *)qFunction)->getValue(state, *it);
	}
	return qValue;
}
	
double CVFunctionFromQFunction::getValue(CStateCollection *state)
{
	double qValue = 0;
	stochPolicy->getActions()->getAvailableActions(availableActions, state);
	CActionSet::iterator it = availableActions->begin();
	stochPolicy->getActionProbabilities(state, availableActions, actionValues);

	for (int i = 0; it != availableActions->end(); it++, i++)
	{
        qValue += actionValues[i] * ((CAbstractQFunction *)qFunction)->getValue(state, *it);
	}
	return qValue;
}
