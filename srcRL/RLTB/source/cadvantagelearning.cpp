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

#include <math.h>

#include "ril_debug.h"
#include "cadvantagelearning.h"
#include "cvfunctionlearner.h"
#include "cpolicies.h"
#include "cvetraces.h"
#include "caction.h"
#include "cstatecollection.h"
#include "cstate.h"
#include "cstateproperties.h"


/*CAdvantageUpdating::CAdvantageUpdating(CRewardFunction *rewardFunction, CAbstractQFunction *qfunction, CVFunctionLearner *vLearner, double dt) : CTDLearner(rewardFunction, qfunction, qfunction->getStandardETraces(), NULL)
{
	this->vFunctionLearner = vLearner;
	this->vFunction = vLearner->getVFunction();

	this->dt = dt;

	addParameter("NormalizeRate", 0.2);
	addParameter("TimeScale", 1.0);
}*/

CAdvantageUpdating::CAdvantageUpdating(CRewardFunction *rewardFunction, CAbstractQFunction *qfunction, CAbstractVFunction *vFunction, double dt) :  CTDLearner(rewardFunction, qfunction, qfunction->getStandardETraces(), NULL)		
{
	//this->vFunctionLearner = NULL;
	this->vFunction = vFunction;
	vETraces = vFunction->getStandardETraces();

	addParameter("TimeIntervall", dt);

	addParameter("NormalizeRate", 0.2);
	addParameter("VLearningRate", 0.2);
	addParameter("TimeScale", 1.0);
	addParameter("DiscountFactor", 0.95);
}

CAdvantageUpdating::~CAdvantageUpdating()
{
	delete vETraces;
}

double CAdvantageUpdating::getTemporalDifference(CStateCollection *oldState, CAction *action, double reward, CStateCollection *nextState)
{
	double K = getParameter("TimeScale");
	double gamma = getParameter("DiscountFactor");
	double dt = getParameter("TimeIntervall");

	double currentMaxValue = qfunction->getMaxValue(oldState, qfunction->getActions());
	double oldVValue = vFunction->getValue(oldState);
	double newVValue = vFunction->getValue(nextState);
	double currentValue = qfunction->getValue(oldState, action);

	double temporalDifference = currentMaxValue + (reward + (pow(gamma, dt * action->getDuration()) * newVValue) - oldVValue) / (dt * K * action->getDuration()) - currentValue;
	DebugPrint('t', "Advantage Updating: %f %f %f %f %f, TD: %f\n", oldVValue, newVValue, currentMaxValue, currentValue, reward, temporalDifference);
	return temporalDifference;
}

void CAdvantageUpdating::addETraces(CStateCollection *oldState, CStateCollection *, CAction *action)
{
	etraces->addETrace(oldState, action);
	vETraces->addETrace(oldState);
}

void CAdvantageUpdating::learnStep(CStateCollection *oldState, CAction *action, double reward, CStateCollection *nextState)
{
	double oldMax = qfunction->getMaxValue(oldState, qfunction->getActions());
	double newMax = 0.0;

	etraces->updateETraces(action);

	addETraces(oldState, nextState,  action);

	etraces->updateQFunction(getParameter("QLearningRate") * getTemporalDifference(oldState, action, reward, nextState));

	newMax = qfunction->getMaxValue(oldState, qfunction->getActions());

	if (fabs(oldMax - newMax) > 0.00001)
	{
		vETraces->updateVFunction(getParameter("VLearningRate") * (newMax - oldMax) / getParameter("QLearningRate"));
		//vFunction->updateValue(oldState, getParameter("VLearningRate") * (newMax - oldMax) / getParameter("QLearningRate"));
	}
	
	// Normalize Step

	DebugPrint('t', "NormalizeStep: Aref: %f\n", newMax);
	CActionSet::iterator it = qfunction->getActions()->begin();

	for (; it != qfunction->getActions()->end(); it ++)
	{
		DebugPrint('t', "%f -> ",qfunction->getValue(oldState, *it));
		qfunction->updateValue(oldState, *it, - newMax * getParameter("NormalizeRate"));
		DebugPrint('t', "%f, ", qfunction->getValue(oldState, *it));
	}
}

CAdvantageLearner::CAdvantageLearner(CRewardFunction *rewardFunction, CGradientQFunction *qfunction, double dt, CAbstractBetaCalculator *betaCalc) : CTDResidualLearner(rewardFunction, qfunction, new CQGreedyPolicy(qfunction->getActions(), qfunction), NULL, NULL, betaCalc)
{
	addParameter("TimeIntervall", dt);
	addParameter("TimeScale", 1.0);
	addParameter("DiscountFactor", 0.95);

	actionDataSet2 = new CActionDataSet(qfunction->getActions());

}

CAdvantageLearner::~CAdvantageLearner()
{
	delete estimationPolicy;
	delete actionDataSet2;
}

double CAdvantageLearner::getTemporalDifference(CStateCollection *oldState, CAction *action, double reward, CStateCollection *nextState)
{
	int duration = action->getDuration();

	double K = getParameter("TimeScale");
	double dt = getParameter("TimeIntervall");
	double gamma = getParameter("DiscountFactor");

	double td = 0.0;

	if (! nextState->isResetState())
	{
		td = (reward + pow(gamma, dt * duration) * qfunction->getValue(nextState, lastEstimatedAction, actionDataSet->getActionData(lastEstimatedAction))) / (dt * duration * K) + (1 - 1/(dt * duration * K)) * qfunction->getMaxValue(oldState, qfunction->getActions()) - qfunction->getValue(oldState, action);
	}
	else
	{
		td = (reward) / (dt * duration * K) +  (1 - 1/(dt * duration * K)) * qfunction->getMaxValue(oldState, qfunction->getActions()) - qfunction->getValue(oldState, action);	
	}
	
	return td;
}

void CAdvantageLearner::addETraces(CStateCollection *oldState, CStateCollection *newState, CAction *action, double td)
{
	if (lastEstimatedAction == NULL)
	{
		lastEstimatedAction = qfunction->getMax(newState, qfunction->getActions(), actionDataSet);
	}

	double duration = action->getDuration();
	double K = getParameter("TimeScale");
	double dt = getParameter("TimeIntervall");

	oldGradient->clear();
	newGradient->clear();
	residualGradientFeatures->clear();

	gradientQFunction->getGradient(oldState, action, action->getActionData(), oldGradient);
	
	if (!newState->isResetState())
	{
		gradientQFunction->getGradient(newState, lastEstimatedAction, actionDataSet->getActionData(lastEstimatedAction), newGradient);
	}

    CAction *maxCurrentAction = qfunction->getMax(oldState, qfunction->getActions(),actionDataSet2);
	gradientQFunction->getGradient(oldState, maxCurrentAction, actionDataSet2->getActionData(maxCurrentAction), residualGradientFeatures);

	residualGradientFeatures->multFactor(- (1 - 1/(dt * duration * K)));

	residualGradientFeatures->add(oldGradient, 1.0);
	residualGradientFeatures->add(newGradient, - pow(getParameter("DiscountFactor"), dt * duration) / (dt * duration * K));

	directGradientTraces->addGradientETrace(oldGradient, td);
	residualGradientTraces->addGradientETrace(residualGradientFeatures, - td);


	// Add Direct Gradient
	gradientQETraces->addGradientETrace(oldGradient, 1.0);

	if (getParameter("ScaleResidualGradient") > 0.5)
	{
		residualGradientFeatures->multFactor(oldGradient->getLength() / residualGradientFeatures->getLength());
	}
	// Add Residual Gradient
	residualETraces->addGradientETrace(residualGradientFeatures, 1.0);
}

