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
#include "creinforce.h"

#include "crewardfunction.h"
#include "cpolicies.h"
#include "cgradientfunction.h"
#include "cfeaturefunction.h"
#include "cstatecollection.h"
#include "cstate.h"
#include "cstateproperties.h"
#include "cstatemodifier.h"
#include "cvetraces.h"
#include "cvfunction.h"
#include "cqfunction.h"
#include "caction.h"


CConstantReinforcementBaseLineCalculator::CConstantReinforcementBaseLineCalculator(double b)
{
	addParameter("ReinforcementBaseLine", b);
}

double CConstantReinforcementBaseLineCalculator::getReinforcementBaseLine(int )
{
	return getParameter("ReinforcementBaseLine");
}


CAverageReinforcementBaseLineCalculator::CAverageReinforcementBaseLineCalculator(CRewardFunction *rewardFunction, double updateRate = 0.1) : CSemiMDPRewardListener(rewardFunction)
{
	averageReward = 0;
	addParameter("AverageRewardMinUpdateRate", updateRate);
	steps = 0;

}


double CAverageReinforcementBaseLineCalculator::getReinforcementBaseLine(int )
{
	return averageReward;
}

void CAverageReinforcementBaseLineCalculator::nextStep(CStateCollection *, CAction *, double reward, CStateCollection *)
{
	steps ++;
	double gamma = getParameter("AverageRewardMinUpdateRate");
	double alpha = 1 / steps;

	if (alpha < gamma)
	{
		alpha = gamma;
	}

	averageReward = (1 - alpha) * averageReward + alpha * reward;
}

void CAverageReinforcementBaseLineCalculator::newEpisode()
{
	steps = 0;
	averageReward = 0;
}


CREINFORCELearner::CREINFORCELearner(CRewardFunction *reward, CStochasticPolicy *policy, CGradientUpdateFunction *updateFunction, CReinforcementBaseLineCalculator *baseLine) : CSemiMDPRewardListener(reward)
{
	this->policy = policy;
	this->baseLine = baseLine;
	this->updateFunction = updateFunction;

	addParameters(policy);
	addParameters(baseLine);

	addParameter("REINFORCELearningRate", 0.2);

	gradient = new CFeatureList();

	eTraces = new CGradientVETraces(NULL);

	addParameters(eTraces);
}

CREINFORCELearner::~CREINFORCELearner()
{
	delete gradient;
	delete eTraces;
}

void CREINFORCELearner::nextStep(CStateCollection *oldState, CAction *action, double reward, CStateCollection *)
{
	gradient->clear();

	policy->getActionProbabilityLnGradient(oldState, action, action->getActionData(), gradient);

	eTraces->updateETraces(action->getDuration());
	eTraces->addGradientETrace(gradient, 1.0);

	gradient->clear();
	CFeatureList *eTraceList = eTraces->getGradientETraces();

	CFeatureList::iterator it = eTraceList->begin();
	for (; it != eTraceList->end(); it ++)
	{
		gradient->update((*it)->featureIndex, (*it)->factor * (reward - baseLine->getReinforcementBaseLine((*it)->featureIndex)));
	}
	
	updateFunction->updateGradient(gradient, getParameter("REINFORCELearningRate"));
}

void CREINFORCELearner::newEpisode()
{
	eTraces->resetETraces();
}

CGradientVETraces *CREINFORCELearner::getETraces()
{
	return eTraces;
}


