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

#include "cagentlistener.h"
#include "crewardfunction.h"
#include "ril_debug.h"

CSemiMDPRewardListener::CSemiMDPRewardListener(CRewardFunction *rewardFunction)
{
	this->semiMDPRewardFunction = rewardFunction;
}

void CSemiMDPRewardListener::nextStep(CStateCollection *oldState, CAction *action, CStateCollection *newState)
{
	nextStep(oldState, action, semiMDPRewardFunction->getReward(oldState, action,  newState), newState);
}

void CSemiMDPRewardListener::intermediateStep(CStateCollection *oldState, CAction *action, CStateCollection *nextState)
{
	intermediateStep(oldState, action, semiMDPRewardFunction->getReward(oldState, action, nextState), nextState);
}

void CSemiMDPRewardListener::setRewardFunction(CRewardFunction *rewardFunction)
{
	this->semiMDPRewardFunction = rewardFunction;	
}
	
CRewardFunction *CSemiMDPRewardListener::getRewardFunction()
{
	return semiMDPRewardFunction;
}


CAdaptiveParameterFromNStepsCalculator::CAdaptiveParameterFromNStepsCalculator(CParameters *targetObject, string targetParameter, int functionKind, int nStepsPerUpdate, double param0, double paramScale, double targetOffset, double targetScale) : CAdaptiveParameterUnBoundedValuesCalculator(targetObject, targetParameter, functionKind, param0, paramScale, targetOffset, targetScale)
{																																				
	targetValue = 0;
	this->nStepsPerUpdate = nStepsPerUpdate;
}

CAdaptiveParameterFromNStepsCalculator::~CAdaptiveParameterFromNStepsCalculator()
{
}

void CAdaptiveParameterFromNStepsCalculator::nextStep(CStateCollection *, CAction *, CStateCollection *)
{
	targetValue += 1;
	
	if (targetValue % nStepsPerUpdate == 0)
	{
		setParameterValue(targetValue);
	}
}

void CAdaptiveParameterFromNStepsCalculator::resetCalculator()
{
	targetValue = 0; 
	setParameterValue(targetValue);
}

CAdaptiveParameterFromNEpisodesCalculator::CAdaptiveParameterFromNEpisodesCalculator(CParameters *targetObject, string targetParameter, int functionKind, double param0, double paramScale, double targetOffset, double targetScale) : CAdaptiveParameterUnBoundedValuesCalculator(targetObject, targetParameter, functionKind, param0, paramScale, targetOffset, targetScale)
{					
	targetValue = 0;
}

CAdaptiveParameterFromNEpisodesCalculator::~CAdaptiveParameterFromNEpisodesCalculator()
{
}

void CAdaptiveParameterFromNEpisodesCalculator::newEpisode()
{
	targetValue += 1;
	setParameterValue(targetValue);
	
}

void CAdaptiveParameterFromNEpisodesCalculator::resetCalculator()
{
	targetValue = 0; 
	setParameterValue(targetValue);
}


CAdaptiveParameterFromAverageRewardCalculator::CAdaptiveParameterFromAverageRewardCalculator(CParameters *targetObject, string targetParameter, CRewardFunction *rewardFunction, int nStepsPerUpdate, int functionKind,  double paramOffset, double paramScale, double targetMin, double targetMax, double alpha) : CAdaptiveParameterBoundedValuesCalculator(targetObject, targetParameter, functionKind, paramOffset, paramScale, targetMin, targetMax), CSemiMDPRewardListener(rewardFunction)
{	
	this->alpha = alpha;
	addParameter("APRewardUpdateRate", alpha);
	this->nStepsPerUpdate = nStepsPerUpdate;
	nSteps = 0;
}

CAdaptiveParameterFromAverageRewardCalculator::~CAdaptiveParameterFromAverageRewardCalculator()
{
}

void CAdaptiveParameterFromAverageRewardCalculator::nextStep(CStateCollection *, CAction *, double reward, CStateCollection *)
{
	targetValue = targetValue * alpha + (1 - alpha) * reward;
	nSteps ++;
	
	if (nSteps % nStepsPerUpdate == 0)
	{
		setParameterValue(targetValue);
	}
}

void CAdaptiveParameterFromAverageRewardCalculator::onParametersChanged()
{
	CAdaptiveParameterBoundedValuesCalculator::onParametersChanged();
	alpha =  getParameter("APRewardUpdateRate");
}

void CAdaptiveParameterFromAverageRewardCalculator::resetCalculator()
{
	targetValue = targetMin; 
	nSteps = 0;
	setParameterValue(targetValue);
	
}

