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

#include "cvfunctionlearner.h"

#include "cstatecollection.h"
#include "cstate.h"
#include "cstateproperties.h"
#include "cstatemodifier.h"
#include "caction.h"
#include "cvetraces.h"
#include "cvfunction.h"

CAdaptiveParameterFromValueCalculator::CAdaptiveParameterFromValueCalculator(CParameters *targetObject, string targetParameter, CAbstractVFunction *l_vFunction, int nStepsPerUpdate, int functionKind, double param0, double paramScale, double targetMin, double targetMax) : CAdaptiveParameterBoundedValuesCalculator(targetObject, targetParameter, functionKind, param0, paramScale, targetMin, targetMax)
{
	this->vFunction = l_vFunction;
	
	nSteps = 0;
	value = 0;
	this->nStepsPerUpdate = nStepsPerUpdate;
}

CAdaptiveParameterFromValueCalculator::~CAdaptiveParameterFromValueCalculator()
{
}

void CAdaptiveParameterFromValueCalculator::nextStep(CStateCollection *, CAction *, CStateCollection *newState)
{
	value += vFunction->getValue(newState);
	
	nSteps ++;
	
	if (nSteps % nStepsPerUpdate == 0)
	{
		setParameterValue(value / nSteps);
		nSteps = 0;
		value = 0;
	}
}

void CAdaptiveParameterFromValueCalculator::resetCalculator()
{

	setParameterValue(targetMin);

	value = 0;	
	nSteps = 0;
}

CVFunctionLearner::CVFunctionLearner(CRewardFunction *rewardFunction, CAbstractVFunction *vFunction, CAbstractVETraces *eTraces) : CSemiMDPRewardListener(rewardFunction)
{
	this->vFunction = vFunction;
	this->eTraces = eTraces;

	bExternETraces = true;

	addParameter("VLearningRate", 0.2);
	addParameter("DiscountFactor", 0.95);

	addParameters(vFunction);
	addParameters(eTraces);
}

CVFunctionLearner::CVFunctionLearner(CRewardFunction *rewardFunction, CAbstractVFunction *vFunction) : CSemiMDPRewardListener(rewardFunction)
{
	this->vFunction = vFunction;
	this->eTraces = vFunction->getStandardETraces();

	bExternETraces = false;

	addParameter("VLearningRate", 0.2);
	addParameter("DiscountFactor", 0.95);


	addParameters(vFunction);
	addParameters(eTraces);
}

CVFunctionLearner::~CVFunctionLearner()
{
	if (!bExternETraces)
	{
		delete eTraces;
	}
}

double CVFunctionLearner::getLearningRate()
{
	return getParameter("VLearningRate");
}

void CVFunctionLearner::setLearningRate(double learningRate)
{
	setParameter("VLearningRate", learningRate);
}

void CVFunctionLearner::nextStep(CStateCollection *oldState, CAction *action, double reward, CStateCollection *nextState)
{
	double td = getTemporalDifference(oldState, action, reward, nextState);
	DebugPrint('t', "TD %f\n", td);
	updateVFunction(oldState, nextState, action->getDuration(), td);
}

void CVFunctionLearner::intermediateStep(CStateCollection *oldState, CAction *action, double reward, CStateCollection *nextState)
{
	addETraces(oldState, nextState, action->getDuration());
	vFunction->updateValue(oldState, getTemporalDifference(oldState, action, reward, nextState) * getLearningRate());
}


void CVFunctionLearner::updateVFunction(CStateCollection *oldState, CStateCollection *newState, int duration, double td)
{
	eTraces->updateETraces(duration);
	addETraces(oldState, newState, duration);
	eTraces->updateVFunction(td * getLearningRate());
}

void CVFunctionLearner::addETraces(CStateCollection *oldState, CStateCollection *, int )
{
	eTraces->addETrace(oldState);
}

CAbstractVETraces *CVFunctionLearner::getVETraces()
{
	return eTraces;
}

double CVFunctionLearner::getTemporalDifference(CStateCollection *oldState, CAction *action, double reward, CStateCollection *nextState)
{
	double oldQValue = vFunction->getValue(oldState);
	double newQValue = 0.0;
	
	double temporalDifference = 0.0;
	
	if (!nextState->getState()->isResetState())
	{
		newQValue = vFunction->getValue(nextState);
		temporalDifference = reward + pow(getParameter("DiscountFactor"), action->getDuration()) * newQValue - oldQValue;
		
		DebugPrint('t',"VFunctionLearner: oldQ %f,newQ %f, reward %f, ",oldQValue, newQValue, reward);
	}
	else
	{
		temporalDifference = reward - oldQValue;
		DebugPrint('t',"VFunctionLearner (last State): oldQ %f, reward %f, ",oldQValue, reward);
	}

	
	
	sendErrorToListeners(temporalDifference, oldState, action, NULL);
	
	return temporalDifference;
}

CAbstractVFunction *CVFunctionLearner::getVFunction()
{
	return vFunction;
}

void CVFunctionLearner::newEpisode()
{
	eTraces->resetETraces();
}


CVFunctionGradientLearner::CVFunctionGradientLearner(CRewardFunction *rewardFunction, CGradientVFunction *vFunction, CResidualFunction *residual, CResidualGradientFunction *residualGradientFunction) : CVFunctionLearner(rewardFunction, vFunction)
{
	this->residual = residual;
	this->residualGradientFunction = residualGradientFunction;

	addParameters(residual);
	addParameters(residualGradientFunction);

	this->gradientVFunction = vFunction;
	this->oldGradient = new CFeatureList();
	this->newGradient = new CFeatureList();
	this->residualGradient = new CFeatureList();
	this->gradientETraces = dynamic_cast<CGradientVETraces *>(eTraces);
}

CVFunctionGradientLearner::~CVFunctionGradientLearner()
{
	delete oldGradient;
	delete newGradient;
	delete residualGradient;
}

void CVFunctionGradientLearner::addETraces(CStateCollection *oldState, CStateCollection *newState, int duration)
{
	oldGradient->clear();
	newGradient->clear();
	residualGradient->clear();

	gradientVFunction->getGradient(oldState, oldGradient);
	
	if (!newState->isResetState())
	{
		gradientVFunction->getGradient(newState, newGradient);
	}

	residualGradientFunction->getResidualGradient(oldGradient, newGradient, duration, residualGradient);

	gradientETraces->addGradientETrace(residualGradient, - 1.0 );
}


double CVFunctionGradientLearner::getTemporalDifference(CStateCollection *oldState, CAction *action, double reward, CStateCollection *nextState)
{
	
	double temporalDifference = 0.0;
	
	if (! nextState->isResetState())
	{
		temporalDifference = residual->getResidual(vFunction->getValue(oldState), reward, action->getDuration(), vFunction->getValue(nextState));
	}
	else
	{
		temporalDifference = residual->getResidual(vFunction->getValue(oldState), reward, action->getDuration(), 0.0);
	}
	
	sendErrorToListeners(temporalDifference, oldState, action, NULL);
		
	return temporalDifference;
}

CVFunctionResidualLearner::CVFunctionResidualLearner(CRewardFunction *rewardFunction, CGradientVFunction *vFunction, CResidualFunction *residual, CResidualGradientFunction *residualGradient, CAbstractBetaCalculator *betaCalc) : CVFunctionGradientLearner(rewardFunction, vFunction, residual, residualGradient)
{
	this->betaCalculator = betaCalc;
	this->residualETraces = new CGradientVETraces(gradientVFunction);
	
	this->directGradientTraces = new CGradientVETraces(gradientVFunction);
	this->residualGradientTraces = new CGradientVETraces(gradientVFunction);

	addParameters(betaCalc);
	addParameters(residualETraces);

	addParameters(directGradientTraces, "Gradient");
	addParameters(residualGradientTraces, "Gradient");


	setParameter("GradientReplacingETraces", 0.0);
}

CVFunctionResidualLearner::~CVFunctionResidualLearner()
{
	delete residualETraces;
	delete residualGradientTraces;
	delete directGradientTraces;
}

void CVFunctionResidualLearner::newEpisode()
{
	CVFunctionGradientLearner::newEpisode();
	residualETraces->resetETraces();
	residualGradientTraces->resetETraces();
	directGradientTraces->resetETraces();
}

void CVFunctionResidualLearner::addETraces(CStateCollection *oldState, CStateCollection *newState, int duration, double td)
{
	oldGradient->clear();
	newGradient->clear();
	residualGradient->clear();

	gradientVFunction->getGradient(oldState, oldGradient);

	if (!newState->isResetState())
	{
		gradientVFunction->getGradient(newState, newGradient);
	}

	residualGradientFunction->getResidualGradient(oldGradient, newGradient, duration, residualGradient);

	if (DebugIsEnabled('v'))
	{
		DebugPrint('v', "Residual Gradient: ");
		residualGradient->saveASCII(DebugGetFileHandle('v'));
		DebugPrint('v', "\n");
	}

	directGradientTraces->addGradientETrace(oldGradient, td);
	residualGradientTraces->addGradientETrace(residualGradient, - td);

	gradientETraces->addGradientETrace(oldGradient, 1.0);
	residualETraces->addGradientETrace(residualGradient, -1.0);
}


void CVFunctionResidualLearner::updateVFunction(CStateCollection *oldState, CStateCollection *newState, int duration, double td)
{
	gradientETraces->updateETraces(duration);
	residualETraces->updateETraces(duration);

	residualGradientTraces->updateETraces(1);
	directGradientTraces->updateETraces(1);

	double beta = betaCalculator->getBeta(directGradientTraces->getGradientETraces(), residualGradientTraces->getGradientETraces());

	addETraces(oldState, newState, duration, td);


	double learningRate = getLearningRate();
	gradientETraces->updateVFunction(td * learningRate * (1 - beta));
	residualETraces->updateVFunction(td * learningRate * beta);
}

CVAverageTDErrorLearner::CVAverageTDErrorLearner(CFeatureVFunction *l_averageErrorFunction, double l_updateRate) : CStateObject(l_averageErrorFunction->getStateProperties())
{
	averageErrorFunction = l_averageErrorFunction;
	updateRate = l_updateRate;
	
	addParameter("TDErrorUpdateRate", updateRate);
}

CVAverageTDErrorLearner::~CVAverageTDErrorLearner()
{
}

void CVAverageTDErrorLearner::onParametersChanged()
{
	
	updateRate = CParameterObject::getParameter("TDErrorUpdateRate");
}
		
void CVAverageTDErrorLearner::receiveError(double error, CStateCollection *state, CAction *, CActionData *)
{
	CState *featureState = state->getState(averageErrorFunction->getStateProperties());
	
	for (unsigned int i = 0; i < featureState->getNumDiscreteStates(); i++)
	{
		int index = featureState->getDiscreteState(i);
		double featureFac = featureState->getContinuousState(i);
		double featureVal = averageErrorFunction->getFeature(index);
				
		featureVal = featureVal * (updateRate  + (1 - featureFac) * (1 - updateRate)) + error * (1 - updateRate) * featureFac;
		averageErrorFunction->setFeature(index, featureVal);
	}
}


CVAverageTDVarianceLearner::CVAverageTDVarianceLearner(CFeatureVFunction *averageErrorFunction, double updateRate) : CVAverageTDErrorLearner(averageErrorFunction, updateRate)
{
	
}

CVAverageTDVarianceLearner::~CVAverageTDVarianceLearner()
{
}

		
void CVAverageTDVarianceLearner::receiveError(double error, CStateCollection *state, CAction *action, CActionData *data)
{
	CVAverageTDErrorLearner::receiveError(pow(error, 2.0), state, action, data);
}



