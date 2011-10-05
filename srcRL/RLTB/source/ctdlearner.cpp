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
#include <stdlib.h>
#include <assert.h>
#include <time.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include <vector>

#include "cpolicies.h"
#include "ctdlearner.h"
#include "cqfunction.h"
#include "cagentcontroller.h"
#include "caction.h"
#include "cstatecollection.h"
#include "cstate.h"
#include "cqetraces.h"
#include "cresiduals.h"
#include "cfeaturefunction.h"

// For RBF network
#include "ckdtrees.h"
#include "cadaptivesoftmaxnetwork.h"

CTDLearner::CTDLearner(CRewardFunction *rewardFunction,CAbstractQFunction *qfunction, CAbstractQETraces *etraces, CAgentController *estimationPolicy) : CSemiMDPRewardListener(rewardFunction) {
    this->qfunction = qfunction;
    this->etraces = etraces;
	this->estimationPolicy = estimationPolicy;

	addParameter("QLearningRate", 0.2);
	addParameter("DiscountFactor",0.95);
	addParameters(qfunction);
	addParameters(etraces);

	addParameter("ResetETracesOnWrongEstimate", 1.0);

	
	if (estimationPolicy)
	{
		addParameters(estimationPolicy);
	}

	this->externETraces = true;

	this->actionDataSet = new CActionDataSet(qfunction->getActions());

	lastEstimatedAction = NULL;
}

CTDLearner::CTDLearner(CRewardFunction *rewardFunction, CAbstractQFunction *qFunction, CAgentController *estimationPolicy) : CSemiMDPRewardListener(rewardFunction)
{
	this->qfunction = qFunction;
    this->etraces = qFunction->getStandardETraces();
	this->estimationPolicy = estimationPolicy;

	addParameter("QLearningRate", 0.2);
	addParameter("DiscountFactor",0.95);
	addParameters(qfunction);
	addParameters(etraces);

	addParameter("ResetETracesOnWrongEstimate", 1.0);

	if (estimationPolicy)
	{
		addParameters(estimationPolicy);
	}
	this->externETraces = false;
	this->actionDataSet = new CActionDataSet(qfunction->getActions());

	lastEstimatedAction = NULL;
}


CTDLearner::~CTDLearner() 
{
	if (!this->externETraces)
	{
		delete etraces;
	}
	delete actionDataSet;
}

void CTDLearner::newEpisode() {
	lastEstimatedAction = NULL;
}

double CTDLearner::getTemporalDifference(CStateCollection *oldState, CAction *action, double reward, CStateCollection *newState)
{
	double newQ = 0.0, oldQ = 0.0;
	double temporalDiff = 0.0;

	
	int duration = 1;

	if (action->isType(MULTISTEPACTION))
	{
		duration = dynamic_cast<CMultiStepAction *>(action)->getDuration();
	}

//	assert(lastEstimatedAction->getIndex() >= 0);
    oldQ = qfunction->getValue(oldState, action); // Save old prediction: Q(st,at)
    
    if (!newState->isResetState())
	{
		if (lastEstimatedAction == NULL)
		{
			lastEstimatedAction = qfunction->getMax(newState, qfunction->getActions(), actionDataSet);
		}
		
	    newQ = qfunction->getValue(newState, lastEstimatedAction, actionDataSet->getActionData(lastEstimatedAction));
	}
	else
	{
		DebugPrint('t', "TD Learner: Last State of Episode, Action %d\n", qfunction->getActions()->getIndex(action));
	}

	temporalDiff = getResidual(oldQ, reward, duration, newQ);

	DebugPrint('t', "OldQValue: %f\n", oldQ);
	DebugPrint('t', "NewQValue: %f\n", newQ);
	DebugPrint('t', "Reward: %f\n", reward);
	DebugPrint('t', "TemporalDiff: %f\n", temporalDiff);

	sendErrorToListeners(temporalDiff, oldState, action, NULL);

	return temporalDiff;
}

void CTDLearner::learnStep(CStateCollection *oldState, CAction *action, double reward, CStateCollection *newState)
{
	DebugPrint('t', "TD Learner Start\n");
	bool resetEtraces = getParameter("ResetETracesOnWrongEstimate") > 0.5;
	if (resetEtraces && (lastEstimatedAction == NULL ||  !action->isSameAction(lastEstimatedAction, actionDataSet->getActionData(lastEstimatedAction))))
	{
		etraces->resetETraces();
	}
	etraces->updateETraces(action);
    
    if (!newState->isResetState())
    {
	    lastEstimatedAction = estimationPolicy->getNextAction(newState, actionDataSet);
        
		// if there is no estimated action, take the greedy policy
		if (lastEstimatedAction == NULL)
		{
			lastEstimatedAction = qfunction->getMax(newState, qfunction->getActions(), actionDataSet);
		}
    }

	addETraces(oldState, newState,  action);

//	assert(qfunction->getActions()->getIndex(lastEstimatedAction) >= 0);
	
	etraces->updateQFunction(getParameter("QLearningRate") * getTemporalDifference(oldState, action, reward, newState));
	
	DebugPrint('t', "TD Learner end\n");
}

double CTDLearner::getResidual(double oldQ, double reward, int duration, double newQ)
{
	return (reward + pow(getParameter("DiscountFactor"), duration) * newQ - oldQ);
}

void CTDLearner::addETraces(CStateCollection *oldState, CStateCollection *, CAction *oldAction)
{
	etraces->addETrace(oldState, oldAction, 1.0);
}


void CTDLearner::nextStep(CStateCollection *oldState, CAction *action, double reward, CStateCollection *nextState) 
{
	
	learnStep(oldState, action, reward, nextState);
}

void CTDLearner::intermediateStep(CStateCollection *oldState, CAction *action, double reward, CStateCollection *nextState)
{
	addETraces(oldState, nextState, action);

	qfunction->updateValue(oldState, action, getParameter("QLearningRate") * getTemporalDifference(oldState, action, reward, nextState));
}

void CTDLearner::saveValues(char *filename) {
    FILE *stream = fopen (filename, "w");
	
    saveValues(stream);
    fclose (stream);
}

void CTDLearner::loadValues(char *filename) {
    FILE *stream = fopen(filename, "r");
    loadValues(stream);
    fclose(stream);
}

void CTDLearner::saveValues(FILE *stream) {
    assert(qfunction != NULL);
    qfunction->saveData(stream);
}

void CTDLearner::loadValues(FILE *stream) {
    assert(qfunction != NULL);
    qfunction->loadData(stream);
}


void CTDLearner::setAlpha(double alpha) {
    setParameter("QLearningRate", alpha); 
}

void CTDLearner::setLambda(double lambda) {
    assert(etraces != NULL);
    etraces->setLambda(lambda);
}

CAgentController* CTDLearner::getEstimationPolicy() {
    return estimationPolicy;
}

void CTDLearner::setEstimationPolicy(CAgentController * estimationPolicy) {
    this->estimationPolicy = estimationPolicy;
}

CAbstractQFunction* CTDLearner::getQFunction() {
    return qfunction;
}

CAbstractQETraces* CTDLearner::getETraces() {
    return etraces;
}


CQLearner::CQLearner(CRewardFunction *rewardFunction, CAbstractQFunction *qfunc) : CTDLearner(rewardFunction, qfunc,  new CQGreedyPolicy(qfunc->getActions(), qfunc)) 
{
}

CQLearner::~CQLearner() 
{
    delete estimationPolicy;
}

CSarsaLearner::CSarsaLearner(CRewardFunction *rewardFunction,  CAbstractQFunction *qfunction, CDeterministicController *agent) : CTDLearner(rewardFunction, qfunction, agent) 
{
	setParameter("ResetETracesOnWrongEstimate", 1.0);
}

CSarsaLearner::~CSarsaLearner() 
{
}

CTDGradientLearner::CTDGradientLearner(CRewardFunction *rewardFunction, CGradientQFunction *qfunction, CAgentController *agentController, CResidualFunction *residual, CResidualGradientFunction *residualGradient) : CTDLearner(rewardFunction, qfunction, new CGradientQETraces(qfunction), agentController)
{
	assert(qfunction->isType(GRADIENTQFUNCTION));
	this->gradientQFunction = qfunction;
	
	this->residual = residual;
	this->residualGradient = residualGradient;

	if (residual)
	{
		addParameters(residual);
	}
	if (residualGradient)
	{
		addParameters(residualGradient);
	}
	this->gradientQETraces = dynamic_cast<CGradientQETraces *>(etraces);
	gradientQETraces->setReplacingETraces(true);

	oldGradient = new CFeatureList();
	newGradient = new CFeatureList();
	residualGradientFeatures = new CFeatureList();
}

CTDGradientLearner::~CTDGradientLearner()
{
	delete oldGradient;
	delete newGradient;
	delete residualGradientFeatures;
}


double CTDGradientLearner::getResidual(double oldQ, double reward, int duration, double newQ)
{
	return residual->getResidual(oldQ, reward, duration, newQ);
}

void CTDGradientLearner::addETraces(CStateCollection *oldState, CStateCollection *newState, CAction *oldAction)
{
	
	double duration = oldAction->getDuration();

	oldGradient->clear();
	newGradient->clear();
	residualGradientFeatures->clear();
	
	gradientQFunction->getGradient(oldState, oldAction, oldAction->getActionData(), oldGradient);
	
	if (!newState->isResetState())
	{
		if (lastEstimatedAction == NULL)
		{
			lastEstimatedAction = qfunction->getMax(newState, qfunction->getActions(), actionDataSet);
		}
		
		gradientQFunction->getGradient(newState, lastEstimatedAction, actionDataSet->getActionData(lastEstimatedAction), newGradient);
	}

	residualGradient->getResidualGradient(oldGradient, newGradient, duration, residualGradientFeatures);
	
	if (DebugIsEnabled('t'))
	{
		DebugPrint('t', "Residual Gradient: ");
		residualGradientFeatures->saveASCII(DebugGetFileHandle('t'));
		DebugPrint('t', "\n");
	}

	gradientQETraces->addGradientETrace(residualGradientFeatures, - 1.0);
}


CTDResidualLearner::CTDResidualLearner(CRewardFunction *rewardFunction, 
                                       CGradientQFunction *qfunction, 
                                       CAgentController *agent, 
                                       CResidualFunction *residual, 
                                       CResidualGradientFunction *residualGradient, 
                                       CAbstractBetaCalculator *betaCalc,
                                       bool adaptive)

: CTDGradientLearner(rewardFunction, qfunction, agent, residual, residualGradient)

{
    adaptiveFeatures = adaptive;
    network = NULL;
    
	this->betaCalculator = betaCalc;

	residualETraces = new CGradientQETraces(qfunction);
	residualETraces->setReplacingETraces(true);

	directGradientTraces = new CGradientQETraces(qfunction);
	directGradientTraces->setReplacingETraces(true);

	residualGradientTraces = new CGradientQETraces(qfunction);
	residualGradientTraces->setReplacingETraces(true);


	addParameters(residualETraces);

	addParameters(directGradientTraces, "Gradient");
	addParameters(residualGradientTraces, "Gradient");

	addParameters(betaCalculator);

	addParameter("ScaleResidualGradient", 0.0);
}

CTDResidualLearner::~CTDResidualLearner()
{
	delete residualETraces;
}

void CTDResidualLearner::learnStep(CStateCollection *oldState, CAction *action, double reward, CStateCollection *newState)
{
	bool resetEtraces = getParameter("ResetETracesOnWrongEstimate") > 0.5;

	if (resetEtraces && (lastEstimatedAction == NULL ||  !action->isSameAction(lastEstimatedAction, actionDataSet->getActionData(lastEstimatedAction))))
	{
		etraces->resetETraces();
		residualETraces->resetETraces();
	}

	etraces->updateETraces(action);
	residualETraces->updateETraces(action);

	directGradientTraces->updateETraces(action);
	residualGradientTraces->updateETraces(action);


	if (!newState->isResetState())
	{
		lastEstimatedAction = estimationPolicy->getNextAction(newState, actionDataSet);

		// if there is no estimated action, take the greedy policy
		if (lastEstimatedAction == NULL)
		{
			lastEstimatedAction = qfunction->getMax(newState, qfunction->getActions(), actionDataSet);
		}
		assert(qfunction->getActions()->getIndex(lastEstimatedAction) >= 0);
	}
    
    //FIXME: not sure whether this should be here or not
    if (adaptiveFeatures)
        adaptFeatures();


	double td = getParameter("QLearningRate") * getTemporalDifference(oldState, action, reward, newState);
	addETraces(oldState, newState,  action, td);

	double beta = betaCalculator->getBeta(directGradientTraces->getGradientETraces(), residualGradientTraces->getGradientETraces());

	gradientQETraces->updateQFunction(td * (1- beta));
	residualETraces->updateQFunction(td * beta);
}

////////////////////////////////////////////////////////////////////////////////////

void CTDResidualLearner::setNetwork(CRBFCenterNetwork* nw) {
    network = nw;
}

CRBFCenterNetwork* CTDResidualLearner::getNetwork() {
    return network;
}

//TODO: extend to multiclass
void CTDResidualLearner::adaptFeatures() {
    int numCenters = network->getNumCenters();
//    int numDim = network->getNumDimensions();
    
    for (int i = 0; i < numCenters; ++i) {
        CRBFBasisFunction* rbf = network->getCenter(i);
        
        ColumnVector* centers = rbf->getCenter();
        ColumnVector* sigmas = rbf->getSigma();
        
        CState *state = stateCol->getState( originalState);
    }
}

////////////////////////////////////////////////////////////////////////////////////

void CTDResidualLearner::addETraces(CStateCollection *oldState, CStateCollection *newState, CAction *action, double td)
{

	double duration = action->getDuration();

	oldGradient->clear();
	newGradient->clear();
	residualGradientFeatures->clear();

	gradientQFunction->getGradient(oldState, action, action->getActionData(), oldGradient);

	if (!newState->isResetState())
	{
		if (lastEstimatedAction == NULL)
		{
			lastEstimatedAction = qfunction->getMax(newState, qfunction->getActions(), actionDataSet);
		}
		gradientQFunction->getGradient(newState, lastEstimatedAction, actionDataSet->getActionData(lastEstimatedAction), newGradient);
	}

	// Add Direct Gradient
	gradientQETraces->addGradientETrace(oldGradient, 1.0);

	residualGradient->getResidualGradient(oldGradient, newGradient,duration, residualGradientFeatures);

	if (getParameter("ScaleResidualGradient") > 0.5)
	{
		residualGradientFeatures->multFactor(oldGradient->getLength() / residualGradientFeatures->getLength());
	}

	// Add Residual Gradient
	residualETraces->addGradientETrace(residualGradientFeatures, - 1.0);

	directGradientTraces->addGradientETrace(oldGradient, td);
	residualGradientTraces->addGradientETrace(residualGradientFeatures, - td);


	if (DebugIsEnabled('t'))
	{
		DebugPrint('t', "Residual Gradient: ");
		residualGradientFeatures->saveASCII(DebugGetFileHandle('t'));
		DebugPrint('t', "\n");
	}
}

void CTDResidualLearner::newEpisode()
{
	CTDGradientLearner::newEpisode();
	residualETraces->resetETraces();

	residualGradientTraces->resetETraces();
	directGradientTraces->resetETraces();
}



CQAverageTDErrorLearner::CQAverageTDErrorLearner(CFeatureQFunction *l_averageErrorFunction, double l_updateRate) : CStateObject(l_averageErrorFunction->getFeatureCalculator())
{
	averageErrorFunction = l_averageErrorFunction;
	updateRate = l_updateRate;
	
	addParameter("TDErrorUpdateRate", updateRate);
}

CQAverageTDErrorLearner::~CQAverageTDErrorLearner()
{
}

void CQAverageTDErrorLearner::onParametersChanged()
{
	updateRate = CParameterObject::getParameter("TDErrorUpdateRate");
}
		
void CQAverageTDErrorLearner::receiveError(double error, CStateCollection *state, CAction *action, CActionData *)
{
	CState *featureState = state->getState(averageErrorFunction->getFeatureCalculator());
	
	for (unsigned int i = 0; i < featureState->getNumDiscreteStates(); i++)
	{
		int index = featureState->getDiscreteState(i);
		double featureFac = featureState->getContinuousState(i);
		double featureVal = averageErrorFunction->getValue(index, action);
				
		featureVal = featureVal * (updateRate  + (1 - featureFac) * (1 - updateRate)) + error * (1 - updateRate) * featureFac;
		averageErrorFunction->setValue(index, action, featureVal);
	}
}


CQAverageTDVarianceLearner::CQAverageTDVarianceLearner(CFeatureQFunction *averageErrorFunction, double updateRate) : CQAverageTDErrorLearner(averageErrorFunction, updateRate)
{
	
}

CQAverageTDVarianceLearner::~CQAverageTDVarianceLearner()
{
}

		
void CQAverageTDVarianceLearner::receiveError(double error, CStateCollection *state, CAction *action, CActionData *data)
{
	CQAverageTDErrorLearner::receiveError(pow(error, 2.0), state, action, data);
}



