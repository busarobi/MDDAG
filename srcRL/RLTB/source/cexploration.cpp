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


#include "cexploration.h"

#include "cstateproperties.h"
#include "ril_debug.h"
#include "cvfunction.h"
#include "cagentcontroller.h"
#include "cutility.h"
#include "caction.h"
#include "cstate.h"
#include "cstatecollection.h"

CVisitStateCounter::CVisitStateCounter(CFeatureVFunction *visits, double decay)
{
	this->visits = visits;
	addParameter("VisitDecayFactor", decay);
	addParameter("VisitDecayUpdateSteps", 1);

	weights = new double[visits->getNumWeights()];
	steps = 0;
}


CVisitStateCounter::~CVisitStateCounter()
{
	delete weights;
}

void CVisitStateCounter::nextStep(CStateCollection *state, CAction *, CStateCollection *)
{
	double decay = getParameter("VisitDecayFactor");
	int updateSteps = my_round(getParameter("VisitDecayUpdateSteps"));
	steps ++;
	if ((updateSteps > 0) && ((steps % updateSteps) == 0) && (decay < 1.0))
	{
		doDecay(decay);
	}
	visits->updateValue(state, 1.0);
}


void CVisitStateCounter::newEpisode()
{
}

void CVisitStateCounter::doDecay(double decay)
{
	visits->getWeights(weights);

	for (int i = 0; i < visits->getNumWeights(); i++)
	{
		weights[i] *= decay;
	}

	visits->setWeights(weights);
}

CVisitStateActionCounter::CVisitStateActionCounter(CFeatureQFunction *visits, double decay )
{
	this->visits = visits;
	addParameter("VisitDecayFactor", decay);
	addParameter("VisitDecayUpdateSteps", 1);

	weights = new double[visits->getNumWeights()];
	steps = 0;
}

CVisitStateActionCounter::~CVisitStateActionCounter()
{
	delete weights;
}


void CVisitStateActionCounter::doDecay(double decay)
{
	visits->getWeights(weights);

	for (int i = 0; i < visits->getNumWeights(); i++)
	{
		weights[i] *= decay;
	}

	visits->setWeights(weights);
}

void CVisitStateActionCounter::nextStep(CStateCollection *state, CAction *action, CStateCollection *)
{
	double decay = getParameter("VisitDecayFactor");
	int updateSteps = my_round(getParameter("VisitDecayUpdateSteps"));
	steps ++;
	if ((updateSteps > 0) && ((steps % updateSteps) == 0) && (decay < 1.0))
	{
		doDecay(decay);
	}
	visits->updateValue(state, action, 1.0);

}

void CVisitStateActionCounter::newEpisode()
{
}

CVisitStateActionEstimator::CVisitStateActionEstimator(CFeatureVFunction *visits, CFeatureQFunction *actionVisits, double decay ) : CVisitStateCounter(visits, decay)
{
	this->actionVisits = actionVisits;
	addParameter("VisitsEstimatorLearningRate", 0.5);
}

CVisitStateActionEstimator::~CVisitStateActionEstimator()
{

}

void CVisitStateActionEstimator::doDecay(double decay)
{
	actionVisits->getWeights(weights);

	for (int i = 0; i < visits->getNumWeights(); i++)
	{
		weights[i] *= decay;
	}

	visits->setWeights(weights);
}

void CVisitStateActionEstimator::nextStep(CStateCollection *state, CAction *action, CStateCollection *nextState)
{
	double oldValue = actionVisits->getValue(state, action);
	double newVisits = visits->getValue(nextState);

	CVisitStateCounter::nextStep(state, action, nextState);

	actionVisits->updateValue(state, action, getParameter("VisitsEstimatorLearningRate") * (newVisits - oldValue));
}

void CVisitStateActionEstimator::newEpisode()
{
}

CExplorationQFunction::CExplorationQFunction(CAbstractVFunction *stateVisitCounter, CAbstractQFunction *actionVisitCounter) : CAbstractQFunction(actionVisitCounter->getActions())
{
	this->actionVisitCounter = actionVisitCounter;
	this->stateVisitCounter = stateVisitCounter;
}

CExplorationQFunction::~CExplorationQFunction()
{
}

void CExplorationQFunction::updateValue(CStateCollection *, CAction *, double , CActionData *)
{
}

void CExplorationQFunction::setValue(CStateCollection *, CAction *, double , CActionData *)
{
}

double CExplorationQFunction::getValue(CStateCollection *state, CAction *action, CActionData *)
{
	double vValue = stateVisitCounter->getValue(state);
	double qValue = actionVisitCounter->getValue(state, action);

	if (fabs(qValue) < 0.000001)
	{
		qValue = 0.00001;
	}
	return vValue / qValue;
}

CAbstractQETraces *CExplorationQFunction::getStandardETraces()
{
	return NULL;
}

CQStochasticExplorationPolicy::CQStochasticExplorationPolicy(CActionSet *actions, CActionDistribution *distribution, CAbstractQFunction *qFunctoin, CAbstractQFunction *explorationFunction, double alpha) :CQStochasticPolicy(actions, distribution, qFunctoin)
{
	this->explorationFunction = explorationFunction;
	addParameter("ExplorationFactor", alpha);
	addParameter("AttentionFactor", 0.5);

	explorationValues = new double[actions->size()];

}

CQStochasticExplorationPolicy::~CQStochasticExplorationPolicy()
{
	delete explorationValues;
}


void CQStochasticExplorationPolicy::getActionValues(CStateCollection *state, CActionSet *availableActions, double *actionValues, CActionDataSet *)
{
	for (unsigned int i = 0; i < availableActions->size(); actionValues[i++] = 0.0);
	qfunction->getActionValues(state, availableActions, actionValues);

	for (unsigned int i = 0; i < availableActions->size(); explorationValues[i++] = 0.0);
	explorationFunction->getActionValues(state, availableActions, explorationValues);

    double alpha = getParameter("ExplorationFactor");
	double attentionFactor = getParameter("AttentionFactor");
	
	for (unsigned int i = 0; i < availableActions->size(); i++)
	{
		actionValues[i] = 2 * (attentionFactor * actionValues[i] + (1 - attentionFactor) * alpha * explorationValues[i]);
	}

}

CSelectiveExplorationCalculator::CSelectiveExplorationCalculator(CQStochasticExplorationPolicy *explorationFunction)
{
	attention = 0.5;
	this->explorationPolicy =  explorationFunction;
	addParameter("SelectiveAttentionSquashingFactor", 1.0);
}

CSelectiveExplorationCalculator::~CSelectiveExplorationCalculator()
{

}

void CSelectiveExplorationCalculator::nextStep(CStateCollection *state, CAction *action, CStateCollection *)
{
	double alpha = getParameter("SelectiveAttentionSquashingFactor");

	double kappa = attention * explorationPolicy->getQFunction()->getValue(state, action, NULL) / explorationPolicy->getQFunction()->getMaxValue(state, explorationPolicy->getActions());
	kappa -= (1 - attention) * explorationPolicy->getExplorationQFunction()->getValue(state, action);

	attention = 0.8 / (1 + exp(- alpha * kappa)) + 0.1;

	explorationPolicy->setParameter("SelectiveAttention", attention);
}

void CSelectiveExplorationCalculator::newEpisode()
{
	attention = 0.5;
	explorationPolicy->setParameter("SelectiveAttention", attention);
}

