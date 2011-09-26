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

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include "ccontinuousactions.h"
#include "cpolicies.h"
#include "cstatecollection.h"
#include "cvfunction.h"
#include "caction.h"
#include "cstate.h"
#include "cstateproperties.h"
#include "cstatemodifier.h"
#include "cfeaturefunction.h"
#include "cvetraces.h"

#include "ril_debug.h"
#include <assert.h>
#include <math.h>

CContinuousActionData::CContinuousActionData(CContinuousActionProperties *properties) : ColumnVector(properties->getNumActionValues())
{
	this->properties = properties;
}

CContinuousActionData::~CContinuousActionData()
{
	//delete actionValues;
}


void CContinuousActionData::initData(double initVal)
{
	for (int i = 0; i < nrows(); i ++)
	{
		element(i) = initVal;
	}
}

double CContinuousActionData::getDistance(ColumnVector *vector)
{
	ColumnVector distance = *this;
	distance = distance - *vector;
	
	return distance.norm_Frobenius();
}

void CContinuousActionData::setData(CActionData *actionData)
{
	if (isChangeAble())
	{
		CContinuousActionData *contData = dynamic_cast<CContinuousActionData *>(actionData);

		*this = *contData;
	}
}

void CContinuousActionData::setActionValue(int dim, double value)
{
	if(isChangeAble())
	{
		//assert(value <= properties->getMaxActionValue(dim) + 0.01 && value >= properties->getMinActionValue(dim) - 0.01);
		element(dim) = value;
	}
}

void CContinuousActionData::normalizeAction()
{
	for (int dim = 0; dim < nrows(); dim ++)
	{
		if (element(dim) > properties->getMaxActionValue(dim))
		{
			element(dim) = properties->getMaxActionValue(dim);
		}
		else
		{
			if (element(dim) < properties->getMinActionValue(dim))
			{
				element(dim) = properties->getMinActionValue(dim);
			}
		}
	}
}

double CContinuousActionData::getActionValue(int dim)
{
	return element(dim);
}

/*double *CContinuousActionData::getActionValues()
{
	return actionValues;
}*/

void CContinuousActionData::saveASCII(FILE *stream)
{
	fprintf(stream,"[");
	for (unsigned int i = 0; i < properties->getNumActionValues(); i++)
	{
		fprintf(stream, "%lf ", element(i));
	} 
	fprintf(stream, "]");
}

void CContinuousActionData::loadASCII(FILE *stream)
{
	int tmp = fscanf(stream,"[");
	for (unsigned int i = 0; i < properties->getNumActionValues(); i++)
	{
		double buf;
		tmp = fscanf(stream, "%lf ", &buf);
		element(i) = buf;
	}
	tmp = fscanf(stream, "]");
}

void CContinuousActionData::saveBIN(FILE *stream)
{
	int tmp = 0;
	for (unsigned int i = 0; i < properties->getNumActionValues(); i++)
	{
		double buf = element(i);
		tmp = fwrite(&buf, sizeof(double), 1, stream);
	}
	
}

void CContinuousActionData::loadBIN(FILE *stream)
{
	for (unsigned int i = 0; i < properties->getNumActionValues(); i++)
	{
		double buf;
		fread(&buf, sizeof(double), 1, stream);
		element(i) = buf;
	}
}

CContinuousActionProperties::CContinuousActionProperties(int numActionValues)
{
	this->numActionValues = numActionValues;

	minValues = new double[numActionValues];
	maxValues = new double[numActionValues];

	for (int i = 0; i < numActionValues; i++)
	{
		minValues[i] = 0.0;
		maxValues[i] = 1.0;
	}
}

CContinuousActionProperties::~CContinuousActionProperties()
{
	delete minValues;
	delete maxValues;
}

unsigned int CContinuousActionProperties::getNumActionValues()
{
	return numActionValues;
}

double CContinuousActionProperties::getMinActionValue(int dim)
{
	return minValues[dim];
}

double CContinuousActionProperties::getMaxActionValue(int dim)
{
	return maxValues[dim];
}

void CContinuousActionProperties::setMinActionValue(int dim, double value)
{
	minValues[dim] = value;
}

void CContinuousActionProperties::setMaxActionValue(int dim, double value)
{
	maxValues[dim] = value; 
}

CContinuousAction::CContinuousAction(CContinuousActionProperties *properties, CContinuousActionData *actionData) : CPrimitiveAction(actionData)
{
	continuousActionData = actionData;
	this->properties = properties;

	addType(CONTINUOUSACTION);
}

CContinuousAction::CContinuousAction(CContinuousActionProperties *properties) : CPrimitiveAction(new CContinuousActionData(properties))
{
	continuousActionData = dynamic_cast<CContinuousActionData *>(actionData);
	this->properties = properties;

	addType(CONTINUOUSACTION);
}

CContinuousAction::~CContinuousAction()
{
}

double CContinuousAction::getActionValue(int dim)
{
	return continuousActionData->getActionValue(dim);
}

unsigned int CContinuousAction::getNumDimensions()
{
	return (unsigned int) continuousActionData->nrows();
}

bool CContinuousAction::equals(CAction *action)
{
	if (action->isType(CONTINUOUSSTATICACTION))
	{
		CStaticContinuousAction *staticAction = dynamic_cast<CStaticContinuousAction *>(action);
		return this == (staticAction->getContinuousAction());
	}
	else
	{
		return this == (action);
	}
}

bool CContinuousAction::isSameAction(CAction *action, CActionData *data)
{
	if (action->isType(CONTINUOUSACTION))
	{
		CContinuousAction *lcontAction = dynamic_cast<CContinuousAction *>(action);
		if (lcontAction->getContinuousActionProperties() == getContinuousActionProperties())
		{
			CContinuousActionData *lcontData;
			if (data)
			{
				lcontData = dynamic_cast<CContinuousActionData *>(data);	
			}
			else
			{
				lcontData = lcontAction->getContinuousActionData();
			}
			ColumnVector distance = *continuousActionData - *lcontData;
			return distance.norm_Frobenius() < 0.0001;
		}
		else
		{
			return false;
		}
	}
	else
	{
		return false;
	}
}

void CContinuousAction::loadActionData(CActionData *data)
{
	CPrimitiveAction::loadActionData(data);
	continuousActionData->normalizeAction();
}

CContinuousActionProperties *CContinuousAction::getContinuousActionProperties()
{
	return properties;
}

CActionData *CContinuousAction::getNewActionData()
{
	return dynamic_cast<CActionData *>(new CContinuousActionData(properties));
}

CContinuousActionController::CContinuousActionController(CContinuousAction *contAction, int l_randomControllerMode) : CAgentController(new CActionSet())
{
	this->contAction = contAction;
	actions->add(contAction);
	randomController = NULL;
	this->randomControllerMode = l_randomControllerMode;

	noise = dynamic_cast<CContinuousActionData *>(contAction->getNewActionData());
}

CContinuousActionController::~CContinuousActionController()
{
	delete actions;
	delete noise;
}

void CContinuousActionController::setRandomController(CContinuousActionRandomPolicy *randomController)
{
	this->randomController = randomController;
	addParameters(randomController);

}

CContinuousActionRandomPolicy *CContinuousActionController::getRandomController()
{
	return randomController;
}

void CContinuousActionController::setRandomControllerMode(int l_randomControllerMode)
{
	this->randomControllerMode = l_randomControllerMode;	
}

int CContinuousActionController::getRandomControllerMode()
{
	return randomControllerMode;	
}


CAction *CContinuousActionController::getNextAction(CStateCollection *state, CActionDataSet *dataSet)
{
	assert(dataSet != NULL);

	CContinuousActionData *actionData = dynamic_cast<CContinuousActionData *>(dataSet->getActionData(contAction));

	getNextContinuousAction(state, actionData);

	if (randomController && randomControllerMode == EXTERN_RANDOM_CONTROLLER)
	{
		randomController->getNextContinuousAction(state, noise);
		(*actionData) << (*actionData) + (*noise);
	}

	return contAction;
}

void CContinuousActionController::getNoise(CStateCollection *state, CContinuousActionData *action, CContinuousActionData *l_noise)
{
	int tempRandomMode = randomControllerMode;

	randomControllerMode = NO_RANDOM_CONTROLLER;

	getNextContinuousAction(state, l_noise);

	randomControllerMode = tempRandomMode;

	(*l_noise) << (*action) - (*l_noise);
}

CStaticContinuousAction::CStaticContinuousAction(CContinuousAction *contAction, double *actionValues, double l_maxDistance) : CContinuousAction(contAction->getContinuousActionProperties())
{
	this->contAction = contAction;

	if (actionValues != NULL)
	{
		for (unsigned int i = 0; i < getNumDimensions(); i ++)
		{
			continuousActionData->element(i) = actionValues[i];
		}
	}

	addType(CONTINUOUSSTATICACTION);

	actionData->setIsChangeAble(false);
	
	this->maximumDistance = l_maxDistance;
}

CStaticContinuousAction::~CStaticContinuousAction()
{
}

double CStaticContinuousAction::getMaximumDistance()
{
	return maximumDistance;
}

void CStaticContinuousAction::setContinuousAction(CContinuousActionData *contAction)
{
	*contAction = *continuousActionData;
}

void CStaticContinuousAction::addToContinuousAction(CContinuousActionData *contAction, double factor)
{
	for (unsigned int i = 0; i < properties->getNumActionValues();i++)
	{
		contAction->setActionValue(i, contAction->getActionValue(i) + factor * getActionValue(i));
	}
}

CContinuousAction *CStaticContinuousAction::getContinuousAction()
{
	//contAction->loadActionData(getActionData());
	return contAction;
}

bool CStaticContinuousAction::equals(CAction *action)
{
	if (action->isType(CONTINUOUSACTION) && !action->isType(CONTINUOUSSTATICACTION))
	{
		return getContinuousAction() == action;
	}
	else
	{
		return this == (action);
	}
}

bool CStaticContinuousAction::isSameAction(CAction *action, CActionData *data)
{
	if (action->isType(CONTINUOUSACTION) && !action->isType(CONTINUOUSSTATICACTION))
	{
		CContinuousActionData *lcontData;
		if (data)
		{
			lcontData = dynamic_cast<CContinuousActionData *>(data);	
		}
		else
		{
			lcontData = dynamic_cast<CContinuousActionData *>(action->getActionData());
		}
		if (contAction->getContinuousActionProperties() == getContinuousActionProperties())
		{
			return continuousActionData->getDistance(lcontData) < 0.0001;
		}
		else
		{
			return false;
		}	
	}
	else
	{
		return this == (action);
	}
}


CContinuousActionLinearFA::CContinuousActionLinearFA(CActionSet *contActions, CContinuousActionProperties *properties)
{
	this->actionProperties = properties;
	this->contActions = contActions;

}

CContinuousActionLinearFA::~CContinuousActionLinearFA()
{
}

void CContinuousActionLinearFA::getContinuousAction(CContinuousActionData *contAction, double *actionFactors)
{
	contAction->initData(0.0);
	CActionSet::iterator it = contActions->begin();
	for (unsigned int i = 0; i < contActions->size(); it ++, i++)
	{
		CLinearFAContinuousAction *lFAcontAction = dynamic_cast<CLinearFAContinuousAction *>(*it);

		lFAcontAction->addToContinuousAction(contAction, actionFactors[i]);
	}
	
}


void CContinuousActionLinearFA::getActionFactors(CContinuousActionData *action, double *actionFactors)
{
	double sum = 0.0;
	double val = 0.0;
	unsigned int i = 0;
	CActionSet::iterator it;
	for (i = 0, it = contActions->begin(); it != contActions->end(); it++, i++)
	{
		val = dynamic_cast<CLinearFAContinuousAction *>(*it)->getActionFactor(action);
		sum += val;
		actionFactors[i] = val;
	}
	assert(sum > 0);
	for (i = 0;i < contActions->size() ; i++)
	{
		actionFactors[i] = actionFactors[i] / sum;
	}
}
	
void CContinuousActionLinearFA::getContinuousAction(unsigned int index, CContinuousActionData *action)
{
	assert(index < contActions->size());
	unsigned int i = 0;

	CActionSet::iterator it;
	for (i = 0, it = contActions->begin(); it != contActions->end(), i < index; it++, i++);
	
	if (it != contActions->end())
	{
		dynamic_cast<CLinearFAContinuousAction *>((*it))->setContinuousAction(action);
	}
}

int CContinuousActionLinearFA::getNumContinuousActionFA()
{
	return contActions->size();
}


CLinearFAContinuousAction::CLinearFAContinuousAction(CContinuousAction *contAction, double *actionValues) : CStaticContinuousAction(contAction, actionValues, 0)
{
}

CContinuousRBFAction::CContinuousRBFAction(CContinuousAction *contAction, double *rbfCenter, double *rbfSigma) : CLinearFAContinuousAction(contAction, rbfCenter)
{
	this->rbfSigma = new double[properties->getNumActionValues()];
	
	if (rbfSigma != NULL)
	{	
		memcpy(this->rbfSigma, rbfSigma, sizeof(double) * properties->getNumActionValues());
	}
	else
	{
		memset(this->rbfSigma, 0, sizeof(double) * properties->getNumActionValues());
	}
	maximumDistance = 0;
}

CContinuousRBFAction::~CContinuousRBFAction()
{
	delete rbfSigma;
}

double CContinuousRBFAction::getActionFactor(CContinuousActionData *dynAction)
{
	double malDist = 0.0;
	double faktor = 0.0;

	for (unsigned int i = 0; i < properties->getNumActionValues(); i++)
	{
		malDist += pow((dynAction->getActionValue(i) - getActionValue(i)) / rbfSigma[i], 2);
	}
	malDist = malDist / 2;

	faktor = exp(- malDist);

	return faktor;
}


/*
CContinuousActionVFunction::CContinuousActionVFunction(CStateProperties *properties,  CContinuousActionProperties *actionProp) : CAbstractVFunction(properties)
{
	this->actionProp = actionProp;
	addType(CONTINUOUSVFUNCTION);
}

void CContinuousActionVFunction::updateValue(CStateCollection *state, CContinuousAction *action, double td)
{
	updateValue(state->getState(getStateProperties()), action, td);
}

void CContinuousActionVFunction::setValue(CStateCollection *state, CContinuousAction *action, double qValue)
{
	setValue(state->getState(getStateProperties()), action, qValue);
}

double CContinuousActionVFunction::getValue(CStateCollection *state, CContinuousAction *action)
{
	return getValue(state->getState(getStateProperties()), action);
}
*/


CContinuousActionQFunction::CContinuousActionQFunction(CContinuousAction *contAction) : CGradientQFunction(new CActionSet())
{
	this->contAction = contAction;
	actions->add(contAction);

	addType(CONTINUOUSACTIONQFUNCTION);
}

CContinuousActionQFunction::~CContinuousActionQFunction()
{
	delete actions;
}

void CContinuousActionQFunction::updateCAValue(CStateCollection *, CContinuousActionData *, double )
{
}

void CContinuousActionQFunction::setCAValue(CStateCollection *, CContinuousActionData *, double )
{
} 

void CContinuousActionQFunction::getCAGradient(CStateCollection *, CContinuousActionData *, CFeatureList *) 
{
	
}

void CContinuousActionQFunction::getWeights(double *)
{
}

void CContinuousActionQFunction::setWeights(double *)
{
}

CAction *CContinuousActionQFunction::getMax(CStateCollection *state, CActionSet *, CActionDataSet *actionDatas)
{
	getBestContinuousAction(state, dynamic_cast<CContinuousActionData *>(actionDatas->getActionData(contAction)));
	return contAction;
}

void CContinuousActionQFunction::updateValue(CStateCollection *state, CAction *action, double td, CActionData *data)
{
	if (data != NULL)
	{
		updateCAValue(state, dynamic_cast<CContinuousActionData*>(data), td);
	}
	else
	{
		updateCAValue(state, dynamic_cast<CContinuousAction*>(action)->getContinuousActionData(), td);
	}
}

void CContinuousActionQFunction::setValue(CStateCollection *state, CAction *action, double qValue, CActionData *data)
{
	if (data != NULL)
	{
		setCAValue(state, dynamic_cast<CContinuousActionData*>(data), qValue);
	}
	else
	{
		setCAValue(state, dynamic_cast<CContinuousAction*>(action)->getContinuousActionData(), qValue);
	}
}

double CContinuousActionQFunction::getValue(CStateCollection *state, CAction *action, CActionData *data)
{
	double value = 0.0;
	if (data != NULL)
	{

		value = getCAValue(state, dynamic_cast<CContinuousActionData *>(data));
	}
	else
	{
		value = getCAValue(state, dynamic_cast<CContinuousAction *>(action)->getContinuousActionData());
	}

	if (! mayDiverge && (value > DIVERGENTVFUNCTIONVALUE || value < - DIVERGENTVFUNCTIONVALUE))
	{
		throw new CDivergentQFunctionException("Continuous Action Q-Function", this, state->getState(), value);
	}
	return value;
}


void CContinuousActionQFunction::getGradient(CStateCollection *state, CAction *action, CActionData *data, CFeatureList *gradient)
{
	if (data)
	{
		getCAGradient(state, dynamic_cast<CContinuousActionData *>(data), gradient);
	}
	else
	{
		getCAGradient(state, dynamic_cast<CContinuousActionData *>(action->getActionData()), gradient);
	}
}


CCALinearFAQFunction::CCALinearFAQFunction(CQFunction *qFunction, CContinuousAction *contAction) : CContinuousActionQFunction(contAction), CContinuousActionLinearFA(qFunction->getActions(), contAction->getContinuousActionProperties())
{
	this->qFunction = qFunction;

	actionFactors = new double[qFunction->getNumActions()];
	CAactionValues = new double[qFunction->getNumActions()];
	tempGradient = new CFeatureList();
}

CCALinearFAQFunction::~CCALinearFAQFunction()
{
	delete [] actionFactors;
	delete [] CAactionValues;
	delete tempGradient;
}


void CCALinearFAQFunction::getBestContinuousAction(CStateCollection *state, CContinuousActionData *actionData)
{
	CStaticContinuousAction *action = dynamic_cast<CStaticContinuousAction *>(qFunction->getMax(state, qFunction->getActions()));

	actionData->setData(action->getActionData());

	/*double sum = 0.0;
	double minVal = actionFactors[0];
	for (unsigned int i = 0; i < qFunction->getActions()->size(); i++)
	{
		sum += actionFactors[i];
		if (minVal > actionFactors[i])
		{
			minVal = actionFactors[i]; 
		}
	}

	for (unsigned int i = 0; i < qFunction->getActions()->size(); i++)
	{
		if (sum - minVal * qFunction->getActions()->size() == 0)
		{
			actionFactors[i] = 1 / qFunction->getActions()->size();
		}
		else
		{
			actionFactors[i] = (actionFactors[i] - minVal) / (sum - minVal * qFunction->getActions()->size());
		}
	}*/

//	getContinuousAction(actionData, actionFactors);
}

void CCALinearFAQFunction::updateCAValue(CStateCollection *state, CContinuousActionData *data, double td)
{
	getActionFactors(data, actionFactors);
	CActionSet::iterator it = qFunction->getActions()->begin();
	for (unsigned int i = 0; i < qFunction->getNumActions(); it++, i++)
	{
		if (actionFactors[i] > 0.0001)
		{
			qFunction->getVFunction(*it)->updateValue(state, td * actionFactors[i]);
		}
	}
}

void CCALinearFAQFunction::setCAValue(CStateCollection *state, CContinuousActionData *data, double qValue)
{
	getActionFactors(data, actionFactors);
	CActionSet::iterator it = qFunction->getActions()->begin();
	for (unsigned int i = 0; i < qFunction->getNumActions(); it++, i++)
	{
		DebugPrint('q', "Set LinearFAQFunction: Action %d, Value %f, ActionFactor %f\n", i, qValue, actionFactors[i]);
		if (actionFactors[i] > 0.0001)
		{
			qFunction->getVFunction(*it)->setValue(state, qValue * actionFactors[i]);
		}
	}
}

double CCALinearFAQFunction::getCAValue(CStateCollection *state, CContinuousActionData *data)
{
	getActionFactors(data, actionFactors);
	this->getQFunctionForCA()->getActionValues(state, getQFunctionForCA()->getActions(), CAactionValues);
	double value = 0.0;


	for (unsigned int i = 0; i < qFunction->getNumActions(); i++)
	{
		value += CAactionValues[i] * actionFactors[i];
	}
	return value;
}


CQFunction *CCALinearFAQFunction::getQFunctionForCA()
{
	return qFunction;
}


void CCALinearFAQFunction::updateWeights(CFeatureList *features)
{
	qFunction->updateGradient(features, 1.0);
}

void CCALinearFAQFunction::getCAGradient(CStateCollection *state, CContinuousActionData *action, CFeatureList *gradient)
{
	getActionFactors(action, actionFactors);

	CActionSet::iterator it;
	int i = 0;

	for (it = this->contActions->begin(); it != contActions->end(); it++,i++)
	{
		tempGradient->clear();
		qFunction->getGradient(state, *it, NULL, tempGradient);
		tempGradient->multFactor(actionFactors[i]);

		CFeatureList::iterator itFeat = tempGradient->begin();

		for (; itFeat != tempGradient->end();itFeat++)
		{
			if (fabs((*itFeat)->factor) > 0.00001)
			{
				gradient->update((*itFeat)->featureIndex, (*itFeat)->factor);
			}
		}
	}
}

int CCALinearFAQFunction::getNumWeights()
{
	return qFunction->getNumWeights();
}

CAbstractQETraces* CCALinearFAQFunction::getStandardETraces()
{
	return new CCALinearFAQETraces(this);
}


void CCALinearFAQFunction::getWeights(double *weights)
{
	qFunction->getWeights(weights);
}

void CCALinearFAQFunction::setWeights(double *weights)
{
	qFunction->setWeights(weights);
}

CCALinearFAQETraces::CCALinearFAQETraces(CCALinearFAQFunction *qfunction) : CQETraces(qfunction->getQFunctionForCA())
{
	contQFunc = qfunction;

	actionFactors = new double[qfunction->getNumContinuousActionFA()];
}

CCALinearFAQETraces::~CCALinearFAQETraces()
{
	delete actionFactors;
}

void CCALinearFAQETraces::addETrace(CStateCollection *State, CAction *action, double factor, CActionData *data)
{
	if (action->isType(CONTINUOUSACTION))
	{
		CContinuousActionData *contAction = NULL;
		if (data == NULL)
		{
			contAction = dynamic_cast<CContinuousActionData *>(action->getActionData());
		}
		else
		{
			contAction = dynamic_cast<CContinuousActionData *>(data);
		}
		contQFunc->getActionFactors(contAction, actionFactors);


		std::list<CAbstractVETraces *>::iterator it = vETraces->begin();

		DebugPrint('e', "Adding CALinearFA Etraces: %f factor\n", factor);
		for (unsigned int i = 0; i < qFunction->getNumActions();i++, it++)
		{
			(*it)->addETrace(State, factor * actionFactors[i]);
			DebugPrint('e', "%f ", actionFactors[i]);
		}
		DebugPrint('e',"\n");
	}
}

CContinuousActionPolicy::CContinuousActionPolicy(CContinuousAction *contAction, CActionDistribution *distribution, CAbstractQFunction *continuousActionQFunc, CActionSet *continuousStaticActions) : CContinuousActionController(contAction)
{
	this->distribution = distribution;
	this->continuousActionQFunc = continuousActionQFunc;
	this->continuousStaticActions = continuousStaticActions;
	actionValues = new double[continuousStaticActions->size()];

	//addParameter("CAPolicyMaximumActionDistance", maximumDistance);

	addParameters(distribution);
}

CContinuousActionPolicy::~CContinuousActionPolicy()
{
	delete [] actionValues;
}

void CContinuousActionPolicy::getNextContinuousAction(CStateCollection *state, CContinuousActionData *action)
{
	CActionSet availableActions;
	continuousStaticActions->getAvailableActions(&availableActions, state);
	
	continuousActionQFunc->getActionValues(state, &availableActions, actionValues, NULL);

	DebugPrint('p', "ContinuousActionPolicy ActionValues: ");

	for (unsigned int i = 0; i < availableActions.size(); i++)
	{
		DebugPrint('p', "%f ", actionValues[i]);
	}
	DebugPrint('p',"\n");

	distribution->getDistribution(state, &availableActions, actionValues);

	if (DebugIsEnabled('p'))
	{
		DebugPrint('p', "ContinuousActionPolicy ActionFactors: ");

		for (unsigned int i = 0; i < availableActions.size(); i++)
		{
			DebugPrint('p', "%f ", actionValues[i]);
		}
		DebugPrint('p',"\n");
	}

	CActionSet::iterator it = availableActions.begin();

	int actionIndex = CDistributions::getSampledIndex(actionValues, availableActions.size());

	ColumnVector *sampledActionData = NULL;

	for (int i = 0; i < actionIndex; i++, it++);

	CStaticContinuousAction *sampledAction = dynamic_cast<CStaticContinuousAction *>(*it);
	sampledActionData = sampledAction->getContinuousActionData();

	it = availableActions.begin();

	action->initData(0.0);

	//double maximumDistance = getParameter("CAPolicyMaximumActionDistance");

	//if (maximumDistance > 0)
	//{
		double sum = 0.0;

		for (int i = 0; it != availableActions.end(); it ++, i++)
		{
			CStaticContinuousAction	 *contAction = dynamic_cast<CLinearFAContinuousAction *>(*it);

			if (i == actionIndex || contAction->getContinuousActionData()->getDistance(sampledActionData) < sampledAction->getMaximumDistance())
			{
				contAction->addToContinuousAction(action, actionValues[i]);
				sum += actionValues[i];
			}
		}
		(*action) *= (1.0 / sum);
	//}
	//else
	//{
	//	action->setVector(sampledActionData);
	//}

	if (DebugIsEnabled('p'))
	{
		DebugPrint('p', "ContinuousActionPolicy Calculated Action: ");

		action->saveASCII(DebugGetFileHandle('p'));
		
		DebugPrint('p',"\n");
	}
}

CContinuousActionRandomPolicy::CContinuousActionRandomPolicy(CContinuousAction *action, double sigma, double alpha) : CContinuousActionController(action)
{
	addParameter("RandomPolicySigma",sigma);
	addParameter("RandomPolicySmoothFactor", alpha);

	lastNoise = new ColumnVector(action->getNumDimensions());
	*lastNoise = 0.0;
	
	currentNoise = new ColumnVector(action->getNumDimensions());
	*currentNoise = 0.0;
	
	this->sigma = sigma;
	this->alpha = alpha;
}

CContinuousActionRandomPolicy::~CContinuousActionRandomPolicy()
{
	delete lastNoise;
	delete currentNoise;
}

void CContinuousActionRandomPolicy::onParametersChanged()
{
	CContinuousActionController::onParametersChanged();
	
	sigma = getParameter("RandomPolicySigma");
	alpha = getParameter("RandomPolicySmoothFactor");		
}

void CContinuousActionRandomPolicy::newEpisode()
{
	*lastNoise = 0.0;
	
	for (unsigned int i = 0; i < getContinuousAction()->getNumDimensions(); i++)
	{
		double randValue = 0.0;
		if (sigma > 0.00001)
		{
			randValue = CDistributions::getNormalDistributionSample(0.0, sigma);
		}
		DebugPrint('p', "Random  Controller : %f\n", randValue);
		currentNoise->element(i) = currentNoise->element(i) * alpha + randValue;
	}
}


ColumnVector *CContinuousActionRandomPolicy::getCurrentNoise()
{
	return currentNoise;
}

ColumnVector *CContinuousActionRandomPolicy::getLastNoise()
{
	return lastNoise;
}

void CContinuousActionRandomPolicy::nextStep(CStateCollection *, CAction *, CStateCollection *)
{
//	double sigma = getParameter("RandomPolicySigma");
//	double alpha = getParameter("RandomPolicySmoothFactor");

	for (unsigned int i = 0; i < getContinuousAction()->getNumDimensions(); i++)
	{
		double randValue = 0.0;
		if (sigma > 0.00001)
		{
			randValue = CDistributions::getNormalDistributionSample(0.0, sigma);
		}
		DebugPrint('p', "Random  Controller : %f, %f\n", lastNoise->element(i), randValue);

		lastNoise->element(i) = currentNoise->element(i);
		currentNoise->element(i) = currentNoise->element(i) * alpha + randValue;
	}
}

void CContinuousActionRandomPolicy::getNextContinuousAction(CStateCollection *, CContinuousActionData *action)
{
	for (int i = 0; i < action->nrows(); i++)
	{
		action->setActionValue(i, currentNoise->element(i));
	}
}

CContinuousActionAddController::CContinuousActionAddController(CContinuousAction *action) : CContinuousActionController(action)
{
	this->controllers = new std::list<CContinuousActionController *>();
	this->controllerWeights = new::map<CContinuousActionController *, double>();

	actionValues = new ColumnVector(action->getNumDimensions());
}

CContinuousActionAddController::~CContinuousActionAddController()
{
	delete controllers;
	delete controllerWeights;

	delete actionValues;
}

void CContinuousActionAddController::getNextContinuousAction(CStateCollection *state, CContinuousActionData *action)
{
	std::list<CContinuousActionController *>::iterator it = controllers->begin();

	*actionValues = 0.0;
	double weightsSum = 0.0;
	for (; it != controllers->end();it ++)
	{
		action->initData(0.0);
		(*it)->getNextContinuousAction(state, action);
		double weight = getControllerWeight(*it);
		weightsSum += weight;
		(*action) *= weight;
		(*actionValues) << (*action) + (*actionValues);
	}
	(*action) << (*actionValues) / weightsSum;
}

void CContinuousActionAddController::addContinuousActionController(CContinuousActionController *controller, double weight)
{
	controllers->push_back(controller);
	(*controllerWeights)[controller] = weight;
}

void CContinuousActionAddController::setControllerWeight(CContinuousActionController *controller, double weight)
{
	(*controllerWeights)[controller] = weight;
}

double CContinuousActionAddController::getControllerWeight(CContinuousActionController *controller)
{
	return (*controllerWeights)[controller];
}

