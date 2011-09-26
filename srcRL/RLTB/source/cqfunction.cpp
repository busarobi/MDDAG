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
#include "cqfunction.h"

#include "caction.h"
#include "cactionstatistics.h"
#include "cenvironmentmodel.h"
#include "cvfunction.h"
#include "cepisode.h"
#include "crewardfunction.h"
#include "cqetraces.h"
#include "ril_debug.h"
#include "cstatecollection.h"
#include "cstate.h"
#include "cstateproperties.h"
#include "cdynamicprogramming.h"
#include "cutility.h"
#include "ctheoreticalmodel.h"
#include "cstatemodifier.h"

#include <sstream>
#include <assert.h>

CAbstractQFunction::CAbstractQFunction(CActionSet *actions) : CActionObject(actions, false)
{
	type = 0;
	mayDiverge = false;
}

CAbstractQFunction::~CAbstractQFunction()
{
}

void CAbstractQFunction::setValue(CStateCollection *, CAction *, double , CActionData *)
{
}

void CAbstractQFunction::getActionValues(CStateCollection *stateCol,  CActionSet *actions, double *actionValues, CActionDataSet *data)
{
	CActionSet::iterator it = actions->begin();
	for (unsigned int i = 0; it != actions->end(); it++, i++)
	{
		if (data)
		{
			actionValues[i] = this->getValue(stateCol, *it, data->getActionData(*it));
		}
		else
		{
			actionValues[i] = this->getValue(stateCol, *it);
		}
	}
}

double CAbstractQFunction::getMaxValue(CStateCollection *state, CActionSet *availableActions)
{
	assert(availableActions->size() > 0);

	double max, value;
	double *actionValues = new double[availableActions->size()];
	

	getActionValues(state, availableActions, actionValues);

    max = actionValues[0];

	for (unsigned int i = 1; i < availableActions->size(); i++)
	{
        value = actionValues[i];
        if ( max < value)
		{
			max = value;
		}
	}

	delete [] actionValues;

	return max;
}


CAction* CAbstractQFunction::getMax(CStateCollection* stateCol, CActionSet *availableActions, CActionDataSet *)
{
	assert(availableActions->size() > 0);

	double max, value;
	double *actionValues = new double[availableActions->size()];

	
	CActionSet::iterator it = availableActions->begin();
	CActionSet *max_list = new CActionSet();

	getActionValues(stateCol, availableActions, actionValues);


    max = actionValues[0];
	max_list->push_back(*it++);

	for (unsigned int i = 1; it != availableActions->end(); it++, i++)
	{
        value = actionValues[i];
        if ( max < value)
		{
			max_list->clear();
			max = value;
			max_list->push_back(*it);
		}
     	else if (max == value)
		{
			max_list->push_back(*it);
		}							
	}

	//int index = rand() % max_list->size();
	int index = 0;
	CAction *action = max_list->get(index);

	DebugPrint('q', "ActionValues: ");
	for (unsigned int j = 0; j < availableActions->size(); j++)
	{
		DebugPrint('q', "%f ", actionValues[j]);
	}
	DebugPrint('q', "\nMax: %d\n", actions->getIndex(action));

	delete max_list;
	delete [] actionValues;
	return action;
}

void CAbstractQFunction::getStatistics(CStateCollection* state, CAction* action, CActionSet *availableActions, CActionStatistics *statistics)
{
	assert(availableActions->size() > 0);
    assert(statistics != NULL);
	
	double *actionValues = new double[availableActions->size()];
	// get Q-Values
	getActionValues(state, availableActions, actionValues);
	// Hier wird die WK-Verteilung erstellt, aus der die Statistik berechnet wird
	//transform: smallest Value = 0, Value sum = 1;
	CDistributions::getS1L0Distribution(actionValues, availableActions->size());

	statistics->action = action;
    statistics->equal = 0;
	statistics->superior = 0;
	statistics->probability = actionValues[availableActions->getIndex(action)];
    	
	for (unsigned int i = 0; i < availableActions->size(); i++)
	{
		if (statistics->probability == actionValues[i]) statistics->equal++;
		if (statistics->probability < actionValues[i]) statistics->superior++;
	}
	delete [] actionValues;
}

int CAbstractQFunction::getType()
{
	return type;
}

void CAbstractQFunction::addType(int Type)
{
	type = type | Type;	
}

bool CAbstractQFunction::isType(int type)
{
	return (this->type & type) > 0;
}


void CAbstractQFunction::saveData(FILE *file)
{
    fprintf(file, "Q-Function:\n");
    fprintf(file, "Actions: %d\n\n", actions->size());
}

void CAbstractQFunction::loadData(FILE *file)
{
	unsigned int buf = 0;
    assert(fscanf(file, "Q-Function:\n") == 0);
	assert(fscanf(file, "Actions: %d\n\n", &buf) == 1 &&  buf == actions->size());
    assert(fscanf(file, "\n") == 0);
}

CQFunctionSum::CQFunctionSum(CActionSet *actions) : CAbstractQFunction(actions)
{
	qFunctions = new std::map<CAbstractQFunction *, double>;
}

CQFunctionSum::~CQFunctionSum()
{
	delete qFunctions;
}

double CQFunctionSum::getValue(CStateCollection *state, CAction *action, CActionData *data)
{
	std::map<CAbstractQFunction *, double>::iterator it = qFunctions->begin();

	double sum = 0.0;
	for (;it != qFunctions->end(); it ++)
	{
		CAbstractQFunction *qFunc = (*it).first;
		if (qFunc->getActions()->isMember(action))
		{
			sum += (*it).second * qFunc->getValue(state, action, data);
		}
	}
	return sum;
}


double CQFunctionSum::getQFunctionFactor(CAbstractQFunction *qFunction)
{
	return (*qFunctions)[qFunction];
}

void CQFunctionSum::setQFunctionFactor(CAbstractQFunction *qFunction, double factor)
{
	(*qFunctions)[qFunction] = factor;
}

void CQFunctionSum::addQFunction(CAbstractQFunction *qFunction, double factor)
{
	addParameters(qFunction);
	(*qFunctions)[qFunction] = factor;
}

void CQFunctionSum::removeQFunction(CAbstractQFunction *qFunction)
{
	std::map<CAbstractQFunction *, double>::iterator it = qFunctions->find(qFunction);

	qFunctions->erase(it);
}

void CQFunctionSum::normFactors(double factor)
{
	std::map<CAbstractQFunction *, double>::iterator it = qFunctions->begin();

	double sum = 0.0;
	for (;it != qFunctions->end(); it ++)
	{
		sum += (*it).second;
	}
	for (;it != qFunctions->end(); it ++)
	{
		(*it).second *= factor / sum;
	}
}


CDivergentQFunctionException::CDivergentQFunctionException(string qFunctionName, CAbstractQFunction *qFunction, CState *state, double value) : CMyException(102, "DivergentQFunction")
{
	this->qFunction = qFunction;
	this->qFunctionName = qFunctionName;
	this->state = state;
	this->value = value;
}

string CDivergentQFunctionException::getInnerErrorMsg()
{
	stringstream stream;

	stream << qFunctionName.c_str() << " diverges (value = " << value << ", |value| > 1000000).";

	return stream.str();
}

CGradientQFunction::CGradientQFunction(CActionSet *actions) : CAbstractQFunction(actions)
{
	addType(GRADIENTQFUNCTION);

	this->localGradientQFunctionFeatures = new CFeatureList();
}

CGradientQFunction::~CGradientQFunction()
{
	delete localGradientQFunctionFeatures;
}

void CGradientQFunction::updateValue(CStateCollection *state, CAction *action,double td, CActionData *data)
{
	localGradientFeatureBuffer->clear();
	getGradient(state, action, data, localGradientFeatureBuffer);

	updateGradient(localGradientFeatureBuffer, td);
}
/*
CGradientDelayedUpdateQFunction::CGradientDelayedUpdateQFunction(CGradientQFunction *qFunction) :  CGradientQFunction(qFunction->getActions()), CGradientDelayedUpdateFunction(qFunction)
{
	this->qFunction = qFunction;
}

double CGradientDelayedUpdateQFunction::getValue(CStateCollection *state, CAction *action, CActionData *data )
{
	return qFunction->getValue(state, action, data);
}

void CGradientDelayedUpdateQFunction::getGradient(CStateCollection *state, CAction *action, CActionData *data, CFeatureList *gradientFeatures)
{
	qFunction->getGradient(state, action, data, gradientFeatures);
}

*/
CAbstractQETraces *CGradientQFunction::getStandardETraces()
{
	return new CGradientQETraces(this);
}

CQFunction::CQFunction(CActionSet *act) : CGradientQFunction(act)
{
	this->vFunctions = new std::map<CAction *, CAbstractVFunction *>();
}

CQFunction::~CQFunction()
{
	delete vFunctions;
}

void CQFunction::updateValue(CState *state, CAction *action, double td, CActionData *)
{
	assert((*vFunctions)[action]);

	(*vFunctions)[action]->updateValue(state, td);
}

void CQFunction::setValue(CState *state, CAction *action, double value, CActionData *)
{
	assert((*vFunctions)[action]);

	(*vFunctions)[action]->setValue(state, value);
}

double CQFunction::getValue(CState *state, CAction *action, CActionData *data)
{
	double value = 0.0;

	if ((*vFunctions)[action] == NULL)
	{
		printf("No V-Function found for action : ");
		if (data)
		{
			data->saveASCII(stdout);
		}
		else
		{
			action->getActionData()->saveASCII(stdout);
		}
		printf("\n");
		assert(false);
	}
	value = (*vFunctions)[action]->getValue(state);

	return value;
}

void CQFunction::updateValue(CStateCollection *state, CAction *action, double td, CActionData *)
{
	assert((*vFunctions)[action]);

	(*vFunctions)[action]->updateValue(state, td);
}

void CQFunction::setValue(CStateCollection *state, CAction *action, double value, CActionData *)
{
	assert((*vFunctions)[action]);

	(*vFunctions)[action]->setValue(state, value);
}

double CQFunction::getValue(CStateCollection *state, CAction *action, CActionData *data)
{
	double value = 0.0;

	if ((*vFunctions)[action] == NULL)
	{
		printf("No V-Function found for action %d : ", actions->getIndex(action));
		if (data)
		{
			data->saveASCII(stdout);
		}
		else
		{
			action->getActionData()->saveASCII(stdout);
		}
		printf("\n");
		assert(false);
	}

	value = (*vFunctions)[action]->getValue(state);

	return value;
}


CAbstractVFunction *CQFunction::getVFunction(CAction *action)
{
	return (*vFunctions)[action];
}

CAbstractVFunction *CQFunction::getVFunction(int index)
{
	return (*vFunctions)[actions->get(index)];
}


void CQFunction::setVFunction(CAction *action, CAbstractVFunction *vfunction, bool bDelete)
{
	if (bDelete && (*vFunctions)[action] != NULL)
	{
		delete (*vFunctions)[action];
	}
	(*vFunctions)[action] = vfunction;

	if (!vfunction->isType(GRADIENTVFUNCTION))
	{
		type = type & (~ GRADIENTQFUNCTION);
	}

	addParameters(vfunction);
}

void CQFunction::setVFunction(int index, CAbstractVFunction *vfunction, bool bDelete)
{
	setVFunction(actions->get(index), vfunction, bDelete);
}

int CQFunction::getNumVFunctions()
{
	return vFunctions->size();
}

void CQFunction::saveData(FILE *file)
{
	CAbstractQFunction::saveData(file);

	CActionSet::iterator it = actions->begin();

	for (; it != actions->end(); it++)
	{
		(*vFunctions)[(*it)]->saveData(file);
	}
}

void CQFunction::loadData(FILE *file)
{
	CAbstractQFunction::loadData(file);

	CActionSet::iterator it = actions->begin();

	for (; it != actions->end(); it++)
	{
		(*vFunctions)[(*it)]->loadData(file);
	}
    //assert(fscanf(file, "Lambda: %f\n", &lambda) == 1);
    assert(fscanf(file, "\n") == 0);
}

void CQFunction::printValues()
{
	CAbstractQFunction::printValues();

	CActionSet::iterator it = actions->begin();

	for (; it != actions->end(); it++)
	{
		(*vFunctions)[(*it)]->printValues();
	}
}

CAbstractQETraces *CQFunction::getStandardETraces()
{
	return new CQETraces(this);
}

void CQFunction::resetData()
{
	CActionSet::iterator it = actions->begin();

	for (; it != actions->end(); it++)
	{
		(*vFunctions)[(*it)]->resetData();
	}
}

void CQFunction::copy(CLearnDataObject *qFunc)
{
	CActionSet::iterator it = actions->begin();

	CQFunction *qFunction = dynamic_cast<CQFunction *>(qFunc);

	for (; it != actions->end(); it++)
	{
		(*vFunctions)[(*it)]->copy(qFunction->getVFunction( *it));
	}

}

/*CStateProperties *CQFunction::getGradientCalculator(CAction *action)
{
    int index = actions->getIndex(action);

	if ((*vFunctions)[action]->isType(GRADIENTVFUNCTION))
	{
		CGradientVFunction *gradVFunc = dynamic_cast<CGradientVFunction *>((*vFunctions)[action]);

		return gradVFunc->getGradientCalculator();
	}
	else
	{
		return NULL;
	}
}*/

void CQFunction::getGradient(CStateCollection *stateCol, CAction *action, CActionData *, CFeatureList *gradient)
{
	if (gradient->size() > 0)
	{
		printf("Warning : CQFunction... getting Gradient, gradient list not empty!!\n");
	}
	
	if ((*vFunctions)[action]->isType(GRADIENTVFUNCTION))
	{
		CGradientVFunction *gradVFunc = dynamic_cast<CGradientVFunction *>((*vFunctions)[action]);
		gradVFunc->getGradient(stateCol, gradient);
		
		
		gradient->addIndexOffset(getWeightsOffset(action));
	}
}

void CQFunction::updateWeights(CFeatureList *features)
{
	unsigned int featureBegin = 0;
	unsigned int featureEnd = 0;

	if (DebugIsEnabled('q'))
	{
		DebugPrint('q', "Updating Features: ");
		features->saveASCII(DebugGetFileHandle('q'));
		DebugPrint('q', "\n");
	}
	
	if(isType(GRADIENTQFUNCTION))
	{
		std::map<CAction *, CAbstractVFunction *>::iterator it = vFunctions->begin();
		CFeatureList::iterator itFeat;

		for (int i = 0; it != vFunctions->end();it++, i++)
		{
			CGradientVFunction *gradVFunction = dynamic_cast<CGradientVFunction *>((*it).second);
			featureEnd += gradVFunction->getNumWeights();
	
			localGradientQFunctionFeatures->clear();

			for (itFeat = features->begin(); itFeat != features->end(); itFeat++)
			{
				if ((*itFeat)->featureIndex >= featureBegin && (*itFeat)->featureIndex < featureEnd)
				{
					localGradientQFunctionFeatures->update((*itFeat)->featureIndex - featureBegin, (*itFeat)->factor);
				}
			}

			if (DebugIsEnabled('q'))
			{
				DebugPrint('q', "Updating Features for Action %d: ",i);
				localGradientQFunctionFeatures->saveASCII(DebugGetFileHandle('q'));
				DebugPrint('q', "\n");
			}
			gradVFunction->updateGradient(localGradientQFunctionFeatures, 1.0);

			featureBegin += gradVFunction->getNumWeights();
		}
	}
}

int CQFunction::getNumWeights()
{
	int nparams = 0;
	std::map<CAction *, CAbstractVFunction *>::iterator it = vFunctions->begin();
	for (; it != vFunctions->end();it++)
	{
		CGradientVFunction *gradVFunction = dynamic_cast<CGradientVFunction *>((*it).second);
		nparams += gradVFunction->getNumWeights();
	}
	return nparams;
}

int CQFunction::getWeightsOffset(CAction *action)
{
	int nparams = 0;
	std::map<CAction *, CAbstractVFunction *>::iterator it = vFunctions->begin();
	for (; it != vFunctions->end();it++)
	{
		if ((*it).first == action)
		{	
			break;
		}
		CGradientVFunction *gradVFunction = dynamic_cast<CGradientVFunction *>((*it).second);
		nparams += gradVFunction->getNumWeights();
	}
	return nparams;
}

void CQFunction::getWeights(double *weights)
{
	double *vFuncWeights = weights;
	std::map<CAction *, CAbstractVFunction *>::iterator it = vFunctions->begin();
	for (; it != vFunctions->end();it++)
	{
		CGradientVFunction *gradVFunction = dynamic_cast<CGradientVFunction *>((*it).second);
		gradVFunction->getWeights(vFuncWeights);
		vFuncWeights += gradVFunction->getNumWeights();
	}
}

void CQFunction::setWeights(double *weights)
{
	double *vFuncWeights = weights;
	std::map<CAction *, CAbstractVFunction *>::iterator it = vFunctions->begin();
	for (; it != vFunctions->end();it++)
	{
		CGradientVFunction *gradVFunction = dynamic_cast<CGradientVFunction *>((*it).second);
		gradVFunction->setWeights(vFuncWeights);
		vFuncWeights += gradVFunction->getNumWeights();
	}
}

CQFunctionFromStochasticModel::CQFunctionFromStochasticModel(CFeatureVFunction *vfunction, CAbstractFeatureStochasticModel *model, CFeatureRewardFunction *rewardfunction) : CAbstractQFunction(model->getActions()), CStateObject(vfunction->getStateProperties())
{
	this->vfunction = vfunction;
	this->model = model;
	this->discretizer = vfunction->getStateProperties();
	this->rewardfunction = rewardfunction;

	discState = new CState(new CStateProperties(0,1,DISCRETESTATE));
	discState->getStateProperties()->setDiscreteStateSize(0, vfunction->getNumFeatures());

	addParameter("DiscountFactor", 0.95);
}

CQFunctionFromStochasticModel::~CQFunctionFromStochasticModel()
{
	delete discState->getStateProperties();
	delete discState;
}

double CQFunctionFromStochasticModel::getValue(CStateCollection *state, CAction *action, CActionData *)
{
	double value = getValue(state->getState(properties), action);

	return value;
}

double CQFunctionFromStochasticModel::getValue(int state, CAction *action, CActionData *)
{
	discState->setDiscreteState(0, state);

	double value = CDynamicProgramming::getActionValue(model, this->rewardfunction, this->vfunction, discState, action, getParameter("DiscountFactor"));
	
	return value;
}


double CQFunctionFromStochasticModel::getValue(CState *featState, CAction *action, CActionData *)
{
	double stateValue = 0.0;
		
	int type = featState->getStateProperties()->getType() & (DISCRETESTATE | FEATURESTATE);
	switch (type)
	{
		case DISCRETESTATE:
		{
			stateValue = CDynamicProgramming::getActionValue(model, this->rewardfunction, this->vfunction, featState, action, getParameter("DiscountFactor"));
			break;
		}
		case FEATURESTATE:
		{
			for (unsigned int i = 0; i < featState->getNumContinuousStates(); i++)
			{
				stateValue += getValue(featState->getDiscreteState(i), action) * featState->getContinuousState(i);
			}
			break;
		}
		default:
		{
			stateValue = getValue(featState->getDiscreteStateNumber(), action);
		}
	}
	return stateValue;
}


/*
CQTable::CQTable(CActionSet *actions, CAbstractStateDiscretizer *discretizer) : CQFunction(actions), CAbstractQFunction(actions)
{
	this->discretizer = discretizer;
	init(discretizer->getDiscreteStateSize());
}


void CQTable::init(int states)
{
	this->states = states;
	CVTable *table = NULL;

	for (CActionSet::iterator it = actions->begin(); it != actions->end(); it++)
	{
		if (discretizer != NULL) table = new CVTable(discretizer);
		else table = new CVTable(discretizer);
		this->setVFunction(*it, table);
	}
}
	
CQTable::~CQTable()
{
	for (std::list<CAbstractVFunction *>::iterator it = vFunctions->begin(); it != vFunctions->end(); it ++)
	{
		delete *it;
	}
}

void CQTable::setDiscretizer(CAbstractStateDiscretizer *discretizer)
{
	assert(discretizer == NULL || discretizer->getDiscreteStateSize() == states);

	this->discretizer = discretizer;
}

CAbstractStateDiscretizer *CQTable::getDiscretizer()
{
	return discretizer;
}


int CQTable::getNumStates()
{
	return states;
}*/

CFeatureQFunction::CFeatureQFunction(CActionSet *actions, CStateProperties *discretizer) : CQFunction(actions)
{
	this->discretizer = discretizer;
	this->features = discretizer->getDiscreteStateSize();
	init();
}

CFeatureQFunction::CFeatureQFunction(CFeatureVFunction *vfunction, CAbstractFeatureStochasticModel *model,  CFeatureRewardFunction *rewardFunction, double gamma) : CQFunction(model->getActions())
{
	this->discretizer = (CStateModifier *) vfunction->getStateProperties();
	this->features = discretizer->getDiscreteStateSize();

	init();

	initVFunctions(vfunction, model, rewardFunction, gamma);
}

void CFeatureQFunction::init()
{
	CFeatureVFunction *vFunction = NULL;

	featureVFunctions = new std::list<CFeatureVFunction *>();
	for (CActionSet::iterator it = actions->begin(); it != actions->end(); it++)
	{
		vFunction = new CFeatureVFunction(discretizer);
		featureVFunctions->push_back(vFunction);
		this->setVFunction(*it, vFunction);
	}
}
	
CFeatureQFunction::~CFeatureQFunction()
{
	for (std::list<CFeatureVFunction *>::iterator it = featureVFunctions->begin(); it != featureVFunctions->end(); it ++)
	{
		delete *it;
	}
	delete featureVFunctions;
}

void CFeatureQFunction::setFeatureCalculator(CStateModifier *discretizer)
{
	assert(discretizer == NULL || discretizer->getDiscreteStateSize() == features);

	this->discretizer = discretizer;
}

CStateProperties *CFeatureQFunction::getFeatureCalculator()
{
	return discretizer;
}

int CFeatureQFunction::getNumFeatures()
{
	return features;
}

void CFeatureQFunction::initVFunctions(CFeatureVFunction *vfunction, CAbstractFeatureStochasticModel *model, CFeatureRewardFunction *rewardFunction, double gamma)
{
	std::list<CAction *>::iterator itAction;

	CState *discState = new CState(new CStateProperties(0,1));

	for (int feature = 0; feature < getNumFeatures(); feature ++)
	{
		discState->setDiscreteState(0, feature);
		for (itAction = actions->begin(); itAction != actions->end(); itAction ++)
		{
			((CFeatureVFunction *)(*vFunctions)[*itAction])->setFeature(feature, CDynamicProgramming::getActionValue(model, rewardFunction, vfunction, discState, *itAction, gamma));
		}
	}
	delete discState->getStateProperties();
	delete discState;
}


void CFeatureQFunction::updateValue(CFeature *state, CAction *action, double td, CActionData *)
{
	((CFeatureVFunction *) getVFunction(action))->updateFeature(state, td);
}

void CFeatureQFunction::setValue(int state, CAction *action, double qValue, CActionData *)
{
	((CFeatureVFunction *) getVFunction(action))->setFeature(state, qValue);
}

double CFeatureQFunction::getValue(int feature, CAction *action, CActionData *)
{
	return ((CFeatureVFunction *) getVFunction(action))->getFeature(feature);
}


void CFeatureQFunction::saveFeatureActionValueTable(FILE *stream)
{
	fprintf(stream, "Q-FeatureActionValue Table\n");
	CActionSet::iterator it;

	for (unsigned int i = 0; i < discretizer->getDiscreteStateSize(); i++)
	{
		fprintf(stream,"State %d: ", i);
		for (it = actions->begin(); it != actions->end(); it++)
		{
			fprintf(stream,"%f ", ((CFeatureVFunction *) (*vFunctions)[*it])->getFeature(i));
		}
		fprintf(stream, "\n");
	}
}

void CFeatureQFunction::saveFeatureActionTable(FILE *stream)
{
	fprintf(stream, "Q-FeatureAction Table\n");
	CActionSet::iterator it;
	double max = 0.0;
	unsigned int maxIndex = 0;

	for (unsigned int i = 0; i < discretizer->getDiscreteStateSize(); i++)
	{
		fprintf(stream,"State %d: ", i);
		
		it = actions->begin();
		max = ((CFeatureVFunction *)(*vFunctions)[*it])->getFeature(i);
		it ++;
		maxIndex = 0;
		for (unsigned int j = 1; it != actions->end(); it++, j++)
		{
			double qValue = ((CFeatureVFunction*)(*vFunctions)[*it])->getFeature(i);
			if (max < qValue)
			{
				max  = qValue;
				maxIndex = j;
			}
		}
		fprintf(stream, "%d", maxIndex);
		fprintf(stream, "\n");
	}
}


CComposedQFunction::CComposedQFunction() : CGradientQFunction(new CActionSet())
{
	this->qFunctions = new std::list<CAbstractQFunction *>();
	//gradientFeatures = new CFeatureList();
}

CComposedQFunction::~CComposedQFunction()
{
	delete qFunctions;
	//delete gradientFeatures;
}

void CComposedQFunction::saveData(FILE *file)
{
	std::list<CAbstractQFunction *>::iterator it = qFunctions->begin();
	fprintf(file, "Composed QFunction (containing %d QFunctions)\n", qFunctions->size());
	for (; it != qFunctions->begin(); it++)
	{
		(*it)->saveData(file);
	}
}

void CComposedQFunction::loadData(FILE *file)
{
	int buf = 0;
	std::list<CAbstractQFunction *>::iterator it = qFunctions->begin();
	fscanf(file, "Composed QFunction (containing %d QFunctions)\n", &buf);
	for (; it != qFunctions->begin(); it++)
	{
		(*it)->loadData(file);
	}
}

void CComposedQFunction::printValues()
{
	std::list<CAbstractQFunction *>::iterator it = qFunctions->begin();
	for (; it != qFunctions->begin(); it++)
	{
		(*it)->printValues();
	}
}


void CComposedQFunction::getStatistics(CStateCollection *state, CAction *action, CActionSet *actions, CActionStatistics* statistics)
{
	std::list<CAbstractQFunction *>::iterator it = qFunctions->begin();
	for (; it != qFunctions->begin(); it++)
	{
		if ((*it)->getActions()->isMember(action))
		{
			(*it)->getStatistics(state, action, actions, statistics);
		}
	}
}

void CComposedQFunction::updateValue(CStateCollection *state, CAction *action, double td, CActionData *data)
{
	std::list<CAbstractQFunction *>::iterator it = qFunctions->begin();
	for (; it != qFunctions->begin(); it++)
	{
		if ((*it)->getActions()->isMember(action))
		{
			(*it)->updateValue(state, action, td, data);
		}
	}
}


void CComposedQFunction::setValue(CStateCollection *state, CAction *action, double qValue, CActionData *data)
{
	std::list<CAbstractQFunction *>::iterator it = qFunctions->begin();
	for (; it != qFunctions->begin(); it++)
	{
		if ((*it)->getActions()->isMember(action))
		{
			(*it)->setValue(state, action, qValue, data);
		}
	}
}

double CComposedQFunction::getValue(CStateCollection *state, CAction *action, CActionData *data)
{
	std::list<CAbstractQFunction *>::iterator it = qFunctions->begin();
	for (; it != qFunctions->begin(); it++)
	{
		if ((*it)->getActions()->isMember(action))
		{
			return (*it)->getValue(state, action, data);
		}
	}
	return 0;
}


void CComposedQFunction::addQFunction(CAbstractQFunction *qFunction)
{
	qFunctions->push_back(qFunction);

	actions->add(qFunction->getActions());

	if (!qFunction->isType(GRADIENTQFUNCTION))
	{
		type = type & (~ GRADIENTQFUNCTION);
	}
	addParameters(qFunction);
}


std::list<CAbstractQFunction *> *CComposedQFunction::getQFunctions()
{
	return qFunctions;
}

int CComposedQFunction::getNumQFunctions()
{
	return qFunctions->size();
}

CAbstractQETraces *CComposedQFunction::getStandardETraces()
{
	return new CComposedQETraces(this);
}


void CComposedQFunction::getGradient(CStateCollection *stateCol, CAction *action, CActionData *data, CFeatureList *gradient)
{
	std::list<CAbstractQFunction *>::iterator it = qFunctions->begin();

	for (; it != qFunctions->end(); it++)
	{
		if ((*it)->getActions()->isMember(action))
		{	
			if ((*it)->isType(GRADIENTQFUNCTION))
			{
				CGradientQFunction *gradQFunc = dynamic_cast<CGradientQFunction *>(*it);
				gradQFunc->getGradient(stateCol, action, data, gradient);
				gradient->addIndexOffset(getWeightsOffset(action));
			}	
		}
	}
}



void CComposedQFunction::updateWeights(CFeatureList *features)
{
	unsigned int featureBegin = 0;
	unsigned int featureEnd = 0;
	if(isType(GRADIENTQFUNCTION))
	{
		std::list<CAbstractQFunction *>::iterator it = qFunctions->begin();
		CFeatureList::iterator itFeat;

		for (; it != qFunctions->end();it++)
		{
			CGradientQFunction *gradQFunction = dynamic_cast<CGradientQFunction *>(*it);
			featureEnd += gradQFunction->getNumWeights();

			localGradientQFunctionFeatures->clear();

			for (itFeat = features->begin(); itFeat != features->end(); it++)
			{
				if ((*itFeat)->featureIndex >= featureBegin && (*itFeat)->featureIndex < featureEnd)
				{
					localGradientQFunctionFeatures->add(*itFeat);
				}
			}
			gradQFunction->updateGradient(localGradientQFunctionFeatures);
		}
	}
}

int CComposedQFunction::getNumWeights()
{
	int nparams = 0;
	std::list<CAbstractQFunction *>::iterator it = qFunctions->begin();
	for (; it != qFunctions->end();it++)
	{
		CGradientQFunction *gradQFunction = dynamic_cast<CGradientQFunction *>(*it);
		nparams += gradQFunction->getNumWeights();
	}
	return nparams;
}

int CComposedQFunction::getWeightsOffset(CAction *action)
{

	int nparams = 0;
	std::list<CAbstractQFunction *>::iterator it = qFunctions->begin();
	for (; it != qFunctions->end();it++)
	{
		CGradientQFunction *gradQFunction = dynamic_cast<CGradientQFunction *>(*it);

		if ((*it)->getActions()->isMember(action))
		{
			break;
		}
		nparams += gradQFunction->getNumWeights();
	}
	return nparams;
}

void CComposedQFunction::getWeights(double *weights)
{
	double *qFuncWeights = weights;
	std::list<CAbstractQFunction *>::iterator it = qFunctions->begin();
	for (; it != qFunctions->end();it++)
	{
		CGradientQFunction *gradQFunction = dynamic_cast<CGradientQFunction *>(*it);
		gradQFunction->getWeights(qFuncWeights);
		qFuncWeights += gradQFunction->getNumWeights();
	}
}

void CComposedQFunction::setWeights(double *weights)
{
	double *qFuncWeights = weights;
	std::list<CAbstractQFunction *>::iterator it = qFunctions->begin();
	for (; it != qFunctions->end();it++)
	{
		CGradientQFunction *gradQFunction = dynamic_cast<CGradientQFunction *>(*it);
		gradQFunction->setWeights(qFuncWeights);
		qFuncWeights += gradQFunction->getNumWeights();
	}
}

void CComposedQFunction::resetData()
{
	std::list<CAbstractQFunction *>::iterator it = qFunctions->begin();
	for (; it != qFunctions->end();it++)
	{
		(*it)->resetData();
	}
}

