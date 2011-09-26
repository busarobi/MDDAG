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
#include "cvfunction.h"
#include "cvfunctionfromqfunction.h"
#include "crewardfunction.h"
#include "cvetraces.h"
#include "cstatemodifier.h"
#include "cstatecollection.h"
#include "cstate.h"
#include "cstateproperties.h"
#include "caction.h"
#include "cdiscretizer.h"
#include "cgradientfunction.h"
#include "cqfunction.h"

#include <assert.h>
#include <math.h>
#include <sstream>


CAbstractVFunction::CAbstractVFunction(CStateProperties *prop) : CStateObject(prop)
{
//	gamma = 0.95;
	type = 0;
	mayDiverge = false;
}

CAbstractVFunction::~CAbstractVFunction()
{
}


void CAbstractVFunction::updateValue(CStateCollection *state, double td)
{
	updateValue(state->getState(properties), td);
}

void CAbstractVFunction::setValue(CStateCollection *state, double qValue)
{
	setValue(state->getState(properties), qValue);
}

double CAbstractVFunction::getValue(CStateCollection *state)
{
	return getValue(state->getState(properties));
}

void CAbstractVFunction::updateValue(CState *state, double td)
{
	setValue(state, getValue(state) + td);
}

int CAbstractVFunction::getType()
{
	return type;
}

bool CAbstractVFunction::isType(int isT)
{
	int temp = type & isT;
	return (temp == isT);
}

void CAbstractVFunction::addType(int newType)
{
	type = type | newType;
}

/*void CAbstractVFunction::setGamma(double gamma)
{
	this->gamma = gamma;
}*/

void CAbstractVFunction::saveData(FILE *file)
{
    fprintf(file, "V-Function:\n");
    fprintf(file, "\n");
}

void CAbstractVFunction::loadData(FILE *file)
{
    assert(fscanf(file, "V-Function:\n") == 0);
    assert(fscanf(file, "\n") == 0);
}

CAbstractVETraces *CAbstractVFunction::getStandardETraces()
{
	return new CStateVETraces(this, properties);
}

CZeroVFunction::CZeroVFunction() : CAbstractVFunction(NULL)
{

}

double CZeroVFunction::getValue(CState *)
{
	return 0;
}

CVFunctionSum::CVFunctionSum() : CAbstractVFunction(NULL)
{
	vFunctions = new std::map<CAbstractVFunction *, double>;
}

CVFunctionSum::~CVFunctionSum()
{
	delete vFunctions;
}


double CVFunctionSum::getValue(CStateCollection *state)
{
	std::map<CAbstractVFunction *, double>::iterator it = vFunctions->begin();

	double sum = 0.0;
	for (;it != vFunctions->end(); it ++)
	{
		CAbstractVFunction *vFunc = (*it).first;
		
		sum += (*it).second * vFunc->getValue(state);
	}
	return sum;
}

double CVFunctionSum::getVFunctionFactor(CAbstractVFunction *vFunction)
{
	return (*vFunctions)[vFunction];
}

void CVFunctionSum::setVFunctionFactor(CAbstractVFunction *vFunction, double factor)
{
	(*vFunctions)[vFunction] = factor;
}

void CVFunctionSum::addVFunction(CAbstractVFunction *vFunction, double factor)
{
	(*vFunctions)[vFunction] = factor;
}

void CVFunctionSum::removeVFunction(CAbstractVFunction *vFunction)
{
	std::map<CAbstractVFunction *, double>::iterator it = vFunctions->find(vFunction);

	vFunctions->erase(it);
}

void CVFunctionSum::normFactors(double factor)
{
	std::map<CAbstractVFunction *, double>::iterator it = vFunctions->begin();

	double sum = 0.0;
	for (;it != vFunctions->end(); it ++)
	{
		sum += (*it).second;
	}
	for (;it != vFunctions->end(); it ++)
	{
		(*it).second *= factor / sum;
	}
}

CDivergentVFunctionException::CDivergentVFunctionException(string vFunctionName, CAbstractVFunction *vFunction, CState *state, double value) : CMyException(101, "DivergentVFunction")
{
	this->vFunction = vFunction;
	this->vFunctionName = vFunctionName;
	this->state = state;
	this->value = value;
}

string CDivergentVFunctionException::getInnerErrorMsg()
{
	char errorMsg[1000];

	sprintf(errorMsg, "%s diverges (value = %f, |value| > 100000)\n", vFunctionName.c_str(), value);
	
	return string(errorMsg);
}

CGradientVFunction::CGradientVFunction(CStateProperties *properties) : CAbstractVFunction(properties)
{
	this->addType(GRADIENTVFUNCTION);

	//this->gradientFeatures = new CFeatureList();
}



CGradientVFunction::~CGradientVFunction()
{
	//delete gradientFeatures;
}

void CGradientVFunction::updateValue(CStateCollection *state, double td)
{
	localGradientFeatureBuffer->clear();
	getGradient(state, localGradientFeatureBuffer);

	updateGradient(localGradientFeatureBuffer, td);
}

void CGradientVFunction::updateValue(CState *state, double td)
{
	localGradientFeatureBuffer->clear();
	getGradient(state, localGradientFeatureBuffer);

	updateGradient(localGradientFeatureBuffer, td);
}

CAbstractVETraces *CGradientVFunction::getStandardETraces()
{
	return new CGradientVETraces(this);
}


/*CGradientDelayedUpdateVFunction::CGradientDelayedUpdateVFunction(CGradientVFunction *vFunction) : CGradientVFunction(vFunction->getStateProperties()), CGradientDelayedUpdateFunction(vFunction)
{
	this->vFunction = vFunction;
}

double CGradientDelayedUpdateVFunction::getValue(CState *state)
{
	return vFunction->getValue(state);
}
	
void CGradientDelayedUpdateVFunction::getGradient(CStateCollection *state, CFeatureList *gradientFeatures)
{
	vFunction->getGradient(state, gradientFeatures);
}
*/

CVFunctionInputDerivationCalculator::CVFunctionInputDerivationCalculator(CStateProperties *modelState)
{
	this->modelState = modelState;
}

unsigned int CVFunctionInputDerivationCalculator::getNumInputs()
{
	return modelState->getNumContinuousStates();
}


CVFunctionNumericInputDerivationCalculator::CVFunctionNumericInputDerivationCalculator(CStateProperties *modelState, CAbstractVFunction *vFunction, double stepSize, std::list<CStateModifier *> *modifiers) : CVFunctionInputDerivationCalculator(modelState)
{
	this->vFunction = vFunction;
	this->stateBuffer = new CStateCollectionImpl(modelState, modifiers);

	addParameter("NumericInputDerivationStepSize", stepSize);
}

CVFunctionNumericInputDerivationCalculator::~CVFunctionNumericInputDerivationCalculator()
{
	delete stateBuffer;
}

void CVFunctionNumericInputDerivationCalculator::getInputDerivation( CStateCollection *state, ColumnVector *targetVector)
{
	CState *inputState = stateBuffer->getState(modelState);
	inputState->setState(state->getState(modelState));

	double stepSize = getParameter("NumericInputDerivationStepSize");

	for (unsigned int i = 0; i < modelState->getNumContinuousStates(); i++)
	{
		double stepSize_i = (modelState->getMaxValue(i) - modelState->getMinValue(i)) * stepSize;
		inputState->setContinuousState(i, inputState->getContinuousState(i) + stepSize_i);
		stateBuffer->newModelState();
		double vPlus = vFunction->getValue(stateBuffer);
		inputState->setContinuousState(i, inputState->getContinuousState(i) - 2 * stepSize_i);
		stateBuffer->newModelState();
		double vMinus = vFunction->getValue(stateBuffer);

		inputState->setContinuousState(i, inputState->getContinuousState(i) + stepSize_i);
		targetVector->element(i) = (vPlus - vMinus) / (2 * stepSize_i);
	}
}


CFeatureVFunction::CFeatureVFunction(int numFeatures) : CGradientVFunction(NULL), CFeatureFunction(numFeatures)
{
	
}

CFeatureVFunction::CFeatureVFunction(CStateProperties *prop) : CGradientVFunction(prop), CFeatureFunction(prop->getDiscreteStateSize(0))
{
}



CFeatureVFunction::CFeatureVFunction(CFeatureQFunction *qfunction, CStochasticPolicy *policy) : CGradientVFunction(qfunction->getFeatureCalculator()) ,CFeatureFunction(qfunction->getFeatureCalculator()->getDiscreteStateSize())
{
	setVFunctionFromQFunction(qfunction, policy);
}

CFeatureVFunction::~CFeatureVFunction()
{
}
	
void CFeatureVFunction::setVFunctionFromQFunction(CFeatureQFunction *qfunction, CStochasticPolicy *policy)
{
	CStateProperties properties(0, 1, DISCRETESTATE);
	properties.setDiscreteStateSize(0, numFeatures);
	CState discState(&properties);
	
	CAbstractVFunction *tempFunction;
	
	if (policy)
	{
		tempFunction = new CVFunctionFromQFunction(qfunction, policy, qfunction->getFeatureCalculator());
	}
	else
	{
		tempFunction = new COptimalVFunctionFromQFunction(qfunction, qfunction->getFeatureCalculator());
	}

	for (unsigned int i = 0; i < numFeatures; i++)
	{
		discState.setDiscreteState(0, i);
		setFeature(i, tempFunction->getValue(&discState));
	}
	delete tempFunction;
}


void CFeatureVFunction::getGradient(CStateCollection *stateCol, CFeatureList *gradient)
{
	CState *state = stateCol->getState(properties);

	int type = state->getStateProperties()->getType() & (DISCRETESTATE | FEATURESTATE);
	switch (type)
	{
	case FEATURESTATE:
		{
			for (unsigned int i = 0; i < state->getNumActiveDiscreteStates(); i ++)
			{
				gradient->update(state->getDiscreteState(i), state->getContinuousState(i));
			}
			break;
		}
	case DISCRETESTATE:
		{
			gradient->set(state->getDiscreteState(0), 1.0);
			break;
		}
	default:
		{
			gradient->set(state->getDiscreteStateNumber(), 1.0);
			break;
		}
	}
	if (DebugIsEnabled('v'))
	{
		DebugPrint('v', "Calculating feature Gradient List : ");
		gradient->saveASCII(DebugGetFileHandle('v'));
		DebugPrint('v', "\n");
	}
}

void CFeatureVFunction::updateValue(CState *state, double td)
{
	if (DebugIsEnabled('v'))
	{
		DebugPrint('v', "Update V-Value: %f", td);
		state->saveASCII(DebugGetFileHandle('v'));
		DebugPrint('v', "\n");
	}

	int type = state->getStateProperties()->getType() & (DISCRETESTATE | FEATURESTATE);
	switch (type)
	{
		case FEATURESTATE:
			{
				for (unsigned int i = 0; i < state->getNumActiveDiscreteStates(); i++)
				{
					updateFeature(state->getDiscreteState(i), td * state->getContinuousState(i));
				}
				break;
			}
		case DISCRETESTATE:
			{
				updateFeature(state->getDiscreteState(0), td);
				break;
			}
		default:
			{
				updateFeature(state->getDiscreteStateNumber(), td);
			}
	}
}

void CFeatureVFunction::setValue(CState *state, double qValue)
{
	if (DebugIsEnabled('v'))
	{
		DebugPrint('v', "Set V-Value: %f", qValue);
		state->saveASCII(DebugGetFileHandle('v'));
		DebugPrint('v', "\n");
	}

	int type = state->getStateProperties()->getType() & (DISCRETESTATE | FEATURESTATE);
	switch (type)
	{
		case FEATURESTATE:
			{
				for (unsigned int i = 0; i < state->getNumActiveDiscreteStates(); i++)
				{
					setFeature(state->getDiscreteState(i), qValue * state->getContinuousState(i));
				}
				break;
			}
		case DISCRETESTATE:
			{
				setFeature(state->getDiscreteState(0), qValue);
				break;
			}
		default:
			{
				setFeature(state->getDiscreteStateNumber(), qValue);
			}
	}
}

double CFeatureVFunction::getValue(CState *state)
{
	double value = 0;

	int type = state->getStateProperties()->getType() & (DISCRETESTATE | FEATURESTATE);
	switch (type)
	{
		case FEATURESTATE:
			{
				for (unsigned int i = 0; i < state->getNumActiveDiscreteStates(); i++)
				{
					value += getFeature(state->getDiscreteState(i)) * state->getContinuousState(i);
				}
				break;
			}
		case DISCRETESTATE:
			{
				value = getFeature(state->getDiscreteState(0));
				break;
			}
		default:
			{
				value = getFeature(state->getDiscreteStateNumber());
			}
	}
	
	if (DebugIsEnabled('v'))
	{
		DebugPrint('v', "Get V-Value: %f", value);
		state->saveASCII(DebugGetFileHandle('v'));
		DebugPrint('v', "\n");
	}
	if (! mayDiverge && (value < - DIVERGENTVFUNCTIONVALUE || value > DIVERGENTVFUNCTIONVALUE))
	{
		throw new CDivergentVFunctionException("Feature Function", this, state, value);
	}
	return value;
} 

void CFeatureVFunction::saveData(FILE *file)
{
	CGradientVFunction::saveData(file);
}

void CFeatureVFunction::loadData(FILE *file)
{
	CAbstractVFunction::loadData(file);

	//CGradientVFunction::loadData(file);

	CGradientVFunction::loadData(file);
}

void CFeatureVFunction::printValues()
{
	CAbstractVFunction::printValues();
	printFeatures();
}

CAbstractVETraces *CFeatureVFunction::getStandardETraces()
{
	return new CFeatureVETraces(this);
}

/*CStateProperties *CFeatureVFunction::getGradientCalculator()
{
	return properties;
}*/

void CFeatureVFunction::updateWeights(CFeatureList *gradientFeatures)
{
	this->updateFeatureList(gradientFeatures, 1.0);
}

int CFeatureVFunction::getNumWeights()
{
	return this->numFeatures;
}

void CFeatureVFunction::resetData()
{
	CFeatureFunction::init(0.0);
}

void CFeatureVFunction::getWeights(double *parameters)
{
	memcpy(parameters, this->features, sizeof(double) * getNumFeatures());
}

void CFeatureVFunction::setWeights(double *parameters)
{
	memcpy(this->features, parameters, sizeof(double) * getNumFeatures());
}

void CFeatureVFunction::setFeatureCalculator(CFeatureCalculator *featCalc)
{
	assert(featCalc->getNumFeatures() < numFeatures);

	properties = featCalc;
}


/*
CFeatureVFunctionInputDerivationCalculator::CFeatureVFunctionInputDerivationCalculator(CStateProperties *inputState, CFeatureVFunction *vFunction) : CVFunctionInputDerivationCalculator(inputState)
{
	this->vFunction = vFunction;
	featureInputDerivation = new ColumnVector(getNumInputs());
	fiDerivationSum = new ColumnVector(getNumInputs());

	normalizedFeatures = true;
}

CFeatureVFunctionInputDerivationCalculator::~CFeatureVFunctionInputDerivationCalculator()
{
	delete featureInputDerivation;
	delete fiDerivationSum;
}

void CFeatureVFunctionInputDerivationCalculator::getInputDerivation( CStateCollection *state, ColumnVector *targetVector)
{
	CState *featureState = state->getState(vFunction->getStateProperties());

	targetVector->initVector(0.0);

	CFeatureCalculator *featCalc = dynamic_cast<CFeatureCalculator *>(vFunction->getStateProperties());
	
	if (!normalizedFeatures)
	{
		for (unsigned int i = 0; i < featureState->getNumContinuousStates(); i ++)
		{
			featureInputDerivation->initVector(0.0);
			featCalc->getFeatureDerivationX(featureState->getDiscreteState(i), state, featureInputDerivation);

			if (DebugIsEnabled('v'))
			{
				DebugPrint('v',"Feature VFunction Input Gradient for Feature %d (%f): ", featureState->getDiscreteState(i), featureState->getContinuousState(i));
				for (int i = 0; i < featureInputDerivation->nrows(); i ++)
				{
					DebugPrint('v', "%f, ", featureInputDerivation->element(i));
				}
				DebugPrint('v', "\n");
			}
			featureInputDerivation->multScalar(vFunction->getFeature(featureState->getDiscreteState(i)));

			targetVector->addVector(featureInputDerivation);
		}
	}
	else
	{
		double fiSum = 0;
		fiDerivationSum->initVector(0.0);
		
		for (unsigned int i = 0; i < featureState->getNumContinuousStates(); i ++)
		{
			featureInputDerivation->initVector(0.0);
			featCalc->getFeatureDerivationX(featureState->getDiscreteState(i), state, featureInputDerivation);

			fiSum += featureState->getContinuousState(i);
			fiDerivationSum->addVector(featureInputDerivation);

			if (DebugIsEnabled('v'))
			{
				DebugPrint('v',"Feature VFunction Input Gradient for Feature %d (%f): ", featureState->getDiscreteState(i), featureState->getContinuousState(i));
				for (int i = 0; i < featureInputDerivation->nrows(); i ++)
				{
					DebugPrint('v', "%f, ", featureInputDerivation->element(i));
				}
				DebugPrint('v', "\n");
			}
			featureInputDerivation->multScalar(vFunction->getFeature(featureState->getDiscreteState(i)));

			targetVector->addVector(featureInputDerivation);
		}

		targetVector->multScalar(1 / fiSum);
		fiDerivationSum->multScalar(- vFunction->getValue(featureState));
		targetVector->addVector(fiDerivationSum);
	}
	
	double length = targetVector->getLength();

	if (length > 0.01)
	{
		targetVector->multScalar(1 / length);
	}
	

	if (DebugIsEnabled('v'))
	{
		DebugPrint('v',"Feature VFunction Input Gradient: ");
		for (unsigned int i = 0; i < targetVector->nrows(); i ++)
		{
			DebugPrint('v', "%f, ", targetVector->element(i));
		}
		DebugPrint('v', "\n");
	}
}
*/

CVTable::CVTable(CAbstractStateDiscretizer *discretizer) : CFeatureVFunction(discretizer)
{
}

CVTable::~CVTable() 
{
}

void CVTable::setDiscretizer(CAbstractStateDiscretizer *discretizer)
{
	assert(discretizer == NULL || discretizer->getDiscreteStateSize() == numFeatures);

	this->properties = discretizer;
}

CAbstractStateDiscretizer *CVTable::getDiscretizer()
{
	return (CAbstractStateDiscretizer*) properties;
}


int CVTable::getNumStates()
{
	return getNumFeatures();
}

CRewardAsVFunction::CRewardAsVFunction(CStateReward *reward) : CAbstractVFunction(reward->getStateProperties())
{
	this->reward = reward;
}

double CRewardAsVFunction::getValue(CState *state)
{
	return reward->getStateReward(state);
}
