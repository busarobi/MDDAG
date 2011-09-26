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
#include "cstatemodifier.h"
#include "cstatecollection.h"
#include "cstate.h"

#include <math.h>
#include "newmat/newmatio.h"

CStateModifier::CStateModifier(unsigned int numContinuousStates, unsigned int numDiscreteStates) : CStateProperties(numContinuousStates, numDiscreteStates)
{
	stateCollections = new std::list<CStateCollectionImpl *>;

	this->changeState = false;
}

CStateModifier::CStateModifier() : CStateProperties()
{
	stateCollections = new std::list<CStateCollectionImpl *>;

	this->changeState = false;
}

CStateModifier::~CStateModifier()
{
	delete stateCollections;
}


void CStateModifier::registerStateCollection(CStateCollectionImpl *stateCollection)
{
	stateCollections->push_back(stateCollection);
}

void CStateModifier::removeStateCollection(CStateCollectionImpl *stateCollection)
{
	stateCollections->remove(stateCollection);
}

void CStateModifier::stateChanged()
{
	std::list<CStateCollectionImpl *>::iterator it = stateCollections->begin();

	for (; it != stateCollections->end(); it++)
	{
		(*it)->setIsStateCalculated(this, false);
	}
}



CNeuralNetworkStateModifier::CNeuralNetworkStateModifier(CStateProperties *originalState) : CStateModifier(getNumInitContinuousStates(originalState, NULL, 0),0)
{
	this->originalState = originalState;

	for (unsigned int i = 0; i < getNumContinuousStates(); i++)
	{
		setMinValue(i, -10000.0);
		setMaxValue(i, 10000.0);
	}
	numDim = 0;
	dimensions = NULL;

	input_mean = new ColumnVector(numDim);
	input_std = new ColumnVector(numDim);

	*input_mean = 0;
	*input_std = 1;

	// preserve normalization on the intervall [-1 1]

	for (unsigned int i = 0; i < numDim; i ++)
	{
		input_mean->element(i) =  (originalState->getMaxValue(dimensions[i]) + originalState->getMinValue(dimensions[i])) / 2.0;
		
		input_std->element(i) = (originalState->getMaxValue(dimensions[i]) - originalState->getMinValue(dimensions[i])) / 2.0;
	}

	buffVector = new ColumnVector(numDim);
}

CNeuralNetworkStateModifier::CNeuralNetworkStateModifier(CStateProperties *originalState, unsigned int *l_dimensions, unsigned int l_numDim) : CStateModifier(getNumInitContinuousStates(originalState, l_dimensions, l_numDim),0)
{
	this->originalState = originalState;

	
	this->numDim = l_numDim;
	dimensions = new unsigned int[numDim];
	memcpy(dimensions, l_dimensions, sizeof(int) * numDim);

	normValues = false;

	for (unsigned int i = 0; i < getNumContinuousStates(); i++)
	{
		setMinValue(i, - 100000.0);
		setMaxValue(i,   100000.0);
	}
	input_mean = new ColumnVector(numDim);
	input_std = new ColumnVector(numDim);

	*input_mean = 0;
	*input_std = 1;

	// preserve normalization on the intervall [-1 1]

	for (unsigned int i = 0; i < numDim; i ++)
	{
		input_mean->element(i) =  (originalState->getMaxValue(dimensions[i]) + originalState->getMinValue(dimensions[i])) / 2.0;
		
		input_std->element(i) = (originalState->getMaxValue(dimensions[i]) - originalState->getMinValue(dimensions[i])) / 2.0;
	}

	buffVector = new ColumnVector(numDim);
}


CNeuralNetworkStateModifier::~CNeuralNetworkStateModifier()
{
	delete dimensions;

	delete input_mean;
	delete input_std;

	delete buffVector;
}

int CNeuralNetworkStateModifier::getNumInitContinuousStates(CStateProperties *properties,unsigned  int *dimensions,unsigned  int numDim)
{

	int numConStates = properties->getNumContinuousStates();
	
	if (numDim == 0)
	{
		for (unsigned int i = 0; i < properties->getNumContinuousStates(); i++)
		{
			if (properties->getPeriodicity(i))
			{
				numConStates ++;
			}
		}
	}
	else
	{
		numConStates = numDim;
		for (unsigned int i = 0; i < numDim; i++)
		{
			if (properties->getPeriodicity(dimensions[i]))
			{
				numConStates ++;
			}
		}
	}
	
	return numConStates;
}

void CNeuralNetworkStateModifier::getModifiedState(CStateCollection *originalStateCol, CState *modifiedState)
{
	int contStateIndex = 0;

	// set Discrete States
	CState *state = originalStateCol->getState(originalState);

//	for (unsigned int i = 0; i < originalState->getNumDiscreteStates(); i++)
//	{
//		modifiedState->setDiscreteState(i, state->getDiscreteState(i));
//	}
	*buffVector = 0;

	if (dimensions == NULL)
	{
		for (unsigned int i = 0; i < originalState->getNumContinuousStates(); i++)
		{
			buffVector->element(i) = state->getContinuousState(i);	
		}
		if (normValues)
		{
			preprocessInput(buffVector, buffVector);
		}
		
		for (unsigned int i = 0; i < originalState->getNumContinuousStates(); i++)
		{
			double stateVal = buffVector->element(i);
		
			if (originalState->getPeriodicity(i))
			{
				modifiedState->setContinuousState(contStateIndex ++, sin(stateVal * M_PI));
				modifiedState->setContinuousState(contStateIndex ++, cos(stateVal * M_PI));
			}
			else
			{
				modifiedState->setContinuousState(contStateIndex ++, stateVal);
			}
		}
	}
	else
	{
		for (unsigned int i = 0; i < numDim; i++)
		{
			buffVector->element(i) = state->getContinuousState(dimensions[i]);	
		}

		if (normValues)
		{
			preprocessInput(buffVector, buffVector);
		}
	
		for (unsigned int i = 0; i <numDim; i++)
		{
			double stateVal = buffVector->element(i);
			
			if (originalState->getPeriodicity(dimensions[i]))
			{
				modifiedState->setContinuousState(contStateIndex, sin(stateVal * M_PI));
				contStateIndex ++;
				modifiedState->setContinuousState(contStateIndex, cos(stateVal * M_PI));
				contStateIndex ++;
			}
			else
			{
				modifiedState->setContinuousState(contStateIndex, stateVal);
				contStateIndex ++;
			}
		}
	
		/*for (unsigned int i = 0; i <numDim; i++)
		{
			double width = originalState->getMaxValue(dimensions[i]) - originalState->getMinValue(dimensions[i]);
			double stateVal = 0.0;
			if (normValues == true)
			{
				stateVal = (state->getContinuousState(dimensions[i]) - originalState->getMinValue(dimensions[i])) / width * 2 - 1.0;
			}
			else
			{
				stateVal = state->getContinuousState(dimensions[i]);
			}
			
			if (originalState->getPeriodicity(dimensions[i]))
			{
				modifiedState->setContinuousState(contStateIndex ++, sin(stateVal * M_PI));
				modifiedState->setContinuousState(contStateIndex ++, cos(stateVal * M_PI));
			}
			else
			{
				modifiedState->setContinuousState(contStateIndex ++, stateVal);
			}
		}*/
	}
	
}

void CNeuralNetworkStateModifier::preprocessInput(ColumnVector *input, ColumnVector *norm_input)
{
	for (int i = 0; i < input->nrows(); i++)
	{
		norm_input->element(i) = (input->element(i) - input_mean->element(i)) / input_std->element(i);
	}
}



void CNeuralNetworkStateModifier::setPreprocessing(ColumnVector *l_input_mean, ColumnVector *l_input_std)
{
	*input_mean = *l_input_mean;
	*input_std = *l_input_std;

	normValues = true;
}


CFeatureCalculator::CFeatureCalculator(unsigned int numFeatures, unsigned int numActiveFeatures) : CStateModifier()
{
	initFeatureCalculator(numFeatures, numActiveFeatures);	
	originalState = NULL;
}

CFeatureCalculator::CFeatureCalculator() : CStateModifier()
{
	numFeatures = 0;
	numActiveFeatures = 0;

	originalState = NULL;
}

void CFeatureCalculator::initFeatureCalculator(unsigned int numFeatures, unsigned int numActiveFeatures)
{
	CStateProperties::initProperties(numActiveFeatures,numActiveFeatures, FEATURESTATE);
	for (unsigned int i = 0; i < this->getNumDiscreteStates(); i++)
	{
		this->setDiscreteStateSize(i, numFeatures);
	}
	this->numFeatures = numFeatures;
	this->numActiveFeatures = numActiveFeatures;

	
}

unsigned int CFeatureCalculator::getDiscreteStateSize(unsigned int )
{
	return numFeatures;
}

unsigned int CFeatureCalculator::getDiscreteStateSize()
{
	return numFeatures;
}

double CFeatureCalculator::getMin(unsigned int )
{
	return 0.0;
}

double CFeatureCalculator::getMax(unsigned int )
{
	return 1.0;
}

unsigned int CFeatureCalculator::getNumFeatures()
{
	return numFeatures;
}

unsigned int CFeatureCalculator::getNumActiveFeatures()
{
	return numActiveFeatures;
}

void CFeatureCalculator::normalizeFeatures(CState *featState)
{
	double sum = 0.0;
	unsigned int i = 0;
	for (i = 0; i < featState->getNumActiveDiscreteStates(); i++)
	{
		sum += featState->getContinuousState(i);
	}
	
	if (sum > 0)
	{
		for (i = 0; i < featState->getNumActiveDiscreteStates(); i++)
		{
			featState->setContinuousState(i, featState->getContinuousState(i) / sum);
		}
	}
}

CStateMultiModifier::CStateMultiModifier()
{
	states  = new std::list<CState *>();
	modifiers = new std::list<CStateModifier *>();
}

CStateMultiModifier::~CStateMultiModifier()
{
	std::list<CState *>::iterator it;
	for (it = states->begin(); it != states->end(); it ++)
	{
		delete *it;
	}
	delete states;
	delete modifiers;
}

void CStateMultiModifier::addStateModifier(CStateModifier *featCalc)
{
	modifiers->push_back(featCalc);
	states->push_back(new CState(featCalc));
}

std::list<CStateModifier *>* CStateMultiModifier::getStateModifiers()
{
	return modifiers;
}


CStateVariablesChooser::CStateVariablesChooser(unsigned int numContStates,unsigned int *l_contStatesInd,unsigned int numDiscStates,unsigned int *l_discStatesInd, CStateProperties *originalState) :  CStateModifier(numContStates, numDiscStates)
{
	contStatesInd = NULL;
	discStatesInd = NULL;

	if (numContStates > 0)
	{
		contStatesInd = new unsigned int[numContStates];
		memcpy(contStatesInd, l_contStatesInd, sizeof(unsigned int) * numContStates);
	}
	if (numDiscStates > 0)
	{
		discStatesInd = new unsigned int[numDiscStates];

		memcpy(discStatesInd, l_discStatesInd, sizeof(unsigned int) * numDiscStates);
	}

	this->originalState = originalState;


	for (int i = 0; i < getNumContinuousStates(); i++)
	{
		setMinValue(i, originalState->getMinValue(contStatesInd[i]));
		setMaxValue(i, originalState->getMaxValue(contStatesInd[i]));
	}

	for (int i = 0; i < getNumDiscreteStates(); i++)
	{
		setDiscreteStateSize(i, originalState->getDiscreteStateSize(discStatesInd[i]));
	}

	
}

CStateVariablesChooser::~CStateVariablesChooser()
{
	if (contStatesInd)
	{
		delete contStatesInd;
	}

	if (discStatesInd)
	{
		delete discStatesInd;
	}
}


void CStateVariablesChooser::getModifiedState(CStateCollection *originalStateCol, CState *modifiedState)
{
	CState *origState = originalStateCol->getState(originalState);
	for (unsigned int i = 0; i < getNumContinuousStates(); i ++)
	{
		modifiedState->setContinuousState(i, origState->getContinuousState(contStatesInd[i]));
	}
	for (unsigned int i = 0; i < getNumDiscreteStates(); i ++)
	{
		modifiedState->setDiscreteState(i, origState->getDiscreteState(discStatesInd[i]));
	}
}

