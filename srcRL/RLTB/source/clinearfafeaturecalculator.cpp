// Copyright (C) 2003
// Gerhard Neumann (gneumann@gmx.net)
// Stephan Neumann (sneumann@gmx.net) 
//                
// This file is part of RL Toolbox.
// http://www.igi.tugraz.at/ril_toolbox
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
#include "clinearfafeaturecalculator.h"
#include "cstate.h"
#include "cstatecollection.h"
#include "cutility.h"

#include <assert.h>
#include <math.h>

CFeatureOperatorOr::CFeatureOperatorOr() : CFeatureCalculator()
{
	addType(FEATURESTATEDERIVATIONX);

	this->featureFactors = new std::map<CStateModifier *, double>();
}

CFeatureOperatorOr::~CFeatureOperatorOr()
{
	delete featureFactors;
}

void CFeatureOperatorOr::getModifiedState(CStateCollection *stateCol, CState *state)
{
	assert(bInit);

	std::list<CStateModifier *>::iterator it = getStateModifiers()->begin();
	std::list<CState *>::iterator stateIt = states->begin();

	CState *stateBuf;

	int i = 0;
	int numFeatures = 0;

	for (; it != getStateModifiers()->end(); it++, stateIt++)
	{
		if (stateCol->isMember(*it))
		{
			stateBuf = stateCol->getState(*it);
		}
		else
		{
			stateBuf = *stateIt;
			(*it)->getModifiedState(stateCol, stateBuf);
		}
		double featureStateFactor = (*this->featureFactors)[*it];
		if (stateBuf->getStateProperties()->isType(FEATURESTATE))
		{
			for (unsigned int j = 0; j < stateBuf->getNumDiscreteStates(); j++)
			{
				state->setDiscreteState(i, stateBuf->getDiscreteState(j) + numFeatures);
				state->setContinuousState(i, stateBuf->getContinuousState(j) * featureStateFactor);
				i ++;
			}
		}
		else
		{
			if (stateBuf->getStateProperties()->isType(DISCRETESTATE))
			{
				state->setDiscreteState(i, stateBuf->getDiscreteState(0) + numFeatures);
				state->setContinuousState(i, featureStateFactor);
				i ++;
			}
		}

		numFeatures += (*it)->getDiscreteStateSize();
	}
	normalizeFeatures(state);
}

void CFeatureOperatorOr::addStateModifier(CStateModifier *featCalc, double factor)
{
	CStateMultiModifier::addStateModifier(featCalc);
	
    if (!featCalc->isType(STATEDERIVATIONX))
	{
		type = getType() & ~ STATEDERIVATIONX;	
	}
	(*this->featureFactors)[featCalc] = factor;

}

CStateModifier *CFeatureOperatorOr::getStateModifier(int feature)
{
	std::list<CStateModifier *>::iterator it = getStateModifiers()->begin();
	int numFeatures = (*it)->getDiscreteStateSize();

	while (it != getStateModifiers()->end() && numFeatures < feature)
	{
		it ++;

		numFeatures += (*it)->getDiscreteStateSize();
	}

	if (it != getStateModifiers()->end())
	{
		return *it;
	}
	else
	{
		return NULL;
	}
}

/*void CFeatureOperatorOr::getFeatureDerivationX(int feature, CStateCollection *state, ColumnVector *targetVector)
{
	assert(false);
	CStateModifier *stateMod = getStateModifier(feature);

	if (stateMod->isType(FEATURESTATEDERIVATIONX))
	{
		CFeatureCalculator *featCalc = dynamic_cast<CFeatureCalculator *>(stateMod);
		featCalc->getFeatureDerivationX(feature, state, targetVector)
	}
}*/

void CFeatureOperatorOr::initFeatureOperator()
{
	assert(!bInit);

	std::list<CStateModifier *>::iterator it;
	int numFeatures = 0;
	int numActiveFeatures = 0;
	for (it = this->modifiers->begin(); it != modifiers->end(); it ++)
	{
		numActiveFeatures += (*it)->getNumDiscreteStates();

		numFeatures += (*it)->getDiscreteStateSize(0);
	}
	
	initFeatureCalculator(numFeatures, numActiveFeatures);
}

CFeatureOperatorAnd::CFeatureOperatorAnd() : CFeatureCalculator()
{
	addType(FEATURESTATEDERIVATIONX);
}


void CFeatureOperatorAnd::getModifiedState(CStateCollection *stateCol, CState *featState)
{
	int featureOffset = 1;

	std::list<CStateModifier *>::iterator it = getStateModifiers()->begin();
	std::list<CState *>::iterator stateIt = states->begin();


	CState *stateBuf;

	for (unsigned int i = 0; i < getNumDiscreteStates();i ++)
	{
		featState->setDiscreteState(i, 0);
		featState->setContinuousState(i, 1.0);
	}

	int repetitions = getNumDiscreteStates();
	for (int j = 0; it != getStateModifiers()->end(); it ++, stateIt ++, j ++)
	{
		repetitions /= (*it)->getNumDiscreteStates();
		stateBuf = NULL;
		if (stateCol->isMember(*it))
		{
			stateBuf = stateCol->getState(*it);
		}
		else
		{
			stateBuf = *stateIt;
			(*it)->getModifiedState(stateCol, stateBuf);
		}
		
		if (stateBuf->getStateProperties()->isType(FEATURESTATE))
		{
			for (unsigned int i = 0; i < getNumDiscreteStates(); i++)
			{
				unsigned int singleStateFeatureNum = (i / repetitions) % stateBuf->getNumDiscreteStates();
				featState->setDiscreteState(i, featState->getDiscreteState(i) + featureOffset * stateBuf->getDiscreteState(singleStateFeatureNum));
				featState->setContinuousState(i, featState->getContinuousState(i) * stateBuf->getContinuousState(singleStateFeatureNum));
			}
		}
		else
		{
			for (unsigned int i = 0; i < getNumDiscreteStates(); i++)
			{
				featState->setDiscreteState(i, featState->getDiscreteState(i) + featureOffset * stateBuf->getDiscreteState(0));				
			}
		}

		featureOffset = featureOffset * (*it)->getDiscreteStateSize();
	}
	normalizeFeatures(featState);
}

/*void CFeatureOperatorAnd::getFeatureDerivationX(int feature, CStateCollection *stateCol, ColumnVector *derivation)
{
	assert(false);

	std::list<CStateModifier *>::iterator it = getStateModifiers()->begin();
	std::list<CState *>::iterator stateIt = states->begin();


	CState *stateBuf;

	double featureFactor = 1.0;
	ColumnVector tempVector(derivation->nrows());
	
	*derivation = (0.0);
	int lfeature = 0;
	double lfeatureFactor = 0.0;

	for (; it != getStateModifiers()->end(); it ++, stateIt ++)
	{
		stateBuf = NULL;
		
		assert((*it)->isType(FEATURESTATEDERIVATIONX));
		CFeatureCalculator *devXFeatCalc = dynamic_cast<CFeatureCalculator *>(*it);

		tempVector = (0.0);
		if (stateCol->isMember(*it))
		{
			stateBuf = stateCol->getState(*it);
		}
		else
		{
			stateBuf = *stateIt;
			(*it)->getModifiedState(stateCol, stateBuf);
		}
		unsigned int i = 0;
		lfeatureFactor = 0.0;
		lfeature = feature % (*it)->getDiscreteStateSize();
		feature = feature / (*it)->getDiscreteStateSize();

		while (i < stateBuf->getNumDiscreteStates() && stateBuf->getDiscreteState(i) != lfeature)
		{
			i++;
		}
		if (i < stateBuf->getNumDiscreteStates())
		{
			lfeatureFactor = stateBuf->getContinuousState(i);
		}

		devXFeatCalc->getFeatureDerivationX(lfeature, stateCol, &tempVector);
		
		tempVector.multScalar(featureFactor);
		derivation->multScalar(lfeatureFactor);
		
		derivation = derivation + &tempVector;

		featureFactor *= lfeatureFactor;
	}
}*/

void CFeatureOperatorAnd::addStateModifier(CStateModifier *featCalc)
{
	CStateMultiModifier::addStateModifier(featCalc);

	if (!featCalc->isType(STATEDERIVATIONX))
	{
		type = getType() & ~ STATEDERIVATIONX;	
	}
}

void CFeatureOperatorAnd::initFeatureOperator()
{
	assert(!bInit);
	std::list<CStateModifier *>::iterator it;
	int numFeatures = 1;
	int numActiveFeatures = 1;
	for (it = this->modifiers->begin(); it != modifiers->end(); it ++)
	{
		numActiveFeatures *= (*it)->getNumDiscreteStates();

		numFeatures *= (*it)->getDiscreteStateSize(0);
	}

	initFeatureCalculator(numFeatures, numActiveFeatures);
}

CGridFeatureCalculator::CGridFeatureCalculator(unsigned int numDim, unsigned int dimensions[], unsigned int part[], double off[], unsigned int numActiveFeatures) : CFeatureCalculator(numActiveFeatures, numActiveFeatures)
{
	this->numDim = numDim;
	this->dimensions = new unsigned int[numDim];
	this->partitions = new unsigned int[numDim];
	this->offsets = new double[numDim];

	dimensionSize = new unsigned int[numDim];

	numFeatures = 1;
	for (unsigned int i = 0; i < numDim; i ++)
	{
		this->dimensions[i] = dimensions[i];
		this->partitions[i] = part[i];
		this->offsets[i] = off[i];
        dimensionSize[i] = numFeatures;
		numFeatures *= partitions[i];
	}

	for (unsigned int i = 0; i < this->getNumDiscreteStates(); i++)
	{
		this->setDiscreteStateSize(i, numFeatures);
	}

	for (unsigned int i = 0; i < this->getNumContinuousStates(); i++)
	{
		this->setMinValue(i, 0.0);
		this->setMaxValue(i, 1.0);
	}

	gridScale = new double[numDim];

	for (unsigned int i = 0; i < numDim; i++)
	{
		gridScale[i] = 1.0;
	}

	originalState = NULL;
}

CGridFeatureCalculator::~CGridFeatureCalculator()
{
	delete offsets;
	delete partitions;
	delete dimensions;
	delete dimensionSize;

	delete gridScale;
}


unsigned int CGridFeatureCalculator::getNumDimensions()
{
	return numDim;
}

void CGridFeatureCalculator::setGridScale(int dimension, double scale)
{
	gridScale[dimension] = scale;
}

void CGridFeatureCalculator::getFeaturePosition(unsigned int feature, ColumnVector *position)
{
	int partition = 0;
	unsigned int temp = feature;
	
	for (unsigned int i = 0; i < numDim; i++)
	{
		partition = temp % partitions[i];
		position->element(i) = offsets[i] + 1.0 / partitions[i] * (0.5 + partition) * gridScale[i];

		temp = temp / partitions[i];
	}
}

unsigned int CGridFeatureCalculator::getActiveFeature(CState *state)
{
	double part = 0;
	int singleStateFeature = 0;
	unsigned int feature = 0;

	for (unsigned int i = 0; i < numDim; i++)
	{	
		assert(dimensions[i] < state->getNumContinuousStates());

		part = (state->getNormalizedContinuousState(dimensions[i]) - offsets[i]);

		if (state->getStateProperties()->getPeriodicity(dimensions[i]) && gridScale[i] >= 1.0)
		{
			part = part - floor(part);
		}

		singleStateFeature = (int) floor(part * partitions[i] / gridScale[i]);

		if (singleStateFeature < 0)
			singleStateFeature = 0;
		if ((unsigned int) singleStateFeature >= partitions[i])
			singleStateFeature = partitions[i] - 1;

		feature += singleStateFeature * dimensionSize[i];
	}
	return feature;
}

void CGridFeatureCalculator::getSingleActiveFeature(CState *state, unsigned int *activeFeature)
{
	double part = 0;
	int tempSingleStateFeature = 0;

	for (unsigned int i = 0; i < numDim; i++)
	{	
		assert(dimensions[i] < state->getNumContinuousStates());

		part = (state->getNormalizedContinuousState(dimensions[i]) - offsets[i]);

		if (state->getStateProperties()->getPeriodicity(dimensions[i]) && gridScale[i] >= 1.0)
		{
			part = part - floor(part);
		}

		tempSingleStateFeature = (int) floor(part * partitions[i] / gridScale[i]);

		if (tempSingleStateFeature < 0)
			tempSingleStateFeature = 0;
		if ((unsigned int) tempSingleStateFeature >= partitions[i])
			tempSingleStateFeature = partitions[i] - 1;

		activeFeature[i] = tempSingleStateFeature;
	}
}

unsigned int CGridFeatureCalculator::getFeatureIndex(int position[])
{
	unsigned int feature = 0;

	for (unsigned int i = 0; i < numDim; i++)
	{	
		feature +=  position[i] * dimensionSize[i];
	}
	return feature;
}

CTilingFeatureCalculator::CTilingFeatureCalculator(unsigned int numDim, unsigned int dimensions[], unsigned int partitions[], double offsets[]) : CGridFeatureCalculator(numDim, dimensions, partitions, offsets, 1)
{
}

CTilingFeatureCalculator::~CTilingFeatureCalculator()
{
}
	
void CTilingFeatureCalculator::getModifiedState(CStateCollection *state, CState *featState)
{
	featState->setDiscreteState(0, getActiveFeature(state->getState(originalState)));
	featState->setContinuousState(0, 1.0);
}


CLinearMultiFeatureCalculator::CLinearMultiFeatureCalculator(unsigned int numDim, unsigned int dimensions[], unsigned int partitions[], double offsets[], unsigned int numActiveFeatures) : CGridFeatureCalculator(numDim, dimensions, partitions, offsets, numActiveFeatures)
{
	areaSize = new unsigned int[numDim];	

	activePosition = new ColumnVector(numDim);
	featurePosition = new ColumnVector(numDim);
	actualPartition = new unsigned int[numDim];
	singleStateFeatures = new unsigned int[numDim];
}

CLinearMultiFeatureCalculator::~CLinearMultiFeatureCalculator()
{
	delete areaSize;

	delete activePosition;
	delete featurePosition;
	delete [] actualPartition;
	delete [] singleStateFeatures;

}

void CLinearMultiFeatureCalculator::initAreaSize()
{
	memset(areaSize, 0, sizeof(unsigned int) * numDim);
}
	
void CLinearMultiFeatureCalculator::calcNumActiveFeatures()
{
	areaNumPart = 1;
	for (unsigned int i = 0; i < numDim; i++)
	{
		areaNumPart *= areaSize[i];
	}
	this->continuousStates = areaNumPart;
	this->discreteStates = areaNumPart;

	numActiveFeatures = areaNumPart;

	this->discreteStateSize = new unsigned int[discreteStates];
	this->minValues = new double[continuousStates];
	this->maxValues = new double[continuousStates];

	for (unsigned int i = 0; i < this->getNumDiscreteStates(); i++)
	{
		this->setDiscreteStateSize(i, numFeatures);
	}

	for (unsigned int i = 0; i < this->getNumContinuousStates(); i++)
	{
		this->setMinValue(i, 0.0);
		this->setMaxValue(i, 1.0);
	}
}

void CLinearMultiFeatureCalculator::getModifiedState(CStateCollection *stateCol, CState *featState)
{
	assert(equals(featState->getStateProperties()));

	memset(actualPartition, 0, sizeof(unsigned int) * numDim);
	unsigned int j = 0;
	CState *state = stateCol->getState(originalState);

/*	printf("State : ");
	for (int i = 0; i < 4; i ++)
	{
		printf("%f ", state->getNormalizedContinuousState(i));
	}
	printf("\n");*/

	getSingleActiveFeature(state, singleStateFeatures);
	unsigned int i;
	int feature = 0;

	//offset to add to the actual feature
	int featureAdd = 0;

	getFeaturePosition(getActiveFeature(state), activePosition);
//	printf("[");
//	for (i = 0; i < numDim; i ++)
//	{
//		printf("%f ", state->getNormalizedContinuousState(dimensions[i])); 
//	}
//	printf("]");
//	activePosition->saveASCII(stdout);
//	printf("\n");

	for (i = 0; i < numDim; i ++)
	{
		int singleFeatureOffset = 0;
		if (areaSize[i] % 2 == 0)
		{
			//double x1 = state->getNormalizedContinuousState(dimensions[i]);
			//double x2 = activePosition->element(i);
			if (state->getNormalizedContinuousState(dimensions[i]) < activePosition->element(i))
			{
				singleFeatureOffset ++;
			}
			singleFeatureOffset += (areaSize[i] - 1) / 2;
		}
		else
		{
			singleFeatureOffset += areaSize[i] / 2;
		}
		singleStateFeatures[i] -= singleFeatureOffset;
		activePosition->element(i) = activePosition->element(i) - singleFeatureOffset * 1.0 / partitions[i] * gridScale[i];
	}

	unsigned int featureIndex = 0;


	for (i = 0; i < areaNumPart; i++)
	{
		feature = 0;
		/*for (j = 0; j < numDim; j++)
		{
			int dist = (actualPartition[j] - areaSize[j]);
			featurePosition->setElement(j,  (activePosition->element(j) + 1.0 / partitions[j] * dist));

			featureAdd = (singleStateFeatures[j] + (actualPartition[j] - areaSize[j]));
			if (state->getStateProperties()->getPeriodicity(j))
			{
				featurePosition->setElement(j, featurePosition->element(j) - floor(featurePosition->element(j)));
				featureAdd = featureAdd - (int) floor((double) featureAdd / (double) partitions[j]) * partitions[j];
			}

			feature = feature + featureAdd * dimensionSize[j];
		} */
		/*for (j = 0; j < numDim; j++)
		{
			int dist = (actualPartition[j] - 1);
			featurePosition->setElement(j,  (activePosition->element(j) + 1.0 / partitions[j] * dist));

			featureAdd = (singleStateFeatures[j] + (actualPartition[j] - 1));
			if (state->getStateProperties()->getPeriodicity(j))
			{
				featurePosition->setElement(j, featurePosition->element(j) - floor(featurePosition->element(j)));
				featureAdd = featureAdd - (int) floor((double) featureAdd / (double) partitions[j]) * partitions[j];
			}
			
			feature = feature + featureAdd * dimensionSize[j];
		} 
		for (i = 0; i < numDim; i ++)
		{
			int singleFeatureOffset = 0;
			if (areaSize[i] % 2 == 0)
			{
				if (state->getContinuousState(dimensions[i]) < activePosition->element(i))
				{
					singleFeatureOffset ++;
				}
				singleFeatureOffset += (singleFeatureOffset - 1) / 2;
			}
			else
			{
				singleFeatureOffset += areaSize[i] / 2;
			}
			singleStateFeatures[i] -= singleFeatureOffset;
			activePosition->setElement(i, activePosition->element(i) - singleFeatureOffset * 1.0 / partitions[i]);

		}*/
		for (j = 0; j < numDim; j++)
		{
			//int dist = (actualPartition[j] - areaSize[j]);
			featurePosition->element(j) =  activePosition->element(j) + (1.0 / partitions[j] * actualPartition[j])  * gridScale[j];

			featureAdd = (singleStateFeatures[j] + actualPartition[j]);
			if (state->getStateProperties()->getPeriodicity(j) && gridScale[j] >= 1.0)
			{
				featurePosition->element(j) = featurePosition->element(j) - floor(featurePosition->element(j));
				featureAdd = featureAdd - (int) floor((double) featureAdd / (double) partitions[j]) * partitions[j];
			}

			feature = feature + featureAdd * dimensionSize[j];
		}
		if (feature >= 0 && (unsigned int) feature < getNumFeatures())
		{
			featState->setDiscreteState(featureIndex, feature);
			featState->setContinuousState(featureIndex, getFeatureFactor(state, featurePosition));
			featureIndex ++;
		}
		
				
		j = 0;
	
		actualPartition[0] ++;
		while (j < numDim && actualPartition[j] >= areaSize[j])
		{
			actualPartition[j] = 0;
			j ++;
			if (j < numDim)
			{
				actualPartition[j]++;
			}
		}
	}
	featState->setNumActiveContinuousStates(featureIndex);
	featState->setNumActiveDiscreteStates(featureIndex);

	for (; featureIndex < areaNumPart; featureIndex ++)
	{
		featState->setDiscreteState(featureIndex, 0);
		featState->setContinuousState(featureIndex, 0.0);
	}

	this->normalizeFeatures(featState);
}

CRBFFeatureCalculator::CRBFFeatureCalculator(unsigned int numDim, unsigned int dimensions[], unsigned int partitions[], double offsets[], double sigma[]) : CLinearMultiFeatureCalculator(numDim, dimensions, partitions, offsets, 0)
{
	this->sigma = new double[numDim];
	memcpy(this->sigma, sigma, sizeof(double) * numDim);
	sigmaMaxSize = 2.0;
	initAreaSize();
	addType(FEATURESTATEDERIVATIONX);
}

CRBFFeatureCalculator::CRBFFeatureCalculator(unsigned int numDim, unsigned int dimensions[], unsigned int partitions[], double offsets[], double sigma[], unsigned int areaSize[]) : CLinearMultiFeatureCalculator(numDim, dimensions, partitions, offsets, 0)
{
	this->sigma = new double[numDim];
	memcpy(this->sigma, sigma, sizeof(double) * numDim);
	sigmaMaxSize = 2.0;
	memcpy(this->areaSize, areaSize, sizeof(unsigned int) * numDim);
	calcNumActiveFeatures();

	addType(FEATURESTATEDERIVATIONX);
}


CRBFFeatureCalculator::~CRBFFeatureCalculator()
{
	delete sigma;
}

void CRBFFeatureCalculator::initAreaSize()
{
	for (unsigned int i = 0; i < numDim; i++)
	{
		areaSize[i] = (2 * (unsigned int)floor(sigmaMaxSize * sigma[i] * partitions[i]) + 1);
		if(areaSize[i] <= 1)
		{
			areaSize[i] += 1;
		}
	}
	calcNumActiveFeatures();
}

double CRBFFeatureCalculator::getFeatureFactor(CState *state, ColumnVector *position)
{
	double exponent = 0.0;
	
//	printf("position : ");
//	position->saveASCII(stdout);

	for (unsigned int i = 0; i < numDim; i++)
	{
		double difference = fabs(state->getNormalizedContinuousState(dimensions[i]) - position->element(i));

		if (state->getStateProperties()->getPeriodicity(dimensions[i]) && difference > 0.5)
		{
			difference = 1 - difference;
		}

		exponent += pow(difference / (sigma[i] * gridScale[i]), 2) / 2;
//		printf("%d: %f\n", i, difference);
	}
//	printf(" Factor %f, exponent %f, %f \n", my_exp(- exponent), exponent, sigma[0]);

	return my_exp(- exponent);
}

/*
void CRBFFeatureCalculator::getFeatureDerivationX(int feature, CStateCollection *stateCol, ColumnVector *targetVector)
{
	assert(false);
	
	getFeaturePosition(feature, activePosition);
	CState *state = stateCol->getState();
	double factor = getFeatureFactor(state, activePosition);
	
	for (unsigned int i = 0; i < this->numDim; i ++)
	{
		targetVector->setElement(dimensions[i],  factor * (- state->getSingleStateDifference(dimensions[i], activePosition->element(i))) / pow(sigma[i], 2));
	}
}*/


CLinearInterpolationFeatureCalculator::CLinearInterpolationFeatureCalculator(unsigned int numDim, unsigned int dimensions[], unsigned int partitions[], double offsets[]) : CLinearMultiFeatureCalculator(numDim, dimensions, partitions, offsets, 0)
{
	initAreaSize();
}

CLinearInterpolationFeatureCalculator::~CLinearInterpolationFeatureCalculator()
{
}

double CLinearInterpolationFeatureCalculator::getFeatureFactor(CState *state, ColumnVector *featPos)
{
	double factor = 1.0;
	
	for (unsigned int i = 0; i < numDim; i++)
	{

		double difference = fabs(state->getNormalizedContinuousState(dimensions[i]) - featPos->element(i));

		if (state->getStateProperties()->getPeriodicity(dimensions[i]) && difference > 0.5)
		{
			difference = 1 - difference;
		}
		factor *= 1 - difference * partitions[i];
	}
	return factor;
}

void CLinearInterpolationFeatureCalculator::initAreaSize()
{
	for (unsigned int i = 0; i < numDim; i ++)
	{
		areaSize[i] = 2;
	}
	calcNumActiveFeatures();
}






CSingleStateFeatureCalculator::CSingleStateFeatureCalculator(int dimension, int numPartitions, double *partitions, int numActiveFeatures) : CFeatureCalculator(numPartitions, numActiveFeatures)
{
	this->dimension = dimension;
	this->partitions = new double[numPartitions];
	memcpy(this->partitions, partitions, sizeof(double) * numPartitions);

	originalState = NULL;
	this->numPartitions = numPartitions;
}

CSingleStateFeatureCalculator::~CSingleStateFeatureCalculator()
{
	delete partitions;
}

void CSingleStateFeatureCalculator::getModifiedState(CStateCollection *stateCol, CState *featState)
{
	CState *state = stateCol->getState(originalState);
	CStateProperties *properties = state->getStateProperties();
	double contState = state->getContinuousState(dimension);
	double width = properties->getMaxValue(dimension) - properties->getMinValue(dimension);


	if (contState < partitions[0] && properties->getPeriodicity(dimension))
	{
		contState += width;
	}

	unsigned int activeFeature = 0;
	unsigned int featureNum = 0, realfeatureNum = 0;

	int featureIndex = 0;

	double part = partitions[activeFeature];

	while (activeFeature < numFeatures && part < contState)
	{
		activeFeature++;

		if (activeFeature < numFeatures)
		{
			part = partitions[activeFeature];
		}

		if (part < partitions[0])
		{
			assert(properties->getPeriodicity(dimension));
			part += width;
		}
	}

	
	if (activeFeature == numFeatures && !properties->getPeriodicity(dimension))
	{
		featureNum ++;
	}

	DebugPrint('l', "Single State Features: [");	
	for (; realfeatureNum < this->numActiveFeatures; realfeatureNum++, featureNum ++)
	{
		if (featureNum % 2 == 0)
		{
			featureIndex = activeFeature + featureNum / 2;
		}
		else
		{
			featureIndex = activeFeature - (featureNum / 2 + 1);
		}

		if (state->getStateProperties()->getPeriodicity(dimension))
		{
			featureIndex = featureIndex % numFeatures;
		}
		
		if (featureIndex >= 0 && featureIndex < (signed int) numFeatures)
		{
			featState->setDiscreteState(realfeatureNum, featureIndex);
			
			double stateDiff = state->getSingleStateDifference(dimension, partitions[featureIndex]);
			
			double diffNextPart = 1.0;	
					
			if (!state->getStateProperties()->getPeriodicity(dimension))
			{
				if (featureIndex == 0 && stateDiff <= 0)
				{
					stateDiff = 0;
				}
				else
				{
					if (featureIndex == (signed int) (numFeatures - 1) && stateDiff > 0)
					{
						stateDiff = 0;
					}
					else
					{
						if (stateDiff <= 0)
						{
							diffNextPart = partitions[featureIndex] - partitions[featureIndex - 1];
						}
						else
						{
							diffNextPart = partitions[featureIndex + 1] - partitions[featureIndex];
						}
					}
				}

			}
			else
			{

				if (stateDiff <= 0)
				{
					diffNextPart = partitions[featureIndex] - partitions[(numPartitions + featureIndex - 1) % numPartitions];
				}
				else
				{
					diffNextPart = partitions[(featureIndex + 1) % numPartitions] - partitions[featureIndex];
				}				
				
				if (diffNextPart < 0)
				{
					diffNextPart += width; 
				}
				if (diffNextPart > 0)
				{
					diffNextPart -= width;
				}
			}
			
			featState->setContinuousState(realfeatureNum, getFeatureFactor(featureIndex, stateDiff, diffNextPart));
			
			DebugPrint('l', "%f %f, ", partitions[featureIndex], featState->getContinuousState(realfeatureNum));
		}
		else
		{
			featState->setContinuousState(realfeatureNum, 0.0);
			featState->setDiscreteState(realfeatureNum, 0);
		}
	}
	this->normalizeFeatures(featState);
	DebugPrint('l', "]\n");	
}


CSingleStateRBFFeatureCalculator::CSingleStateRBFFeatureCalculator(int dimension, int numPartitions, double *partitions, int numActiveFeatures) : CSingleStateFeatureCalculator(dimension, numPartitions, partitions, numActiveFeatures)
{
	//addType(FEATURESTATEDERIVATIONX);
}

/*
void CSingleStateRBFFeatureCalculator::getFeatureDerivationX(int feature, CStateCollection *stateCol, ColumnVector *targetVector)
{
	assert(false);
	// NOT WORKING !!!
	//double distance = stateCol->getState(originalState)->getSingleStateDifference(dimension, partitions[feature]);

	//double dev = my_exp(- pow(distance / sigma, 2) / 2) * distance * (- 1 / pow(sigma,2));
	//targetVector->setElement(dimension, targetVector->element(dimension) + dev);
}*/

double CSingleStateRBFFeatureCalculator::getFeatureFactor(int , double difference, double diffNextPart)
{
	double distance = fabs(difference);

	return my_exp(- pow(distance / diffNextPart * 2, 2)) ;
}


CSingleStateLinearInterpolationFeatureCalculator::CSingleStateLinearInterpolationFeatureCalculator(int dimension, int numPartitions, double *partitions) : CSingleStateFeatureCalculator(dimension, numPartitions, partitions, 2)
{
}

CSingleStateLinearInterpolationFeatureCalculator::~CSingleStateLinearInterpolationFeatureCalculator()
{
}

double CSingleStateLinearInterpolationFeatureCalculator::getFeatureFactor(int , double difference, double diffNextPart)
{
	return 1 - fabs(difference) / diffNextPart;
}

/*void CSingleStateLinearInterpolationFeatureCalculator::getFeatureDerivationX(int feature, CStateCollection *stateCol, ColumnVector *targetVector)
{
	double distance = fabs(stateCol->getState(originalState)->getSingleStateDifference(dimension, partitions[feature]));

	double dev = 0;
	
	if (distance > 0)
	{
		dev = - 1.0 /  getMirroredStateValue(dimension, partitions[feature] - partitions[(feature + 1) % numFeatures]);
	}
	else
	{
		dev = 1.0 / getMirroredStateValue(dimension, partitions[feature] - partitions[(feature + 1) % numFeatures]);
	}

	targetVector->setElement(dimension, dev);
}*/

CFeatureStateNNInput::CFeatureStateNNInput(CFeatureCalculator *l_featureStateCalc) : CStateModifier(l_featureStateCalc->getNumFeatures(), 0)
{
	this->featureStateCalc = l_featureStateCalc;
	this->featureState = new CState(l_featureStateCalc);	
}

CFeatureStateNNInput::~CFeatureStateNNInput()
{
	delete featureState;
}


void CFeatureStateNNInput::getModifiedState(CStateCollection *stateCol, CState *state)
{
	CState *featureStateBuff;
	state->resetState();

	if (stateCol->isMember(featureStateCalc))
	{
		featureStateBuff = stateCol->getState(featureStateCalc);
	}
	else
	{
		featureStateCalc->getModifiedState(stateCol, featureState);
		featureStateBuff = featureState;
	}

	for (unsigned int i = 0;i < featureStateCalc->getNumActiveFeatures(); i++)
	{
		state->setContinuousState(featureStateBuff->getDiscreteState(i), featureStateBuff->getContinuousState(i));
	}
}

