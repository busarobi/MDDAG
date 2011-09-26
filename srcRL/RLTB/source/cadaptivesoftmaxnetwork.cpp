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

#include <assert.h>
#include "ril_debug.h"
#include "cadaptivesoftmaxnetwork.h"

#include "cfeaturefunction.h"
#include "cgradientfunction.h"
#include "cvfunction.h"
#include "cerrorlistener.h"
#include "clinearfafeaturecalculator.h"
#include "cinputdata.h"
#include "crbftrees.h"
#include "cnearestneighbor.h"
#include "ckdtrees.h"
#include "cstatecollection.h"
#include "cstate.h"

CRBFCenterNetwork::CRBFCenterNetwork(std::vector<CRBFBasisFunction *> *l_centers, int l_numDim)
{
	centers = l_centers;
	initNetwork( );

	numDim = l_numDim;
}

CRBFCenterNetwork::CRBFCenterNetwork(int l_numDim)
{
	centers = new std::vector<CRBFBasisFunction *>;

	numDim = l_numDim;
}

 CRBFCenterNetwork::~CRBFCenterNetwork()
{
	deleteCenters();
	delete centers;
}

void CRBFCenterNetwork::saveData(FILE *stream)
{
	for (int i = 0; i < getNumCenters(); i ++)
	{
		CRBFBasisFunction *basisFunction = getCenter( i);
		
		for (int j = 0; j < basisFunction->getCenter()->nrows(); j ++)
		{
			fprintf(stream, "%f ", basisFunction->getCenter()->element(j));
		}

		for (int j = 0; j < basisFunction->getSigma()->nrows(); j ++)
		{
			fprintf(stream, "%f ", basisFunction->getSigma()->element(j));
		}
		fprintf(stream, "\n");
	}
}

void CRBFCenterNetwork::loadData(FILE *stream)
{
	deleteCenters();

	ColumnVector center(getNumDimensions());
	ColumnVector sigma(getNumDimensions());

	while (! feof(stream))
	{
		int results = 0;
		double dBuf = 0;
		for (int j = 0; j < getNumDimensions(); j ++)
		{
			results += fscanf(stream, "%lf ", &dBuf);
			center.element(j) = dBuf;
		}

		for (int j = 0; j < getNumDimensions(); j ++)
		{
			results += fscanf(stream, "%lf ", &dBuf);
			sigma.element(j) = dBuf;
		}
		fscanf(stream, "\n");
		if (results < 2 * getNumDimensions())
		{
			break;
		}
		CRBFBasisFunction *basisFunction = new CRBFBasisFunction(&center, &sigma);
		centers->push_back(basisFunction);
	}
	
	initNetwork();
}
			
void CRBFCenterNetwork::addCenter(CRBFBasisFunction* center){
    centers->push_back(center);
}

CRBFBasisFunction *CRBFCenterNetwork::getCenter(int index)
{
	return (*centers)[index];
}
		
int CRBFCenterNetwork::getNumCenters()
{
	return centers->size();
}


int CRBFCenterNetwork::getNumDimensions()
{
	return numDim;
}

void CRBFCenterNetwork::deleteCenters()
{
	std::vector<CRBFBasisFunction *>::iterator it = centers->begin();

	for (; it != centers->end(); it++)
	{
		delete *it;
	}
	centers->clear();
}


CRBFCenterNetworkSimpleSearch::CRBFCenterNetworkSimpleSearch(int numDim) : CRBFCenterNetwork(numDim)
{
}

CRBFCenterNetworkSimpleSearch::~CRBFCenterNetworkSimpleSearch()
{
}


void CRBFCenterNetworkSimpleSearch::getNearestNeigbors(ColumnVector *state, unsigned int K,  DataSubset *subset)
{
	assert(K > 0);
	
	subset->clear();

	ColumnVector difference = *state;

	std::list<double> distSet;	
	std::list<int> subsetList;

	for (int i = 0; i < getNumCenters(); i ++)
	{
		difference = state - getCenter(i)->getCenter();
		double distance = difference.norm_Frobenius();

		
		if (distSet.size() < K || (*distSet.begin() > distance))
		{
			std::list<double>::iterator itDist = distSet.begin();
			std::list<int>::iterator it = subsetList.begin();
	
			while (itDist != distSet.end() && (*itDist) > distance)
			{
				it ++;
				itDist ++;
			}
			if (distSet.size() > 0)
			{
				it --;
				itDist --;
			}
			distSet.insert(itDist, distance);
			subsetList.insert(it, i);
		}
		if (distSet.size() > K)
		{
			distSet.pop_back();
			subsetList.pop_back();
		}
	}

	std::list<int>::iterator it = subsetList.begin();

	for (;it != subsetList.end(); it ++)
	{
		subset->insert(*it);
	}
}

void CRBFCenterNetworkKDTree::initNetwork()
{
	std::vector<CRBFBasisFunction *>::iterator it = centers->begin();

	for (; it != centers->end(); it ++)
	{
		dataset->addInput((*it)->getCenter());
	}

	if (tree)
	{
		delete tree;
		delete nearestNeighbors;
	}

	tree = new CKDTree(dataset, 1);
	nearestNeighbors = new CKNearestNeighbors(tree, dataset, K);
}



CRBFCenterNetworkKDTree::CRBFCenterNetworkKDTree(int numDim, unsigned int l_K) : CRBFCenterNetwork(numDim)
{
	K = l_K;
	tree = NULL;
	nearestNeighbors = NULL;

	dataset = new CDataSet(getNumDimensions());
}

CRBFCenterNetworkKDTree::~CRBFCenterNetworkKDTree()
{
	if (tree)
	{
		delete tree;
		delete nearestNeighbors;
	}
	delete dataset;
}

void CRBFCenterNetworkKDTree::getNearestNeigbors(ColumnVector *state, unsigned int l_K,  DataSubset *subset)
{
	subset->clear();

	std::list<int> subsetList;	

	nearestNeighbors->getNearestNeighbors(state, &subsetList, l_K);

	std::list<int>::iterator it = subsetList.begin();

	for (;it != subsetList.end(); it ++)
	{
		subset->insert(*it);
	}
	
}



CRBFCenterFeatureCalculator::CRBFCenterFeatureCalculator(CStateProperties *stateProperties, CRBFCenterNetwork *l_network, unsigned int l_K) : CFeatureCalculator(l_network->getNumCenters() + 1, l_K + 1)
{
	network = l_network;
	
	normalizeFeat = true;
	useBias = false;

	setOriginalState(stateProperties);
}

CRBFCenterFeatureCalculator::~CRBFCenterFeatureCalculator()
{
	
}

void CRBFCenterFeatureCalculator::getModifiedState(CStateCollection *stateCol, CState *targetState)
{
	CState *state = stateCol->getState( originalState);

	DataSubset subset;

	
	targetState->resetState();

	
	

	// set Feature 0 for bias
	int bias = 0;
	if (useBias)
	{
		targetState->setContinuousState(0, 1.0);		
		targetState->setDiscreteState(0, 0);

		bias = 1;	
	}	

	network->getNearestNeigbors( state, getNumActiveFeatures() - bias  , &subset);
	
	targetState->setNumActiveContinuousStates( subset.size() + bias);
	targetState->setNumActiveDiscreteStates( subset.size() + bias);
	
	double sumFac = 0.0;

	DataSubset::iterator it = subset.begin();	
	for (int i = 0; it != subset.end(); it ++, i++)
	{
		CRBFBasisFunction *basisFunction = network->getCenter( *it);
		
		
		targetState->setDiscreteState(i + bias, *it + bias );
		double factor = basisFunction->getActivationFactor( state);
		targetState->setContinuousState(i + bias, factor);
		sumFac += factor;
	}

	if (normalizeFeat)
	{
		for (unsigned int i = 0; i < subset.size(); i ++)
		{
			targetState->setContinuousState( i + bias, targetState->getContinuousState( i + bias) / sumFac);
		}
	}
}
