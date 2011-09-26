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

#ifndef C_ADAPTIVESOFTMAX__H
#define C_ADAPTIVESOFTMAX__H

#include <stdio.h> 
#include <vector>
#include <list>
#include <map>

#include "clearndataobject.h"
#include "cstatemodifier.h"

class CRBFBasisFunction;

class DataSubset;
class CDataSet;

class CKDTree;
class CKNearestNeighbors;

class CStateCollection;
class CState;
class CStateProperties;

class CRBFCenterNetwork : public CLearnDataObject
{
protected:
    int numDim;
    
    std::vector<CRBFBasisFunction*> *centers;
    
    virtual void initNetwork() {};
    
    CRBFCenterNetwork(std::vector<CRBFBasisFunction *> *centers, int numDim);
public:
    
    CRBFCenterNetwork(int numDim);
    
    virtual ~CRBFCenterNetwork();
    
    virtual void getNearestNeigbors(ColumnVector *state, unsigned int K,  DataSubset *subset) = 0;
    
    virtual void saveData(FILE *stream);
    virtual void loadData(FILE *stream);
    
    virtual void resetData() {};
    virtual void addCenter(CRBFBasisFunction* center);
    
    CRBFBasisFunction *getCenter(int index);
    int getNumCenters();
    
    virtual void deleteCenters();
    
    int getNumDimensions();
};

class CRBFCenterNetworkSimpleSearch : public CRBFCenterNetwork
{
protected:
    virtual void initNetwork() {};
    
    
public:
    CRBFCenterNetworkSimpleSearch(int numDim);
    
    virtual ~CRBFCenterNetworkSimpleSearch();
    
    
    virtual void getNearestNeigbors(ColumnVector *state, unsigned int K,  DataSubset *subset);
};

class CRBFCenterNetworkKDTree : public CRBFCenterNetwork
{
protected:
    CDataSet *dataset;
    
    virtual void initNetwork();
    
    unsigned int K;
    
    CKDTree *tree;
    CKNearestNeighbors *nearestNeighbors;
public:
    CRBFCenterNetworkKDTree(int numDim, unsigned int K);
    
    virtual ~CRBFCenterNetworkKDTree();
    
    virtual void getNearestNeigbors(ColumnVector *state, unsigned int K,  DataSubset *subset);
};

class CRBFCenterFeatureCalculator : public CFeatureCalculator
{
protected:
	CRBFCenterNetwork *network;
    
public:
	bool useBias;
	bool normalizeFeat;
    
	CRBFCenterFeatureCalculator(CStateProperties *stateProperties, CRBFCenterNetwork *network, unsigned int K);
	virtual ~CRBFCenterFeatureCalculator();
    
	virtual void getModifiedState(CStateCollection *state, CState *targetState);
    virtual void saveData(FILE *stream) {
        network->saveData(stream);
    }

    virtual void addCenter(CRBFBasisFunction* center) {
        network->addCenter(center);
    }
    
};


#endif

