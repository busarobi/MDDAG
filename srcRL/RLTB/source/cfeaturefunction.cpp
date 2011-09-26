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



#include "cfeaturefunction.h"
#include "ril_debug.h"

#include <assert.h>
#include <math.h>


CFeature::CFeature(unsigned int Index, double updateValue)
{
	this->featureIndex = Index;
	this->factor = updateValue;
}

CFeature::CFeature()
{
	this->featureIndex = 0;
	this->factor = 0.0;
}

CFeature::~CFeature()
{
	featureIndex = 0;
}

CFeatureList::CFeatureList(int initMemSize, bool isSorted, bool sortAbs)
{
	freeResources = new std::list<CFeature *>();
	allResources = new std::list<CFeature *>();

	featureMap = new std::map<int, CFeature *>();

	for (int i = 0; i < initMemSize; i++)
	{
		CFeature *feat = new CFeature(0,0.0);
		freeResources->push_back(feat);
		allResources->push_back(feat);
	}
	this->isSorted = isSorted;
	this->sortAbs = sortAbs;
}

CFeatureList::~CFeatureList()
{
	delete freeResources;
	delete featureMap;

	std::list<CFeature *>::iterator it = allResources->begin();

	for (int i = 0; it != allResources->end(); it ++, i ++)
	{
		delete *it;
	}
}

CFeature *CFeatureList::getFreeFeatureResource()
{
	CFeature *freeFeat=NULL;
	if (freeResources->size() > 0)
	{
		freeFeat = *freeResources->begin();
		freeResources->pop_front();
	}
	else
	{
		freeFeat = new CFeature(0,0);
		allResources->push_back(freeFeat);
	}
	return freeFeat;
}

void CFeatureList::sortFeature(CFeature *feature)
{
	std::list<CFeature *>::remove(feature);
	CFeatureList::iterator it = begin();
	if (sortAbs)
	{
		while (it != end() && fabs((*it)->factor) > fabs(feature->factor))
		{
			it ++;
		}
	}
	else
	{
		while (it != end() && (*it)->factor > feature->factor)
		{
			it ++;
		}
	}
	
	std::list<CFeature *>::insert(it, feature);
}

void CFeatureList::add(CFeature *feature)
{
	CFeature *feat = NULL;
	if ((*featureMap)[feature->featureIndex] == NULL)
	{
		feat = getFreeFeatureResource();
		feat->featureIndex = feature->featureIndex;
		feat->factor = feature->factor;
		push_back(feat);
		(*featureMap)[feature->featureIndex] = feat;

	}
	else
	{
		feat = (*featureMap)[feature->featureIndex];
		feat->factor += feature->factor;
	}
	if (isSorted)
	{
		sortFeature(feat);
	}
}

void CFeatureList::add(CFeatureList *featureList, double factor)
{
	CFeatureList::iterator it = featureList->begin();

	for (;it != featureList->end(); it ++)
	{
		update((*it)->featureIndex, (*it)->factor * factor);
	}
}

double CFeatureList::multFeatureList(CFeatureList *featureList)
{
	CFeatureList::iterator it = featureList->begin();

	double sum = 0.0;
	for(; it != featureList->end(); it ++)
	{
		sum += getFeatureFactor((*it)->featureIndex) * (*it)->factor;
	}
	return sum;
}



void CFeatureList::set(int feature, double factor)
{
	CFeature *feat = NULL;
	if ((*featureMap)[feature] == NULL)
	{
		feat = getFreeFeatureResource();
		feat->featureIndex = feature;
		feat->factor = factor;
		push_back(feat);
		(*featureMap)[feature] = feat;

	}
	else
	{
		feat = (*featureMap)[feature];
		feat->factor = factor;
	}
	if (isSorted)
	{
		sortFeature(feat);
	}
}

void CFeatureList::update(int feature, double factor)
{
	CFeature *feat = NULL;
	if ((*featureMap)[feature] == NULL)
	{
		feat = getFreeFeatureResource();
		feat->featureIndex = feature;
		feat->factor = factor;
		push_back(feat);
		(*featureMap)[feature] = feat;

	}
	else
	{
		feat = (*featureMap)[feature];
		feat->factor += factor;
	}
	if (isSorted)
	{
		sortFeature(feat);
	}
}


double CFeatureList::getFeatureFactor(int featureIndex)
{
	if ((*featureMap)[featureIndex] != NULL)
	{
		return (*featureMap)[featureIndex]->factor;
	}
	else
	{
		return 0.0;
	}
}
	
CFeature* CFeatureList::getFeature(int featureIndex)
{
	return (*featureMap)[featureIndex];
}

void CFeatureList::normalize()
{
	CFeatureList::iterator it = begin();

	double sum = 0.0;
	for(; it != end(); it ++)
	{
		sum += (*it)->factor;
	}

	multFactor(1.0 / sum);
}

double CFeatureList::getLength()
{
	CFeatureList::iterator it = begin();

	double sum = 0.0;
	for(; it != end(); it ++)
	{
		sum += pow((*it)->factor, (double) 2.0);
	}

	return sqrt(sum);
}

void CFeatureList::remove(CFeature *feature)
{
	remove(feature->featureIndex);
}

void CFeatureList::remove(int feature)
{
	CFeature *feat = (*featureMap)[feature];
	if (feat != NULL)
	{
		std::list<CFeature *>::remove(feat);
		freeResources->push_back(feat);
		featureMap->erase(feature);
	}
}

void CFeatureList::clear()
{
	CFeatureList::iterator it = begin();
	for (; it !=end(); it ++)
	{
		freeResources->push_back(*it);
	}
	std::list<CFeature *>::clear();
	featureMap->clear();
}

void CFeatureList::multFactor(double factor)
{
	CFeatureList::iterator it = begin();
//	printf("Multiplying Feature List... %d, %d Features\n", size(), freeResources->size());
	for (int i = 0; it !=end(); it ++, i++)
	{
		(*it)->factor *= factor;
/*		printf("%d", i);
		it ++;
		printf("_");
    	it !=end();
    	printf("_");*/
	}
}

void CFeatureList::addIndexOffset(int offset)
{
	CFeatureList::iterator it = begin();
	this->featureMap->clear();
	for (; it !=end(); it ++)
	{
		(*it)->featureIndex += offset;
		(*featureMap)[(*it)->featureIndex] = *it;
	}
}


CFeatureList::iterator CFeatureList::getFeaturePos(unsigned int feature)
{
	CFeatureList::iterator it = begin();
	for (int i = 0; it != end(); it ++, i ++)
	{
		if ((*it)->featureIndex >= feature)
		{
			break;
		}
	}
	return it;
}

void CFeatureList::saveASCII(FILE *stream)
{
	CFeatureList::iterator it = begin();

	fprintf(stream, "[");
	for (; it != end(); it++)
	{
		fprintf(stream, "(%d,%1.3f)", (*it)->featureIndex, (*it)->factor);
	}
	fprintf(stream, "]");
}

void CFeatureList::loadASCII(FILE *stream)
{
	clear();
	CFeatureList::iterator it = begin();

	fscanf(stream, "[");
	for (; it != end(); it++)
	{
		int index;
		double buffer;
		fscanf(stream, "(%d,%lf)", &index, &buffer);
		set(index, buffer);
	}
	fscanf(stream, "]");
}


CFeatureList::iterator CFeatureList::begin()
{
	return std::list<CFeature *>::begin();
}

CFeatureList::iterator CFeatureList::end()
{
	return std::list<CFeature *>::end();
}

CFeatureList::reverse_iterator CFeatureList::rbegin()
{
	return std::list<CFeature *>::rbegin();
}

CFeatureList::reverse_iterator CFeatureList::rend()
{
	return std::list<CFeature *>::rend();
}

CFeatureFunction::CFeatureFunction(unsigned int numFeatures)
{
	externFeatures = false;

	this->numFeatures = numFeatures;
	features = new double[numFeatures];

	for (unsigned int i = 0; i < numFeatures; i++)
	{
		 features[i] = 0.0;
	}
}

CFeatureFunction::CFeatureFunction(unsigned int numFeatures, double *features)
{
	externFeatures = true;
	this->numFeatures = numFeatures;
	this->features = features;
}


CFeatureFunction::~CFeatureFunction()
{
	if (!externFeatures) delete features;
}

void CFeatureFunction::randomInit(double min, double max)
{
	for (unsigned int i = 0; i < numFeatures; i++)
	{
		features[i] = ((double) rand()) / ((double) RAND_MAX) * (max - min) + min;
	}
}

void CFeatureFunction::init(double value)
{
	for (unsigned int i = 0; i < numFeatures; i++)
	{
		features[i] = value;
	}
}


void CFeatureFunction::setFeature(CFeature *update, double value)
{
	assert(update->featureIndex < numFeatures);

	features[update->featureIndex] = value * update->factor;
}

void CFeatureFunction::setFeature(unsigned int featureIndex, double value)
{
	features[featureIndex] = value;
}

void CFeatureFunction::setFeatureList(CFeatureList *updateList, double value)
{
	CFeatureList::iterator it = updateList->begin();
	for (; it != updateList->end(); it ++)
	{
		setFeature(*it, value);
	}
}

void CFeatureFunction::updateFeature(CFeature *update, double difference)
{
	assert(update->featureIndex < numFeatures);

	double oldV = features[update->featureIndex];
	features[update->featureIndex] += difference * update->factor;

	DebugPrint('f', "Feature (%d %f) Update: %2.4f -> %2.4f\t\n", update->featureIndex,update->factor, oldV, features[update->featureIndex]);
}

void CFeatureFunction::updateFeature(int feature, double difference)
{
	features[feature] += difference;
}

void CFeatureFunction::updateFeatureList(CFeatureList *updateList, double difference)
{
	CFeatureList::iterator it = updateList->begin();

	for (; it != updateList->end(); it ++)
	{
		updateFeature(*it, difference);
	}
}

double CFeatureFunction::getFeature(unsigned int featureIndex)
{
	assert(featureIndex < numFeatures);

	return features[featureIndex];
}

double CFeatureFunction::getFeatureList(CFeatureList *featureList)
{
	double sum = 0.0;
	CFeatureList::iterator it = featureList->begin();

	for (; it != featureList->end(); it ++)
	{
		sum += (*it)->factor * getFeature((*it)->featureIndex);
	}

	return sum;
}

void CFeatureFunction::saveFeatures(FILE *stream) {
    fprintf (stream, "FeatureFunction (Features: %d)\n", numFeatures);
    for (unsigned int i = 0; i < numFeatures; i++) 
	{
        fprintf (stream, " %8.4lf", features[i]);          
    }
	fprintf(stream, "\n");
}

void CFeatureFunction::loadFeatures(FILE *stream) {
    
    unsigned int nfields, tmpnumFeatures;
	
	char tmpName[100];

    nfields = fscanf (stream, "%s (Features: %d)\n", tmpName, &tmpnumFeatures);

	if (features != NULL)
	{
		assert(tmpnumFeatures == numFeatures);	
	}
	else
	{
		numFeatures = tmpnumFeatures;
		features = new double[numFeatures];
	}

	 for (unsigned int j = 0; j < numFeatures; j++) 
	{
         assert(fscanf (stream, " %lf", &features[j]) == 1);			
    }
    fscanf(stream, "\n");			
}

void CFeatureFunction::printFeatures()
{
    saveFeatures(stdout);	
}

unsigned int CFeatureFunction::getNumFeatures()
{
	return numFeatures;
}

void CFeatureFunction::postProcessWeights(double mean, double std)
{
	// de-normalizaton if the output has been normalized, works only if feature[0] is the bias!!
	for (unsigned int i = 0; i < numFeatures; i ++)
	{
		features[i] *= std;
	}
	features[0] += mean;
}
