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

#include "cstateproperties.h"
#include "ril_debug.h"
#include <assert.h>
#include <memory.h>
#include <stdlib.h>
#include <math.h>


CStateProperties::CStateProperties(unsigned int continuousStates, unsigned int discreteStates, int type)
{
	initProperties(continuousStates, discreteStates, type);
}


CStateProperties::CStateProperties(CStateProperties *properties)
{
	unsigned int i;

	initProperties(properties->getNumContinuousStates(), properties->getNumDiscreteStates(), properties->getType());

	for (i = 0; i < discreteStates; i++)
	{
		discreteStateSize[i] = properties->getDiscreteStateSize(i);
	}
	for (i = 0; i < continuousStates; i++)
	{
		minValues[i] = properties->getMinValue(i);
		maxValues[i] = properties->getMaxValue(i);
		isPeriodic[i] = false;
	}
}

CStateProperties::CStateProperties()
{
	this->continuousStates = 0;
	this->discreteStates = 0;

	this->discreteStateSize = NULL;
	minValues = NULL;
	maxValues = NULL;
	isPeriodic = NULL;
	bInit = false;
}

int CStateProperties::getType()
{
	return type;
}

void CStateProperties::addType(int Type)
{
	type = type | Type;	
}

bool CStateProperties::isType(int type)
{
	return (this->type & type) > 0;
}

void CStateProperties::initProperties(unsigned int continuousStates, unsigned int discreteStates,int type)
{
	this->continuousStates = continuousStates;
	this->discreteStates = discreteStates;

	this->discreteStateSize = new unsigned int[discreteStates];
	minValues = new double[continuousStates];
	maxValues = new double[continuousStates];
	isPeriodic = new bool[continuousStates];

	memset(discreteStateSize, 0, discreteStates * sizeof(unsigned int));
	for (unsigned int i = 0; i < continuousStates; i++)
	{
		minValues[i] = 0.0;
		maxValues[i] = 1.0;
		isPeriodic[i] = false;
	} 

	this->type = type;
	bInit = true;
}


void CStateProperties::setMinValue(unsigned int dim, double value)
{
	assert(dim < continuousStates);
	minValues[dim] = value;
}

double CStateProperties::getMinValue(unsigned int dim)
{
	assert(dim < continuousStates);
	return minValues[dim];
}
	
void CStateProperties::setMaxValue(unsigned int dim, double value)
{
	assert(dim < continuousStates);
	maxValues[dim] = value;
}

double CStateProperties::getMaxValue(unsigned int dim)
{
	assert(dim < continuousStates);
	return maxValues[dim];
}

CStateProperties::~CStateProperties()
{
	if (discreteStateSize != NULL) delete [] discreteStateSize;
	delete [] minValues;
	delete [] maxValues;
	delete [] isPeriodic;
}

unsigned int CStateProperties::getNumContinuousStates()
{
	return continuousStates;
}

bool CStateProperties::equals(CStateProperties *object)
{
	unsigned int i;
	bool bEquals = object->getNumContinuousStates() == getNumContinuousStates();
	bEquals = bEquals && object->getNumDiscreteStates() == getNumDiscreteStates();

	for (i = 0; i < getNumDiscreteStates(); i++)
	{
		bEquals = bEquals && getDiscreteStateSize(i) == object->getDiscreteStateSize(i);
	}
	for (i = 0; i < getNumContinuousStates(); i++)
	{
		bEquals = bEquals && getMinValue(i) == object->getMinValue(i) && getMaxValue(i) == object->getMaxValue(i);
	}

	return bEquals;
}


unsigned int CStateProperties::getNumDiscreteStates()
{
	return discreteStates;
}

void CStateProperties::setDiscreteStateSize(unsigned int dim, unsigned int size)
{
	assert(dim < discreteStates);
	discreteStateSize[dim] = size;
}

unsigned int CStateProperties::getDiscreteStateSize(unsigned int dim)
{
	assert(dim < discreteStates);
	return discreteStateSize[dim];
}

unsigned int CStateProperties::getDiscreteStateSize() 
{
	unsigned int dim = 1;
	
	for (unsigned int i = 0; i < getNumDiscreteStates(); i ++)
	{
		dim *= getDiscreteStateSize(i);
	}
	return dim;
}

void CStateProperties::setPeriodicity(unsigned int index, bool isPeriodic)
{
	assert(index < continuousStates);
	this->isPeriodic[index] = isPeriodic;
}

bool CStateProperties::getPeriodicity(unsigned int index)
{
	assert(index < continuousStates);
	return isPeriodic[index];
}

double CStateProperties::getMirroredStateValue(unsigned int index, double value)
{
	double period = getMaxValue(index) - getMinValue(index);
	return value - floor((value - getMinValue(index)) / period) * period;
}
