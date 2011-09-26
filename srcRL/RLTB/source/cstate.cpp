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
#include "cstate.h"
#include "cenvironmentmodel.h"
#include "cstateproperties.h"
#include "cutility.h"

#include <assert.h>
#include <math.h>

CState::CState(CStateProperties *properties) : CStateObject(properties),ColumnVector(properties->getNumContinuousStates())
{
	//continuousState = getData();
	discreteState = new int[getNumDiscreteStates()];

	resetState();
}

CState::CState(CState *copy) : CStateObject(copy->getStateProperties()) , ColumnVector(copy->getStateProperties()->getNumContinuousStates())
{
	//continuousState = getData();
	discreteState = new int[getNumDiscreteStates()];

	resetState();

	setState(copy);
}

CState::CState(CEnvironmentModel *model) : CStateObject(model->getStateProperties()), ColumnVector(model->getStateProperties()->getNumContinuousStates())
{
	//continuousState = getData();
	discreteState = new int[getNumDiscreteStates()];

	resetState();
}

CState::CState(CStateProperties *properties, FILE *stream, bool binary) : CStateObject(properties), ColumnVector(properties->getNumContinuousStates())
{
	this->properties = properties;

	if (binary)
	{
		loadBinary(stream);
	}
	else
	{
		loadASCII(stream);
	}
	numActiveContinuousStates  = properties->getNumContinuousStates();
	numActiveDiscreteStates = properties->getNumDiscreteStates();
}


void CState::resetState()
{
	unsigned int i = 0;

	for (i = 0; i < getNumContinuousStates(); i++)
	{
		element(i) = 0;
	}
	for (i = 0; i < getNumDiscreteStates(); i++)
	{
		discreteState[i] = 0;
	}
	numActiveContinuousStates  = properties->getNumContinuousStates();
	numActiveDiscreteStates = properties->getNumDiscreteStates();

	b_resetState = false;
}

void CState::setState(CState *copy)
{
	unsigned int i = 0;
	assert(equalsModelProperties(copy));
	for (i = 0; i < getNumContinuousStates(); i++)
	{
		element(i) = copy->getContinuousState(i);
	}
	for (i = 0; i < getNumDiscreteStates(); i++)
	{
		discreteState[i] = copy->getDiscreteState(i);
	}

	numActiveContinuousStates  = copy->getNumActiveContinuousStates();
	numActiveDiscreteStates = copy->getNumActiveDiscreteStates();

	b_resetState = copy->isResetState();
}

CState::~CState()
{
//	delete [] continuousState;
	delete [] discreteState;
}

double CState::getContinuousState(unsigned int dim)
{
	assert(dim < getNumContinuousStates());
	return element(dim);
}

double CState::getNormalizedContinuousState(unsigned int dim)
{
	assert(dim < getNumContinuousStates());
	return (element(dim) - this->getStateProperties()->getMinValue(dim)) / 
		(this->getStateProperties()->getMaxValue(dim) - this->getStateProperties()->getMinValue(dim));
}

int CState::getDiscreteState(unsigned int dim)
{
	assert(dim < getNumDiscreteStates());
	return discreteState[dim];
}

int CState::getDiscreteStateNumber()
{
	int index = 0;
	int stateSize = 1;
	for (unsigned int i = 0; i < getNumDiscreteStates(); i++)
	{
		index += discreteState[i] * stateSize;
		stateSize *= properties->getDiscreteStateSize(i);
	}
	return index;
}

CState* CState::clone()
{
	return new CState(this);
}

void CState::setContinuousState(unsigned int dim, double value)
{
	assert(dim < getNumContinuousStates());
	element(dim)= value;

	if (properties->getPeriodicity(dim))
	{
		if ((element(dim) < properties->getMinValue(dim)) || (element(dim) > properties->getMaxValue(dim)))
		{
			double Period = (properties->getMaxValue(dim) - properties->getMinValue(dim));
			assert(Period > 0);
			element(dim) = element(dim) - Period * floor((element(dim) - properties->getMinValue(dim)) / Period);
		}
	}
	else
	{
		double minVal = properties->getMinValue(dim);
		if (this->element(dim) < minVal)
		{
			this->element(dim) = minVal;
		}
		else
		{
			double maxVal = properties->getMaxValue(dim);
			if (this->element(dim) > maxVal)
			{
				this->element(dim) = maxVal;
			}
		}
	}
}

void CState::setDiscreteState(unsigned int dim, int value)
{
	assert(dim < getNumDiscreteStates());
	if ((value < 0 || (unsigned int) value >= properties->getDiscreteStateSize(dim)) && properties->getDiscreteStateSize(dim) > 0)
	{
		fprintf(stderr, "Error: Setting discrete state variable %d to %d, State Size is %d\n", dim, value,  properties->getDiscreteStateSize(dim));
		assert(false);
	}
	
	discreteState[dim]= value;	
}

CState *CState::getState(CStateProperties *)
{
	return this;
}

CState *CState::getState()
{
	return this;
}

bool CState::isMember(CStateProperties *stateModifier)
{
	return getStateProperties() == stateModifier;
}

unsigned int CState::getNumActiveDiscreteStates()
{
	return numActiveDiscreteStates;
}

unsigned int CState::getNumActiveContinuousStates()
{
	return numActiveContinuousStates;
}

void CState::setNumActiveDiscreteStates(int numActiveStates)
{
	numActiveDiscreteStates = numActiveStates;
}

void CState::setNumActiveContinuousStates(int numActiveStates)
{
	numActiveContinuousStates = numActiveStates;
}


/*CActionSet* CState::getAvailableActions()
{
	return availableActions;
}

void CState::setAvailableActions(CActionSet *aset)
{
	CActionSet::iterator it;

	availableActions->clear();

	for (it = aset->begin(); it != aset->end(); it++)
	{
		availableActions->push_back(*it);
	}

}
*/

bool CState::equals(CState *state)
{
	unsigned int i;
	if (! this->equalsModelProperties(state))
	{
		return false;
	}
	for (i = 0; i < getNumContinuousStates(); i++)
	{
		if (getContinuousState(i) != state->getContinuousState(i))
		{
			return false;
		}
	}
	for (i = 0; i < getNumDiscreteStates(); i++)
	{
		if (getDiscreteState(i) != state->getDiscreteState(i))
		{
			return false;
		}
	}
	return true;
}


void CState::saveBinary(FILE *stream)
{
	for (unsigned int i = 0; i < getNumContinuousStates(); i++)
	{
		double buf = element(i);
		fwrite(&buf, sizeof(double), 1, stream);
	}
	fprintf(stream, "\n");
	fwrite(this->discreteState, sizeof(bool), getNumDiscreteStates(), stream);
	fprintf(stream, "\n");
}

void CState::loadBinary(FILE *stream)
{
	for (unsigned int i = 0; i < getNumContinuousStates(); i++)
	{
		double buf ;
		fread(&buf, sizeof(double), 1, stream);
		element(i) = buf;
	}
	fscanf(stream, "\n");
	fread(this->discreteState, sizeof(bool), getNumDiscreteStates(), stream);
	fscanf(stream, "\n");
}

void CState::saveASCII(FILE *stream)
{
	unsigned int i = 0;
	fprintf(stream, "[");
	for (i = 0; i < getNumContinuousStates(); i++)
	{
		fprintf(stream, "%lf ", getContinuousState(i));
	}
	for (i = 0; i < getNumDiscreteStates(); i++)
	{
		fprintf(stream, "%d ", (int)getDiscreteState(i));
	}
	fprintf(stream, "]");
}

void CState::loadASCII(FILE *stream)
{
	unsigned int i;
	int bBuf;
	fscanf(stream, "[");
	for (i = 0; i < getNumContinuousStates(); i++)
	{
		double buf;
		fscanf(stream, "%lf ", &buf);
		element(i)= buf;
	}
	for (i = 0; i < getNumDiscreteStates(); i++)
	{
		fscanf(stream, "%d ", &bBuf);
        discreteState[i] = (bBuf != 0);
	}
	fscanf(stream, "]");
}

double CState::getDistance(ColumnVector *vector)
{
	ColumnVector distance = (*this) - (*vector);
	assert(nrows() == vector->nrows());
	
	return distance.norm_Frobenius();
}


double CState::getSingleStateDifference(int i, double value)
{
	double distance = 0.0;
	if (properties->getPeriodicity(i))
	{
		double period = properties->getMaxValue(i) - properties->getMinValue(i);
		assert(period != 0);
		distance = (element(i)-value);
		if (distance < - period / 2)
		{
			distance += period;
		}
		else
		{
			if (distance > period / 2)
			{
				distance -= period;
			}
		}	
	}
	else
	{
		distance = element(i) - value;
	}
	return distance;
}


CStateList::CStateList(CStateProperties *properties) : CStateObject(properties)
{
	continuousStates = new std::vector<std::vector<double> *>();
	discreteStates = new std::vector<std::vector<int> *>();
	
	resetStates = new std::vector<bool>();

	unsigned int i;
	for (i = 0; i < properties->getNumContinuousStates(); i++)
	{
		continuousStates->push_back(new std::vector<double>());
	}
	for (i = 0; i < properties->getNumDiscreteStates(); i++)
	{
		discreteStates->push_back(new std::vector<int>());
	}

	numStates = 0;
}

CStateList::~CStateList()
{
	unsigned int i;
	for (i = 0; i < getNumContinuousStates(); i++)
	{
		delete (*continuousStates)[i];
	}
	for (i = 0; i < getNumDiscreteStates(); i++)
	{
		delete (*discreteStates)[i];
	}
	delete continuousStates;
	delete discreteStates;
	delete resetStates;
}

void CStateList::addState(CState *state)
{
	unsigned int i;
	assert(state->equalsModelProperties(this));

	for (i = 0; i < getNumContinuousStates(); i++)
	{
		(*continuousStates)[i]->push_back(state->getContinuousState(i));
	}
	for (i = 0; i < getNumDiscreteStates(); i++)
	{
		(*discreteStates)[i]->push_back(state->getDiscreteState(i));
	}
	resetStates->push_back(state->isResetState());
	
	numStates ++;
}

void CStateList::removeLastState()
{
	unsigned int i;
	for (i = 0; i < getNumContinuousStates(); i++)
	{
		(*continuousStates)[i]->pop_back();
	}
	for (i = 0; i < getNumDiscreteStates(); i++)
	{
		(*discreteStates)[i]->pop_back();
	}
	resetStates->pop_back();
	
	numStates --;
}

unsigned int CStateList::getNumStates()
{
	return numStates;
}

void CStateList::clear()
{
	unsigned int i;
	for (i = 0; i < getNumContinuousStates(); i++)
	{
		(*continuousStates)[i]->clear();
	}
	for (i = 0; i < getNumDiscreteStates(); i++)
	{
		(*discreteStates)[i]->clear();
	}
	resetStates->clear();
	numStates  = 0;
}

void CStateList::getState(unsigned int num, CState *state)
{
	unsigned int i;
	for (i = 0; i < getNumContinuousStates(); i++)
	{
		state->setContinuousState(i, (*(*continuousStates)[i])[num]);
	}
	for (i = 0; i < getNumDiscreteStates(); i++)
	{
		state->setDiscreteState(i, (*(*discreteStates)[i])[num]);
	}
	
	state->setResetState((*resetStates)[num]);
}

void CStateList::loadBIN(FILE *stream)
{
	unsigned int i, j, buf;
	double dBuf;
	int nBuf;
	buf = 0;
	
	fread(&buf, sizeof(int), 1, stream);
	
	for (i = 0; i < properties->getNumContinuousStates(); i++)
	{
		for (j = 0; j < buf; j++)
		{
			dBuf = 0.0;

			int r  = fread( &dBuf, sizeof(double), 1, stream);
			assert(r == 1);
			(*continuousStates)[i]->push_back(dBuf);
		}
	}

	for (i = 0; i < properties->getNumDiscreteStates(); i++)
	{
		for (j = 0; j < buf; j++)
		{
			int r = fread( &nBuf, sizeof(int), 1, stream);
			assert(r == 1);
			(*discreteStates)[i]->push_back(nBuf);
		}
	}	
	for (j = 0; j < buf; j++)
	{
		bool nBuf;
		int r = fread( &nBuf, sizeof(bool), 1, stream);
		assert(r == 1);
		resetStates->push_back(nBuf);
	}
	numStates = buf;
}

void CStateList::saveBIN(FILE *stream)
{
	int buf = getNumStates();
	unsigned int i, j;
    int nBuf;
    double dBuf;
	
	fwrite(&buf, sizeof(int), 1, stream);
	for (i = 0; i < properties->getNumContinuousStates(); i++)
	{
		for (j = 0; j < getNumStates(); j++)
		{
            dBuf = (*(*continuousStates)[i])[j];
			fwrite(&dBuf, sizeof(double), 1, stream);
		}
	}
	for (i = 0; i < properties->getNumDiscreteStates(); i++)
	{
		for (j = 0; j < getNumStates(); j++)
		{
            nBuf = (*(*discreteStates)[i])[j];
			fwrite( &nBuf, sizeof(int), 1, stream);
		}
	}
	
	for (j = 0; j < getNumStates(); j++)
	{
       	bool bBuf = (*resetStates)[j];
		fwrite(&bBuf, sizeof(bool), 1, stream);
	}
	
}

void CStateList::loadASCII(FILE *stream)
{
	int buf1, buf2, res;
	unsigned int i, j, buf;

	double dBuf;
	int bBuf;
	res = fscanf(stream,"States: %d\n", &buf);
	assert(res == 1);
	fscanf(stream, "\n");
	for (i = 0; i < properties->getNumContinuousStates(); i++)
	{
		res = fscanf(stream, "ContinuousState %d:\n", &buf1);
		assert(res == 1);
		for (j = 0; j < buf; j++)
		{
			assert(fscanf(stream, "%lf ", &dBuf) == 1);

			(*continuousStates)[i]->push_back(dBuf);
		}
		fscanf(stream, "\n");
	}
	for (i = 0; i < properties->getNumDiscreteStates(); i++)
	{
		res = fscanf(stream, "DiscreteState %d:\n", &buf2);
		assert(res == 1);
		for (j = 0; j < buf; j++)
		{
			assert(fscanf( stream, "%d ", &bBuf) == 1);
			(*discreteStates)[i]->push_back(bBuf);
		}
		fscanf(stream, "\n");
	}
	res = fscanf(stream, "Reset States:\n");
	assert(res == 0);
	for (j = 0; j < buf; j++)
	{
		assert(fscanf( stream, "%d ", &bBuf) == 1);
		resetStates->push_back((bool) bBuf);
	}
	fscanf(stream, "\n");
	
	numStates = buf;
}

void CStateList::saveASCII(FILE *stream)
{
	fprintf(stream,"States: %d\n", getNumStates());
	fprintf(stream, "\n");
	unsigned int i, j;
	for (i = 0; i < properties->getNumContinuousStates(); i++)
	{
		fprintf(stream, "ContinuousState %d:\n", i);
		for (j = 0; j < getNumStates(); j++)
		{
			fprintf(stream, "%f ", (*(*continuousStates)[i])[j]);
		}
		fprintf(stream, "\n");
	}
	for (i = 0; i < properties->getNumDiscreteStates(); i++)
	{
		fprintf(stream, "DiscreteState %d:\n", i);
		for (j = 0; j < getNumStates(); j++)
		{
			fprintf( stream, "%d ", (int)((*(*discreteStates)[i])[j]));
		}
		fprintf(stream, "\n");
	}
	
	fprintf(stream, "Reset States:\n");
	
	for (j = 0; j < getNumStates(); j++)
	{
		fprintf( stream, "%d ", (int) (*resetStates)[j]);
	}
	fprintf(stream, "\n");
}

void CStateList::initWithDiscreteStates(CState *initState, vector<int> dimensions)
{
	int numDiscStates = properties->getNumDiscreteStates();

   	int numTakenStates = dimensions.size();
   	assert(numTakenStates > 0);
  
   	int i, j, dim;
   	// Calculate Size of State List
   	vector<int> stateSize(numTakenStates);
   	vector<int> curState(numTakenStates);
   	int numStates = 1;
   	for (i=0; i<numTakenStates; i++) 
   	{
    		dim = dimensions[i];
    		assert(dim >= 0 && dim < numDiscStates);

    		stateSize[i] = properties->getDiscreteStateSize(dim);
    		if (stateSize[i] > 0)
      		{
			numStates *= stateSize[i];
    		}
		curState[i] = 0;
  	}

  	// Generate all States
  	CState dummyState(initState);

  	bool increment;
  	int curIndex;

  	for (i=0; i<numStates; i++) 
  	{
    		for (j=0; j<numTakenStates; j++) 
    		{
      			dim = dimensions[j];
      			dummyState.setDiscreteState(dim, curState[j]);
    		}

    		addState(&dummyState);

    		// Increment state
    		curIndex = 0;
    		increment = true;
    		while (increment) 
		{
      			curState[curIndex]++;
      			if (curState[curIndex] < stateSize[curIndex])
			{
				increment = false;
			}
      			else 
			{
				curState[curIndex] = 0;
				curIndex++;
			}
			if (curIndex >= numTakenStates)
			    increment = false;
      		}
    	}
  
}

void CStateList::initWithContinuousStates(CState *initState, vector<int> dimensions, vector<double> partitions)
{

	int numContStates = properties->getNumContinuousStates();	
   	int numTakenStates = dimensions.size();
   	assert(numTakenStates > 0);
  
   	int i, j, dim;
   	// Calculate Size of State List
   	vector<int> stateSize(numTakenStates);
   	vector<int> curState(numTakenStates);
   	
	int numStates = 1;
   	for (i=0; i<numTakenStates; i++) 
   	{
    		dim = dimensions[i];
    		assert(dim >= 0 && dim < numContStates);

		numStates *= (partitions[i] + 1);

		curState[i] = 0;
  	}

  	// Generate all States
  	CState dummyState(initState);

  	bool increment;
  	int curIndex;

  	for (i=0; i<numStates; i++) 
  	{
    		for (j=0; j<numTakenStates; j++) 
    		{
      			dim = dimensions[j];

			double width = properties->getMaxValue(dim) - properties->getMinValue(dim);
			double x = properties->getMinValue(dim) + curState[j] * width / partitions[j];

      			dummyState.setContinuousState(dim, x);
    		}

    		addState(&dummyState);

    		// Increment state
    		curIndex = 0;
    		increment = true;
    		while (increment) 
		{
      			curState[curIndex]++;
      			if (curState[curIndex] < stateSize[curIndex])
			{
				increment = false;
			}
      			else 
			{
				curState[curIndex] = 0;
				curIndex++;
			}
			if (curIndex >= numTakenStates)
			    increment = false;
      		}
    	}
}
