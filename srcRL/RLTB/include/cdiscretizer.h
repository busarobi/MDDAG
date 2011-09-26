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

#ifndef C_DISCRETIZER__H
#define C_DISCRETIZER__H

#include "cstatemodifier.h"

#include <map>

/// Interface for all state discretizer.
/**Normal state discretization is done by all subclasses of CAbstractStateDiscretizer. CAbstractStateDiscretizer is a subclass 
of CStateModifier. Normal state discretization assigns a single discrete state index to the current model state. 
This number is calculated by the function int getDiscreteStateNumber(CStateCollection *), which has to be implemented by all subclasses. This function is called by the getModifiedState method and registered in the target state. 
<p>
CAbstractStateDiscretizer also offers you the possibility to make state substitutions, 
which is done by addStateSubstitution. With state substitutions you can replace a special 
discrete state number with another state from a specified modifier. 
This is needed if you want a more precise resolution of the model state only for some specific discrete state numbers. 
It is also possible to add a feature state as state substitution. The state discretizer produces then feature states instead of discrete states. 
*/

class CAbstractStateDiscretizer : public CStateModifier
{
protected:
	/// list of state substitutions
	std::map<int, std::pair<CStateModifier *, CState *>*> *stateSubstitutions;

public:
	/// creates a state discretizer which with numState different states.
	CAbstractStateDiscretizer(unsigned int numStates);
	virtual ~CAbstractStateDiscretizer();

/// Returns the discrete State size of discretizer, this is the discrete state of the 1st state variable.
	virtual unsigned int getDiscreteStateSize();
/// Function used to determine the State index from a state collection (usually from the model state)
	virtual unsigned int getDiscreteStateNumber(CStateCollection *state) = 0;	

/// Registers the discrete state number into the modified state object.
/**
The state number is calculated by the interface function getDiscreteStateNumber. The getModifiedState method passes through all state substitutions until the calculated state number is reached, summing up all discrete state sizes of the modifiers from the substitutions. 
This sum is than added to the calculated discrete state number to make the state index unique again. 
Whenever a substitution has been assigned to the current state number, the state to substitute is calculated (or taken from the state collection). Then the state is stored in the target state, and the the calculated discrete state number + the sum of the state sizes is added to all discrete state variables of the state substitutions. 
*/
	virtual void getModifiedState(CStateCollection *originalState, CState *modifiedState);

/// Adds a state substitution for a discrete state index
	virtual void addStateSubstitution(int discState, CStateModifier *modifier);
/// Removes the state substitution for the specified state
	virtual void removeStateSubstitution(int discState);
};

/// Class calculating a single discrete state from several state variables
/** The class calcualtes from several discrete state variables of the original state (defined by the given state properties in the constructor) a single discrete state number. This is done by an "and" combination of the discrete state variables.  You can also choose which discrete state variables you want in the constructor.
@see CDiscreteStateOperatorAnd
*/

class CModelStateDiscretizer : public CAbstractStateDiscretizer
{
protected:
	CStateProperties *originalState;
	unsigned int numDiscStateVar;
	int *discreteStates;

	unsigned int calcDiscreteStateSize(CStateProperties *prop, int *discreteStates, unsigned int num);

public:
/// Creates an ModelStateDiscretizer which use the state defined by the properties as original state.
/**
With the parameter *discretStates you can choose the discrete state variables you want for your new discrete state. If you omit this parameter
all discrete state from the original state are choosen. 
The properties object are the properties of the original state (most time the model state)
*/
	CModelStateDiscretizer(CStateProperties *properties, int *discretStates = NULL, unsigned int numDiscreteStates = 0);
	virtual ~CModelStateDiscretizer();

/// Calculates the discrete State Number using the and operator for all discrete state variable of the orignal state
	virtual unsigned int getDiscreteStateNumber(CStateCollection *state);	
};

/// Discretizes the original States i'th continuous state into the specified partitions.
/**The class serves as tools so that you don't have to create your own state discretizer classes, you can build your discrete state with already available classes. 
This class only discretize a single continuous state of the model state into given partitions. You can specify an double array as partition, and which continuous state variable should be used of the model state. 
The discretizer than calculates the partition in which the continuous state variable is located and returns its number. 
Since the values in the partition array are the limits of the partitions, the discrete state size is the partition array's size + 1.
The discrete states created by CSingleStateDiscretizer can then be combined by CDiscreteStateOperatorAnd.
*/

class CSingleStateDiscretizer : public CAbstractStateDiscretizer
{
protected:
/// The index of the continuous state variable to discretize
	int dimension;
	int numPartitions;
	double *partitions;

/// The original state which's state variable is discretized	
	CStateProperties *originalState;

public:
	CSingleStateDiscretizer(int dimension, int numPartitions, double *partitions);
	virtual ~CSingleStateDiscretizer();
/// Discretizes the specified continuous state variable with the partitions array.
	virtual unsigned int getDiscreteStateNumber(CStateCollection *state);	
/// Set the original State
	virtual void setOriginalState(CStateProperties *originalState);
};

/// The and operator combines several discrete states to one discrete state
/** Combination the discrete states by the and operator. Several discrete state variables are transformed to one non-ambiguous discrete state variable. The state size of the new state is the product of all state sizes. 
<p>
When the operator is created, it is empty and has no discrete states to combine, you can add as many discrete states as you want with the function addStateModifier. 
The class is often use in combination with CSingleStateDiscretizer because here you have several discrete states variables comming from the single state discretizer (usually you will have 1 discretizer for each dimension of your model state) and you need to combine these discrete state numbers to one unique discrete state number for your learners. 
*/

class CDiscreteStateOperatorAnd : public CStateMultiModifier, public CAbstractStateDiscretizer
{
public:
	CState *stateBuf;

public:
	CDiscreteStateOperatorAnd();
	virtual ~CDiscreteStateOperatorAnd();
/// Calculates the discrete state index as a "and" combination of all added discrete states.
	virtual unsigned int getDiscreteStateNumber(CStateCollection *state);

/// Add the discrete state to the operator
	virtual void addStateModifier(CAbstractStateDiscretizer *featCalc);
};

#endif

