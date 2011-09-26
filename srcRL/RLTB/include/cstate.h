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

#ifndef CMODELSTATE_H
#define CMODELSTATE_H


#include "cbaseobjects.h"

#include <vector>
#include <stdio.h>
#include "newmat/newmat.h"


class CEnvironmentModel;
class CState;

/// Interface for a state collection
/** From a state collection you can retrieve the specific state you want. A state collection has one general state (usually the model state) and many other
states comining from state modifiers. 
This interface only provides functions to gather states from the collection. The only thing you need is to call getState
with the specific state properties object. Therefore state properties object pointer serves as the "ID" of the state in
the collection, so it doesn't work to transfer a properties object with just the same properties but which is another instance.
If the getState method is called without parameter, the "general" state is calculated. 
<p>
Implementations of CStateCollections are CStateCollectionImpl and CState itself. The CState class can be seen as state collection 
only containing the CState object itself. This implentation was choosen in order to provide more generality for many functions.
*/
class CStateCollection
{
protected:
	bool b_resetState;
	
public:
	CStateCollection() {b_resetState = false;};
	virtual ~CStateCollection() {};
/// retrieves the state with the specified properties
	virtual CState *getState(CStateProperties *properties) = 0;
/// retrieves the general state
	virtual CState *getState() = 0;
/// Checks wether there exists a state with the specified state properties in the collection
	virtual bool isMember(CStateProperties *stateModifier) = 0;

	virtual bool isResetState() {return b_resetState;};
	virtual void setResetState(bool reset) {b_resetState = reset;};
};


/// Represents a single state 
/** 
The CState object saves the values of state-variables. There are discrete and continuous state variables. 
The number of discrete and continuous state variables are taken from the CStateProperties object. These values
stand for the maximum number of continuous resp. discrete states, the arrays are created with this size.
The attributes numActiveContinuousStates resp numActiveDiscreteStates stand for the used state variables by the
actual state, these values can differ from the maximum values especially with feature states. These values are used
to gain performance when updtating the Q-Function, so you only have to look at the first, active state variables.
States can be of type DISCRETESTATE, FEATURESTATE or a normal state. The type is specified in the state properties.
<p>
CState implents the CStateCollection interface, the class can be seen as state collection only containing the CState object itself. 
This implentation was choosen in order to provide more generality for many functions which take only a state collection.
@see CStateProperties
*/
class CState : public CStateObject,public CStateCollection, public ColumnVector
{
protected:

/// array for the discrete states
	int *discreteState;

/// number of active continuous state-variables used by the actual state
	unsigned int numActiveContinuousStates;
/// number of active discrete state-variables used by the actual state
	unsigned int numActiveDiscreteStates;

public:
/// Creates a new State with the specified properties
	CState(CStateProperties *properties);
/// Creates a new State with the properties of the model
	CState(CEnvironmentModel *model);
/// Copies the given state
	CState(CState *copy);
/// Loads a state from file
/** The file stream has to stand on the right position, and the properties have to be the same (the same values in the properties)
as the properties from the saved State.
@param properties Properties of the state
@param stream Stream from where to load the state
@param binary whether to load the state from a binary or a txt file.*/
	CState(CStateProperties *properties, FILE *stream, bool binary = true);
/// Sets all the values according to @param copy . The properties have to be the same.
	void setState(CState *copy);
/// Resets the state
/** Sets all state-variables to 0, sets numActiveContinuousStates and numActiveDiscreteStates to the values given by the properties,
so that all the state-variables are used.*/
	void resetState();

	virtual ~CState();

/// Implements the CStateCollection interface
/**Always returns the state itself. So the returned state doesn't have to have the specified properties!*/
	virtual CState *getState(CStateProperties *properties);
/// Implements the CStateCollection interface
/**A lways returns the state itself, so it is the "general" state of its own state collection.*/
	virtual CState *getState();

/** Checks wether there exists a state with the specified state properties in the collection
	Actually since there is just one state in the "state collection" it just returns if the properties pointer of
	the state is the same as "stateModifier" */
	virtual bool isMember(CStateProperties *stateModifier);

/// returns the number of continuous state-variables used for the actual state
	unsigned int getNumActiveDiscreteStates();
/// returns the number of discrete state-variables used for the actual state
	unsigned int getNumActiveContinuousStates();

/// sets the number of continuous state-variables used for the actual state
	void setNumActiveDiscreteStates(int numActiveStates);
/// sets the number of discrete state-variables used for the actual state
	void setNumActiveContinuousStates(int numActiveStates);

/// Save as a binary
	void saveBinary(FILE *stream);
/// Load from a binary
	void loadBinary(FILE *stream);

/// Save as text
	void saveASCII(FILE *stream);
/// Load from a text
	void loadASCII(FILE *stream);

/// Clones the actual state
	virtual CState* clone();

/// returns the dim th continuous state
	virtual double getContinuousState(unsigned int dim);
/// returns the dim th normalized continuous state
/** The continuous state gets transformed in the intervall [0,1] according to
	its given min and max values.*/
	virtual double getNormalizedContinuousState(unsigned int dim);
/// returns the dim th  discrete state
	virtual int getDiscreteState(unsigned int dim);
	virtual int getDiscreteStateNumber();

/// sets the dim th continuous state
	virtual void setContinuousState(unsigned int dim, double val);
/// sets the dim th discrete state
	virtual void setDiscreteState(unsigned int dim, int val);

/// Compares the two states by comparing all state variables. As soon as one variable differs it returns false.
	virtual bool equals(CState *state);

	virtual double getDistance(ColumnVector *vector);

	double getSingleStateDifference(int dim, double value);

};

/// Class for logging a list of States
/** The class maintains a list of double and a list of integer vectors. For each state variable defined in its state properties there is an own
vector. So only states with the same properties as the state list can be added. When a state is added with addState(...) the state 
gets decomposed in its state variables, and each value of the state variable is added to its vector. So the CStateList class doesn't
save the State object itself, it only saves the state variables. This is done due to performance reasons because by saving the state object
a new CState object would have to be instantiated each time.
<p>
To Retrieve a state from the List you have to call the function getState(int num, CState *state). The values of the num th state are then filled
in the specified state object. */


class CStateList : virtual public CStateObject
{
protected:
	/// vectors for the continuous state variables
	std::vector<std::vector<double> *> *continuousStates;
	/// vectors for the discrete state variables
	std::vector<std::vector<int> *> *discreteStates;
	
	std::vector<bool> *resetStates;

	/// number of states saved
	int numStates;
public:
	/// Creates a state list
	/** The number of vectors for the discrete and continuous state variables are taken from the properties object 
	*/
	CStateList(CStateProperties *properties);
	virtual ~CStateList();

	/// add a state to the state list
	/**
	The state must have the same properties as the state list object. The state gets then decomposed in its state-variables and
	the state variables are stored.
	*/
	void addState(CState *state);
	/// Get a state from the state list.
	/** 
	The state must have the same properties as the state list object. The state variables of the num th state are retrieved 
	from the vectors and set in the state object. 
	@param num Number of the state to retrieve
	@param state The state object in which the retrieved state variables are filled in*/
	void getState(unsigned int num, CState *state);

	void removeLastState();
	void removeFirstState();

	/// load a statelist from a binary file
	/** The saved statelist must have had the same properties as the state list (at least same number of discrete and continuous states).
	*/
	virtual void loadBIN(FILE *stream);
	/// save the statelist to a binary file
	virtual void saveBIN(FILE *stream);

	/// load a statelist from a text file
	/** The saved statelist must have had the same properties as the state list (at least same number of discrete and continuous states).
	*/
	virtual void loadASCII(FILE *stream);
	/// save the statelist to a text file
	virtual void saveASCII(FILE *stream);
	
	/// get the number of states in the list
	unsigned int getNumStates();
	void clear();

	/// initialize the state list with continuous states comming from a d-dimensional grid
	void initWithContinuousStates(CState *initState, vector<int> dimensions, vector<double> partitions);

	/// initialize the state list with all possible combination of the states of the given state variables 
	void initWithDiscreteStates(CState *initState, vector<int> dimensions);
};



#endif


