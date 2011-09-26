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

#ifndef C_STATEMODIFIER_H
#define C_STATEMODIFIER_H

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include <list>
#include "cstateproperties.h"
#include <newmat/newmat.h>

class CStateCollection;
class CStateCollectionImpl;
class CState;

class CDataSet;
class CDataSet1D;
class CFeatureVFunction;
/// Abstract class for calculating modified states from the original Model state
/**State modifier take the original model state or any other modified state and transform the state in a new state. The new state can be
any type of state. Feature states are created by all sub classes of CFeatureCalculator and discrete states are calculated by all subclasses of CAbstractStateDiscretizer.
The only function of CStateModifier is getModifiedState(CStateCollection *originalState, CState *targetState). All a state modifier has to do is to take 
the information of the actual state from the "originalState" collection and writes the new, modified state into the target state.
To use an instantiated state modifier you have to add the modifier to the state modifier list of the agent. The agent again 
adds the modifier to its state collections, so the modified state gets available to all listeners.
\par
CStateModifier is also subclass of CStateProperties. This has basically 2 reasons. First of all this is an easy way to specify the
properties of the modified states calculated by this modifier, and the other important reason is that the modifier itself serves as 
the id for a state in a state collection.
*/
class CStateModifier : public CStateProperties
{
protected:
	bool changeState;

	CStateModifier();

	std::list<CStateCollectionImpl *> *stateCollections;

	virtual void registerStateCollection(CStateCollectionImpl *stateCollection);
	virtual void removeStateCollection(CStateCollectionImpl *stateCollection);

	virtual void stateChanged();


	friend class CStateCollectionImpl;
public:
/// constructor for state modifiers, sets the number of continuous and discrete states of the resulting modified state objects.
	CStateModifier(unsigned int numContinuousStates, unsigned int numDiscreteStates);
	virtual ~CStateModifier();

/// Virtual function for calculating the modified state from the original state (usually the model state)
	virtual void getModifiedState(CStateCollection *originalState, CState *modifiedState) = 0;

	
};

/// State Modifier used for Neural Networks input states
/** 
This state modifier does the input data preprocessing for neural networks. For Non-Periodic continuous state variables, the current state value gets scaled into the intervall [-1,1], periodic state variables get a special treatment. In order to model the periodicity more accurate, the periodic state variable gets splitted into two state variables, one representing the sinus and one the cosinus of the normalized periodic state. So a neural network state has the number of periodic state variables more state variables than the originalstate. The discrete state variables remain the same. The original state can be set in the constructor and is usually the modelstate.
*/
class CNeuralNetworkStateModifier : public CStateModifier
{
protected:
	CStateProperties *originalState;
	unsigned int *dimensions;
	unsigned int numDim;

	ColumnVector *input_mean;
	ColumnVector *input_std;

	ColumnVector *buffVector;

	virtual void preprocessInput(ColumnVector *input, ColumnVector *norm_input);
	
	/// Returns the number of continuous states for the NN-state
	static int getNumInitContinuousStates(CStateProperties *properties, unsigned int *dimensions,unsigned int numDim);
public:
	bool normValues;

	CNeuralNetworkStateModifier(CStateProperties *originalState);
	CNeuralNetworkStateModifier(CStateProperties *originalState, unsigned int *dimensions, unsigned int numDim);
	~CNeuralNetworkStateModifier();

	///Data preprocessing for NN's
	/** 
	For Non-Periodic continuous state variables, the current state value gets scaled into the intervall [-1,1], periodic state variables get a special treatment. In order to model the periodicity more accurate, the periodic state variable gets splitted into two state variables, one representing the sinus and one the cosinus of the normalized periodic state. So a neural network state has the number of periodic state variables more state variables than the originalstate. The discrete state variables remain the same. The original state can be set in the constructor and is usually the modelstate.
	*/
	virtual void getModifiedState(CStateCollection *originalState, CState *modifiedState);

	void setPreprocessing(ColumnVector *input_mean, ColumnVector *input_std);
};

/// Base class for all Feature Calculators
/** 
Features always have a unique feature index and a feature activation factor, they are used for linear approximators. Feature calculators determine the feature activation factors of a feature state. A feature state consists of N discrete and continuous state variables, each pair represents a feature. All features which are not listed in a feature state have an activation factor of 0.0, so N is the maximum number of active features in a feature state. 
\par
The CFeatureCalculator class is the interface of all other feature calculators, it sets the state properties (numActiveFeatures continuous + discrete states, min = 0, max = 1.0 of all continuous states, and numFeatures is the discrete state size of all discrete state variables)
\par
It also provides functions for setting the original state. The original state is used by all feature calculators as the "working" state. This is per default the model state, but can be set differently if needed. 
\par 
The feature calculation itself is implemented by the subclasses in the function getModifiedState.
*/
class CFeatureCalculator : public CStateModifier
{
protected:
/// Normalizes all active features of the given feature state
	/** 
	Normalizes the feature factors so that the sum of all factors equals 1.0.
	*/
	void normalizeFeatures(CState *featState);

	unsigned int numFeatures;
	unsigned int numActiveFeatures;

	CFeatureCalculator();

	/// sets the state properties
	void initFeatureCalculator(unsigned int numFeatures, unsigned int numActiveFeatures);

	CStateProperties *originalState;
public:
	CFeatureCalculator(unsigned int numFeatures, unsigned int numActiveFeatures);
	virtual ~CFeatureCalculator() {};

	virtual unsigned int getNumFeatures();
	virtual unsigned int getNumActiveFeatures();

	virtual unsigned int getDiscreteStateSize(unsigned int i);
	virtual unsigned int getDiscreteStateSize();

	virtual double getMin(unsigned int i);
	virtual double getMax(unsigned int i);

		/// Sets the original state
	/** 
	The original state is used by all feature calculators as the "working" state. This is per default the model state, but can be set differently if needed. 
	*/
	CStateProperties *getOriginalState() {return originalState;};
	void setOriginalState(CStateProperties *originalState) {this->originalState = originalState;};
};

class CNewFeatureCalculator
{
public:
	virtual ~CNewFeatureCalculator() {};

	virtual CFeatureCalculator * getFeatureCalculator(CFeatureVFunction *vFunction, CDataSet *inputData, CDataSet1D *outputData) = 0;
};

/// interface for all state modifier how have acces to several other state modifier for state calculation.
/** 
Base class for feature operators (CFeatureOperatorOr, CFeatureOperatorAnd) and discrete state operators (CDiscreteStateOperatorAnd).
*/
class CStateMultiModifier
{
protected:
	std::list<CState *> *states;
	std::list<CStateModifier *> *modifiers;
public:
	CStateMultiModifier();
	virtual ~CStateMultiModifier();

	virtual void addStateModifier(CStateModifier *featCalc);

	std::list<CStateModifier *> *getStateModifiers();
};

class CStateVariablesChooser : public CStateModifier
{
protected:
	unsigned int *contStatesInd;
	unsigned int *discStatesInd;

	CStateProperties *originalState;
public:
	CStateVariablesChooser(unsigned int numContStates, unsigned int *contStatesInd, unsigned int numDiscStates, unsigned int *discStatesInd, CStateProperties *originalState = NULL);

	~CStateVariablesChooser();


	virtual void getModifiedState(CStateCollection *originalState, CState *modifiedState);
};

#endif

