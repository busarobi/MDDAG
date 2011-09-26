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

#ifndef C_ABSTRACTVFUNCTION_H
#define C_ABSTRACTVFUNCTION_H

#include <stdio.h>

#include "clearndataobject.h"
#include "cmyexception.h"
#include "cbaseobjects.h"
#include "cgradientfunction.h"
#include "cfeaturefunction.h"

class CAbstractVETraces;
class CFeatureQFunction;
class CStochasticPolicy;
class CRewardFunction;
class CStateCollectionImpl;
class CFeatureCalculator;
class CAbstractStateDiscretizer;
class CStateReward;

#define DIVERGENTVFUNCTIONVALUE 1000000

#define CONTINUOUSVFUNCTION 1
#define GRADIENTVFUNCTION 2
/// Interface reprenting a Value Function
/**Value functions return for each state the total discount reward which they expect to get from that state following a specific policy. Usually this is a Greedy Policy, greedy on the value function. 
In the RIL toolbox the Q-Functions are composed of v-functions, one v-function for each action, so the Value Function is the essential part part of each learning algorithm. 
The kind of the value function is for the most learning algorithm independent, so it does'nt matter what value functions you use for the Q-Function, the algorthm just works with the inteface of its v-function. 
\par
The class CAbstractVFunction is the interface for all value functions. The interface provides a gamma-value for each value function, serving as discount factor. 
The value functions have to implement functions for getting V-Values , setting V-Values and updating V-Values for an specific state. These three functions are:
<ul>
<li> getValue(CState *). The function returns the expected total discount reward for the given state. </li>
<li> setState(CState *, double value). This function is usually used for initialisation of th value function. It has to set the total discount reward of the state to the specified value as good as it is possible (for function  approximators). </li>
<li> updateValue(CState *, double td) is the function usually used for learning. Adds the td value to the current value of the function. </li>
</ul>
All these function have an companion piece with state collections, which are used by the learning algorithm. So each value function maintains an own state properties object to retrieve the required state from the given state collection and call the demanded function with a state as parameter. 
In the RIL toolbox there are 3 different kinds of value functions, 
V-Tables, V-FeatureFunctions and V-Functions using Neural Networks (The Torch toolbox is used for the neural networks). All these
value functions support the possibilty to save and load the learned values.
The class is subclass of CStateObject, with the consequent state properties object the desired state is fetches from the
state collection.
<p>
The class also has a function getStandardETraces to determine which E-Traces should be used. The function has to return a
new instantiated CAbstractVETraces object for the V-Function, which is used to compose the CQETtraces object. The function returns
CStateVETraces as standard.
*/
class CAbstractVFunction : public CStateObject, virtual public CLearnDataObject {
protected:
	int type;

	void addType(int newType);
	   
public:
	bool mayDiverge;

/// constructor, the properties are needed to fetch the state from the state collection.
	CAbstractVFunction(CStateProperties *properties);

	virtual ~CAbstractVFunction();

	virtual void resetData() {};

/// Calls updateValue(CState *state, double td) with the state assigned to the value function
	virtual void updateValue(CStateCollection *state, double td);
/// Calls setValue(CState *state, double qValue) with the state assigned to the value function
	virtual void setValue(CStateCollection *state, double qValue); 
/// Calls setValue(CState *state, double qValue) with the state assigned to the value function
	virtual double getValue(CStateCollection *state);
	
/// sets the value of the state to the current value + td
	virtual void updateValue(CState *state, double td);
/// sets the value of the state, has to be implemented by the other V-Functions
	virtual void setValue(CState *, double ) {}; 
/// returns the value of the state, has to be implemented by the other V-Functions
	virtual double getValue(CState *state) = 0;
	
/// Saves the Paramters of the Value Function
	virtual void saveData(FILE *file);
/// Loads the Paramters of the Value Function
	virtual void loadData(FILE *file);
/// Prints the Paramters of the Value Function
	virtual void printValues (){};

	int getType();
	bool isType(int isT);

/// Returns a standard VETraces object
	/**
The function has to return a new instantiated CAbstractVETraces object, which is used to compose the CQETtraces object. The function returns
CStateVETraces as standard.
	*/
	virtual CAbstractVETraces *getStandardETraces();
};

/// Value Function always returning zero
class CZeroVFunction : public CAbstractVFunction
{
protected:
public:
	CZeroVFunction();

	virtual double getValue(CState *state);
};

class CVFunctionSum : public CAbstractVFunction
{
protected:
	std::map<CAbstractVFunction *, double> *vFunctions;
public:
	CVFunctionSum();
	virtual ~CVFunctionSum();


	/// Interface for getting a Q-Value
	virtual double getValue(CStateCollection *state);
	virtual double getValue(CState *state) {return getValue((CStateCollection *) state);};


	virtual CAbstractVETraces *getStandardETraces() {return NULL;};

	double getVFunctionFactor(CAbstractVFunction *vFunction);
	void setVFunctionFactor(CAbstractVFunction *vFunction, double factor);

	void addVFunction(CAbstractVFunction *vFunction, double factor = 1.0);
	void removeVFunction(CAbstractVFunction *vFunction);

	void normFactors(double factor);
};

/// This exception is thrown if a value function has become divergent
/** 
There can be many reasons why a value function can become divergent, for example the learning rate is too high. The exception is thrown if the absolute value of a state gets higher than 100000. If your value function is doublely that high, please scale your reward function. 
*/
class CDivergentVFunctionException : public CMyException
{
protected:
	virtual string getInnerErrorMsg();
public:
	string vFunctionName;
	CAbstractVFunction *vFunction;
	CState *state;
	double value;

	CDivergentVFunctionException(string vFunctionName, CAbstractVFunction *vFunction, CState *state, double value);
	virtual ~CDivergentVFunctionException(){};
};

///Interface for all classes that can use gradients for updating
/** 
Gradient V-Functions are able to calculate the gradient of the V-Function with respect to the weights in the current state and can be also updated by a gradient object (represented as a CFeatureList object). In the toolbox all gradients are represented as feature lists, where the feature index is the weight index and the feature factor represents the gradient value of that weights. All weights that are not listed in the feature list have a zero gradient.
\par
For the gradient calculation all subclasses have to implement the function getGradient(CStateCollection *state, CFeatureList *gradientFeatures), where the gradient in the current state is calculated and written in the given feature list. The feature list is supposed to be empty.
\par
All gradient-VFunctions implement the interface CGradientUpdateFunction as the interface for updating a gradient function, so additionally the subclasses have to implement the functions:
- updateWeights(CFeatureList *gradient): Update the weights according to the gradient.
- getWeights(double *parameters), write all weights in the double array
- setWeights(double *parameters), set the weights according to the double array
- resetData(): reset all weights, needed when a new learning process is started
- getNumWeights(): return the number of weights.
As the V-Functions implement the gradient update interface, they can use varaible learning rates for different weights (see CAdaptiveEtaCalculator).
*/
class CGradientVFunction : public CAbstractVFunction, virtual public CGradientUpdateFunction
{
protected:

public:
	/// constructor, the properties are needed to fetch the state from the state collection.
	CGradientVFunction(CStateProperties *properties);
	virtual ~CGradientVFunction();

	/// Calls updateValue(CState *state, double td) with the state assigned to the value function
	virtual void updateValue(CStateCollection *state, double td);
	/// sets the value of the state to the current value + td
	virtual void updateValue(CState *state, double td);

	virtual void getGradient(CStateCollection *state, CFeatureList *gradientFeatures) = 0;

	virtual void resetData() = 0;
	virtual void loadData(FILE *stream) {CGradientUpdateFunction::loadData(stream);};
	virtual void saveData(FILE *stream) {CGradientUpdateFunction::saveData(stream);};

	virtual CAbstractVETraces *getStandardETraces();


	virtual void copy(CLearnDataObject *vFunction) {CGradientUpdateFunction::copy(vFunction);};	
};

/*
class CGradientDelayedUpdateVFunction : public CGradientVFunction, public CGradientDelayedUpdateFunction
{
protected:
	virtual void updateWeights(CFeatureList *dParams) {CGradientDelayedUpdateFunction::updateWeights(dParams);};

	CGradientVFunction *vFunction;
public:
	/// constructor, the properties are needed to fetch the state from the state collection.
	CGradientDelayedUpdateVFunction(CGradientVFunction *vFunction);
	virtual ~CGradientDelayedUpdateVFunction() {};

	virtual double getValue(CState *state);
	virtual void getGradient(CStateCollection *state, CFeatureList *gradientFeatures);

	virtual void resetData() {CGradientDelayedUpdateFunction::resetData();};

	///  Returns the number of weights.
	virtual int getNumWeights(){return CGradientDelayedUpdateFunction::getNumWeights();};

	virtual void getWeights(double *parameters) {CGradientDelayedUpdateFunction::getWeights(parameters);};
	virtual void setWeights(double *parameters) {CGradientDelayedUpdateFunction::setWeights(parameters);};

	virtual void loadData(FILE *stream) {CGradientVFunction::loadData(stream);};
	virtual void saveData(FILE *stream) {CGradientVFunction::saveData(stream);};

};
*/
/// Interface class for calculating the gradient dV(x)/dx
/** 
Interface for calcualting the input derivation of a feature function. The input derivation is calculated in the function getInputDerivation and written in the given targetVector, which always has the dimension of the model state (only for continuous state variables).
\par
By now there is only the numeric input derivation calculator, calculating the derivation analytically is supported by feature v-functions and torch-vfunctions but its not tested, so its recommended to use the numeric derivation.
*/
class CVFunctionInputDerivationCalculator : virtual public CParameterObject
{
protected:
	CStateProperties *modelState;
public:
	CVFunctionInputDerivationCalculator(CStateProperties *modelState);

	virtual void getInputDerivation( CStateCollection *state, ColumnVector *targetVector) = 0;
	unsigned int getNumInputs();
};


/// Calculating the input derivation of a V-Function numerically
/** 
The derivation is calculated by the three point rule for each input state variable, so the formular
$f'(x) = (f(x + stepSize) - f(x - stepSize))/ 2 * stepSize$ is used, stepSize is set in the constructor and also can be set by the Parameter "NumericInputDerivationStepSize". For each input state variable the stepsize is scaled with the size of the intervall of the state variable, so the "NumericInputDerivationStepSize" parameter is given in percent, and not an absolute value.
\par
The class "CVFunctionNumericInputDerivationCalculator" has the following Parameters:
- "NumericInputDerivationStepSize": stepSize of the numeric differentation.
*/
class CVFunctionNumericInputDerivationCalculator : public CVFunctionInputDerivationCalculator
{
protected:
	CAbstractVFunction *vFunction;
	CStateCollectionImpl *stateBuffer;
public:
	CVFunctionNumericInputDerivationCalculator(CStateProperties *modelState, CAbstractVFunction *vFunction, double stepSize, std::list<CStateModifier *> *modifiers);
	virtual ~CVFunctionNumericInputDerivationCalculator();

	virtual void getInputDerivation( CStateCollection *state, ColumnVector *targetVector);
};



/**
Feature Functions can be used as linear approximators, like tile-coding and RBF-networks, the exact usage of a feature function depends on the feature state it uses.
Feature states are a very common possibility to discretize continuous states. A feature consists of its index and a feature activation factor. The sum of all these factors should sum up to 1.
<p> 
Feature value function are modeled by the class CFeatureVFunction. For each feature it stores a feature value, so its just a table of features, 
the only difference to the tabular case is the calculation of the values (the sum of feature value * feature factor). CFeatureVFunctions are supposed to
get a feature calculator or discretizer as state properties object. With this state property object the value function is able retrieve its feature state from the state collections. 
CFeatureVFunction inherits all access functions for the features from CFeatureFunction. Additionaly it implements the 
functions for setting and getting the values with states. These functions decompose the feature state into its discrete states and call
the companion pieces of the functions for features (e.g. integer values instead of states) and multiply the values by the feature factors.
<p>
To create a feature value function you have to pass a state properties object to the constructor. The number of features for the function is calculated by the discrete state size of the 
properties object. Of course the properties object is passed to the the super calss CAbstractVFunction and is used to retrieve the 
wanted state from the state collection. 
You also have an additional contructor at yours disposal, which can be used to calculate the value function
from a stochastic policy given a Q-Function. The Values of the features are calculated as the expectation of the action propabilities multiplied with the
Q-Values. 
<p>
The standard VETraces object of feature value functions are CVFeatureETraces, which store the features in a feature list. 

*/
class CFeatureVFunction : public CGradientVFunction, public CFeatureFunction
{
protected:
	

public:
	CFeatureVFunction(int numFeatures);

	/// Creates a feature v-function with the specific feature state.
/**The number of features for the function is calculated by the discrete state size of the 
properties object. Of course the properties object is passed to the the super calss CAbstractVFunction and is used to retrieve the 
wanted state from the state collection. The properties are supposed to be from a feature or a discrete state.*/

	CFeatureVFunction(CStateProperties *featureFact);

/**Can be used to calculate the value function
from a stochastic policy given a Q-Function. The Values of the features are calculated as the expectation of the action propabilities multiplied with the
Q-Values. The constrcutor calls setVFunctionFromQFunction to do this. The state properties and so the number of features are taken from the Q-Function.
*/
	CFeatureVFunction(CFeatureQFunction *qfunction, CStochasticPolicy *policy);
	
	~CFeatureVFunction();
/**Can be used to calculate the value function
from a stochastic policy given a Q-Function. The Values of the features are calculated as the expectation of the action propabilities multiplied with the
Q-Values*/
	virtual void setVFunctionFromQFunction(CFeatureQFunction *qfunction, CStochasticPolicy *policy);

	virtual void updateWeights(CFeatureList *gradientFeatures);


/// Updates the value function given a feature or discrete state
/** Decomposes the feature state in its discrete state variable and adds the "td" value to the values of the features.
Each update is multiplied with the coresponding feature factor.
*/
	virtual void updateValue(CState *state, double td);
/// Sets the value given a feature or discrete state
/** Decomposes the feature state in its discrete state variable and sets the values of the features to the "qValue" value.
Each value is multiplied with the coresponding feature factor.
*/
	virtual void setValue(CState *state, double qValue);
/// Returns the value given a feature or discrete state
/** Decomposes the feature state in its discrete state variable and calculates the value by summing up the feature values
of the aktiv features multiplied with their factor.
*/
	virtual double getValue(CState *state);

/// Calls the saveFeatures function of CFeatureFunction
	virtual void saveData(FILE *file);
/// Calls the loadFeatures function of CFeatureFunction
	virtual void loadData(FILE *file);
	virtual void printValues();

/// Returns a new CFeatureVETraces object
	virtual CAbstractVETraces *getStandardETraces();

	virtual void getGradient(CStateCollection *state, CFeatureList *gradientFeatures);


	virtual int getNumWeights();

	virtual void resetData();


	virtual void getWeights(double *parameters);
	virtual void setWeights(double *parameters);

	void setFeatureCalculator(CFeatureCalculator *featCalc);

};

/*
class CFeatureVFunctionInputDerivationCalculator : public CVFunctionInputDerivationCalculator
{
protected:
	CFeatureVFunction *vFunction;
	ColumnVector *featureInputDerivation;
	ColumnVector *fiDerivationSum;

public:
	bool normalizedFeatures; 

	CFeatureVFunctionInputDerivationCalculator(CStateProperties *inputState, CFeatureVFunction *vFunction);
	~CFeatureVFunctionInputDerivationCalculator();

	virtual void getInputDerivation( CStateCollection *state, ColumnVector *targetVector);
};*/


/// Value Function as a table
/*
Tables are just the same as feature functions. The only difference is the kind of states for the value-table, it uses discrete states. The class CVTable represents tabular value functions, 
the class is subclass of CFeatureVFunction. Value-Tables can only be used with CAbstractStateDiscretizer.
*/
class CVTable : public CFeatureVFunction
{
public:
	CVTable(CAbstractStateDiscretizer *state);
	
	~CVTable();
		
	void setDiscretizer(CAbstractStateDiscretizer *discretizer);
	CAbstractStateDiscretizer *getDiscretizer();

	int getNumStates();
};

class CRewardAsVFunction : public CAbstractVFunction
{
protected:
	CStateReward *reward;
public:
	CRewardAsVFunction(CStateReward *reward);
	virtual ~CRewardAsVFunction() {};

	virtual double getValue(CState *state);
};


#endif

