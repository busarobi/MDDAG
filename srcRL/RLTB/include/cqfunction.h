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

#ifndef CRIABSTRACTQFUNCTION_H
#define CRIABSTRACTQFUNCTION_H

#include <stdio.h>

#include "clearndataobject.h"
#include "cbaseobjects.h"
#include "cmyexception.h"
#include "cgradientfunction.h"

class CAbstractFeatureStochasticModel;
class CAbstractQETraces;
class CGradientQETraces;

class CAbstractVFunction;
class CFeatureVFunction;
class CFeatureRewardFunction;
class CActionStatistics;
class CFeature;

#define GRADIENTQFUNCTION 1
#define CONTINUOUSACTIONQFUNCTION 2


/// Interface for all Q-Functions
/**Q-Functions depend on the state and the action you can choose in the current state, so the policy is able to decide which action is best (usually 
the action with the highest Q-Value). CAbstractQFunction is the base class of all Q-Functions. It just provides the interface for getting, setting and updating (adding a value to the current value) Q-Values. These functions are again
getValue, setValue and updateValue, now there is always an action object as additional parameter. The class maintains an action set of all
actions which are stored in the Q-Function. In addition the class provides one function to
retrieve the action with the best Q-Value (getMax), one function for retrieving the value of these best action (getMaxValue) and there
is also a function which writes all Q-Values of a state (i.e. for all actions) in a double array (getActionValues). 
\par
The class also maintains a gamma factor which should be the same as the gamma factor of its value functions.
In addition the interface provides functions for storing and loading the Values of the QFunction (storeValues and loadValues).
@see CQFunction
*/


class CAbstractQFunction : public CActionObject, virtual public CLearnDataObject
{
protected:
	int type;
public:
	bool mayDiverge;

	int getType();
	bool isType(int type);
	void addType(int Type);


/// Creates a QFunction, handling Q-Values for all actions in the actionset.
	CAbstractQFunction(CActionSet *actions);
	virtual ~CAbstractQFunction();

	virtual void saveData(FILE *file);
	virtual void loadData(FILE *file);
	virtual void printValues (){};

	virtual void resetData() {};

/// Writes the Q-Values of the specified actions in the actionValues array.
/** so the size of the array has to be at least the size of the action set.*/
	void getActionValues(CStateCollection *state, CActionSet *actions, double *actionValues, CActionDataSet *data = NULL);

/// Calculates the best action from a given action set.
/** Returns the best action from the availableActions action set. If several actions have the same 
best Q-Value, the first action which has this value in the action set is choosen.*/
	virtual CAction* getMax(CStateCollection *state, CActionSet *availableActions, CActionDataSet *data = NULL);
/// Returns the best action value from a given action set.
/** Returns the best action value from the availableActions action set.*/
	virtual double getMaxValue(CStateCollection *state, CActionSet *availableActions);
/// Returns the statistics for a given action.
    virtual void getStatistics(CStateCollection *state, CAction *action, CActionSet *actions, CActionStatistics* statistics);
	
/// Interface for updating a Q-Value
	virtual void updateValue(CStateCollection *, CAction *, double , CActionData * = NULL) {};
/// Interface for setting a Q-Value
	virtual void setValue(CStateCollection *state, CAction *action, double qValue, CActionData *data = NULL); 
/// Interface for getting a Q-Value
	virtual double getValue(CStateCollection *state, CAction *action, CActionData *data = NULL) = 0;

	virtual CAbstractQETraces *getStandardETraces() {return NULL;};

protected:
};


class CQFunctionSum : public CAbstractQFunction
{
protected:
	std::map<CAbstractQFunction *, double> *qFunctions;
public:
	CQFunctionSum(CActionSet *actions);
	virtual ~CQFunctionSum();


	/// Interface for getting a Q-Value
	virtual double getValue(CStateCollection *state, CAction *action, CActionData *data = NULL);

	virtual CAbstractQETraces *getStandardETraces() {return NULL;};

	double getQFunctionFactor(CAbstractQFunction *qFunction);
	void setQFunctionFactor(CAbstractQFunction *qFunction, double factor);

	void addQFunction(CAbstractQFunction *qFunction, double factor);
	void removeQFunction(CAbstractQFunction *qFunction);


	void normFactors(double factor);

};

/// This exception is thrown if a value function has become divergent
/** 
There can be many reasons why a value function can become divergent, for example the learning rate is too high.
*/
class CDivergentQFunctionException : public CMyException
{
protected:
	virtual string getInnerErrorMsg();
public:
	string qFunctionName;
	CAbstractQFunction *qFunction;
	CState *state;
	double value;

	CDivergentQFunctionException(string qFunctionName, CAbstractQFunction *qFunction, CState *state, double value);
	virtual ~CDivergentQFunctionException(){};
};

class CGradientQFunction : public CAbstractQFunction, virtual public CGradientUpdateFunction
{
protected:
	CFeatureList *localGradientQFunctionFeatures;

public:
	CGradientQFunction(CActionSet *actions);
	virtual ~CGradientQFunction();

	virtual int getWeightsOffset(CAction *) {return 0;};

	virtual void getGradient(CStateCollection *state, CAction *action, CActionData *data, CFeatureList *gradient) = 0;

	/// Interface for updating a Q-Value
	virtual void updateValue(CStateCollection *state, CAction *action, double td, CActionData *data = NULL);
	
	virtual void resetData() {CAbstractQFunction::resetData();};
	virtual void loadData(FILE *stream) {CGradientUpdateFunction::loadData(stream);};
	virtual void saveData(FILE *stream) {CGradientUpdateFunction::saveData(stream);};

	virtual CAbstractQETraces *getStandardETraces();

	virtual void copy(CLearnDataObject *qFunction) {CGradientUpdateFunction::copy(qFunction);};
};

/*
class CGradientDelayedUpdateQFunction : public CGradientQFunction, public CGradientDelayedUpdateFunction
{
protected:
	virtual void updateWeights(CFeatureList *dParams) {CGradientDelayedUpdateFunction::updateWeights(dParams);};

	CGradientQFunction *qFunction;
public:
	/// constructor, the properties are needed to fetch the state from the state collection.
	CGradientDelayedUpdateQFunction(CGradientQFunction *qFunction);
	virtual ~CGradientDelayedUpdateQFunction() {};

	virtual double getValue(CStateCollection *state, CAction *action, CActionData *data = NULL);
	virtual void getGradient(CStateCollection *state, CAction *action, CActionData *data, CFeatureList *gradientFeatures);

	virtual void resetData() {CGradientDelayedUpdateFunction::resetData();};

	///  Returns the number of weights.
	virtual int getNumWeights(){return CGradientDelayedUpdateFunction::getNumWeights();};

	virtual void getWeights(double *parameters) {CGradientDelayedUpdateFunction::getWeights(parameters);};
	virtual void setWeights(double *parameters) {CGradientDelayedUpdateFunction::setWeights(parameters);};

	virtual void loadData(FILE *stream) {CGradientQFunction::loadData(stream);};
	virtual void saveData(FILE *stream) {CGradientQFunction::saveData(stream);};

};
*/
/// Compounded Q-Function consisting of V-Funcions
/**
For Q-Functions there is one value function for each action of the Q-Function, so its obvious to compose Q-Functions of
value Functions. RIL toolbox gives you the possibility to do this. This has the advantage that you can choose an own value functions for each action, so its possible to create own discretizations or 
even other kinds of value functions for an action. This possibility is only available for certain algorithm, e.g. the model based prioritized sweeping 
algorithm expects an Q-Function consisting of feature value functions with the same feature calculator (at least with the same number of features).
The composition of value function is modeled by the class CQFunction. The class maintains a list of value functions, which has the same size as the action set of the Q-Function.
You can assign specific value functions to specific actions. This is done by setVFunction. 
\par
If one of the functions for accessing the Q-Values is called (getValue, setValue, updateValue) the composed Q-Functions refers
the call to the value function of the specified action. There are 2 subclasses of CQFunction, one for feature Q-Functions and one for Q-Tables.
*/
class CQFunction : public CGradientQFunction
{
protected:
/// The list of V-Functions
/**Has the same order as the actionset, so the first qFunction coresponds to the first action.
*/
	std::map<CAction *, CAbstractVFunction *> *vFunctions;

	virtual int getWeightsOffset(CAction *action);
   
	virtual void updateWeights(CFeatureList *features);

public:
/// Creates a composed Q-Function for the given actions
/**
The Value-Functions list is initialized with NULL with the same size of the action set, so the V-Functions
has to be set by the user with the function setVFunction.*/
	CQFunction(CActionSet *actions);
	virtual ~CQFunction();

/// Updates the Value of the value function assigned to the given action
/** Calls the updateValue Function of the specified value function.
*/
	virtual void updateValue(CStateCollection *state, CAction *action, double td, CActionData *data = NULL);
/// Sets the Value of the value function assigned to the given action
/** Calls the setValue Function of the specified value function.
*/
	virtual void setValue(CStateCollection *state, CAction *action, double qValue, CActionData *data = NULL); 
/// Returns the Value of the value function assigned to the given action
/** Returns the value of  the getValue Function of the specified value function.
*/
	virtual double getValue(CStateCollection *state, CAction *action, CActionData *data = NULL);

/// Updates the Value of the value function assigned to the given action
/** Calls the updateValue Function of the specified value function. The given state must be the state
used by the value function (at least a state that can be used by the value function)
*/
	virtual void updateValue(CState *state, CAction *action, double td, CActionData *data = NULL);
/// Sets the Value of the value function assigned to the given action
/** Calls the setValue Function of the specified value function. The given state must be the state
used by the value function (at least a state that can be used by the value function.
*/	
	virtual void setValue(CState *state, CAction *action, double qValue, CActionData *data = NULL); 
/// Returns the Value of the value function assigned to the given action
/** Returns the value of  the getValue Function of the specified value function. The given state must be the state
used by the value function (at least a state that can be used by the value function.
*/
	virtual double getValue(CState *state, CAction *action, CActionData *data = NULL);

/// Saves the Value Functoins
/**
Calls the saveValues method of all Value Functions
*/
	virtual void saveData(FILE *file);
/// Loads the Value Functions
/** Calls the loadValues method of all Value Functions. So the value Function list has already to be initialized
and the V-Functions has to be in the same order as they were when the Q-Function was save.*/
	virtual void loadData(FILE *file);
/// Calls saveValues with stdout as outputstream
	virtual void printValues();

/// Returns the value function assigned to the given action
	CAbstractVFunction *getVFunction(CAction *action);
/// Returns the indexth value function (so the value function assigned to the indexth action).
	CAbstractVFunction *getVFunction(int index);
/// Sets the Value-Function of the specified action.
/**
If bDeleteOld is true (default) the old Value Function is deleted
*/
	void setVFunction(CAction *action, CAbstractVFunction *vfunction, bool bDeleteOld = true);
/// Sets the Value-Function of the indexth action.
/**
If bDeleteOld is true (default) the old Value Function is deleted
*/
	void setVFunction(int index, CAbstractVFunction *vfunction, bool bDeleteOld = true);
/// Returns the number of V-Functions, which is always the number of Actions
	int getNumVFunctions();

	virtual CAbstractQETraces *getStandardETraces();

	//virtual CStateProperties *getGradientCalculator(CAction *action);
	virtual void getGradient(CStateCollection *state, CAction *action, CActionData *data, CFeatureList *gradient);

	virtual int getNumWeights();

	virtual void getWeights(double *weights);
	virtual void setWeights(double *weights);

	virtual void resetData();
	virtual void copy(CLearnDataObject *qFunction);
};

/// Converts a VFunction and a Model to a Q-Function
/**The class calculates the Q-Value by combining the information from a model and a feature V-Function.
So the class obviuosly only provides the functions for getting a Q-Value. The Q-Value of an action is calculated
the following way: Q(s,a)=sum_{s'} P(s'|s,a)*(R(s,a,s') + gamma * V(s')).
<p>
This class is used for the policies if you only have a V-Function (e.g. model based learning), since policies can only handle 
Q-Functions.
*/

class CQFunctionFromStochasticModel :  public CAbstractQFunction, public CStateObject
{
protected:

/// The given V-Function
	CFeatureVFunction *vfunction;
/// The model
	CAbstractFeatureStochasticModel *model;
/// Discretizer used by the V-Function
	CStateProperties *discretizer;
/// feature Reward Function for the learning problem.
	CFeatureRewardFunction *rewardfunction;

/// state buffer
	CState *discState;

public:
/// Creates a new QFunction from VFunction object for the given V-Function and the given model, the discretizer is take nfrom the V-Function.
	CQFunctionFromStochasticModel(CFeatureVFunction *vfunction, CAbstractFeatureStochasticModel *model, CFeatureRewardFunction *rewardfunction);

	virtual ~CQFunctionFromStochasticModel();

// Writes the Action-Values in the actionValues Array.
//	void getActionValues(CStateCollection *state, double *actionValues, CActionSet *actions);

/// Does nothing
	virtual void updateValue(CStateCollection *, CAction *, double , CActionData * = NULL) {};
/// Does nothing
	virtual void setValue(CStateCollection *, CAction *, double , CActionData * = NULL) {}; 

/// getValue function for state collections
/** Calls the getValue function for the specific state (retrieved from the collection by the discretizer)*/
	virtual double getValue(CStateCollection *state, CAction *action, CActionData *data = NULL);

/// getValue functoin for states
/**
Decomposes the feature state in its discrete state variables and calls the getValue(int, CAction *) function. The results are
wheighted by the feature factors and summed up. For discrete states obviously just the getValue(int, CAction *) with the discrete state
number is called.
*/
	virtual double getValue(CState *featState, CAction *action, CActionData *data = NULL);
/// calculates the Action-Value for a specific discrete state number
/**
The Q-Value of an action is calculated the following way: Q(s,a)= sum_{s'} P(s'|s,a)*(R(s,a,s') + gamma * V(s')).
*/
	virtual double getValue(int feature, CAction *action, CActionData *data = NULL);

	virtual CAbstractQETraces *getStandardETraces() {return NULL;};
};



/// Composed feature Q-Function
/**
The class CFeatureQFunction is a composed Q-Function which consists of feature value function with the same feature calculator (or the same discretizer).
Very often this ia all you need for learning, and CFeatureQFunction objects are easier to create. Its also needed because some learning algorithm expect the value functions to be all of the same kind (prioritized sweeping) and 
<p>
The class also provides methods for manipulating the features directly, without accessing the value functions explicitly. 
<p>
For creation of an feature Q-Function object you only need a state properties object of a feature calculator or discretizer. 
<p>
In addition you have the possibility to initialise you Q-Function with the values from a V-Function combined with a theoretical model (see CQFunctionFromStochasticModel). The difference to CQFunctionFromStochasticModel
is, that all Action-Values for all states get calculated and stored in the value Function. With CQFunctionFromStochasticModel the value of the state gets calculated directly by the model and V-Function.
So you can convert a Value Function to a Q-Function.
*/

class CFeatureQFunction : public CQFunction
{
protected:
/// Discretizer used for retrieving the state from the state collection
	CStateProperties *discretizer;
/// number of features from the Value-Functions
	unsigned int features;

	std::list<CFeatureVFunction *> *featureVFunctions;

/// initializes the V-Function list with CFeatureVFunction objects
	virtual void init();

/// initializes the V-Functions with the Values calculated by a V-Function, a theoretical model and a reward function
/**
The action values are calculated for each action in each state by the function CDynmaicProgramming::getActionValue and then they aer stored
in the V-Functions.
*/
	void initVFunctions(CFeatureVFunction *vfunction, CAbstractFeatureStochasticModel *model,  CFeatureRewardFunction *rewardFunction, double gamma);

public:
/// Creates an Q-Function with the specified discretizer.
	CFeatureQFunction(CActionSet *actions, CStateProperties *discretizer);
/// initializes the Value Functions with the values comming from a V-Function combined with a model and a reward funcion
/**
The actionset is taken from the model, the discretizer from the feature V-Function. The V-Funcions are initialized by the Function
initVFunctions.
*/
	CFeatureQFunction(CFeatureVFunction *vfunction, CAbstractFeatureStochasticModel *model,  CFeatureRewardFunction *rewardFunction,double gamma);
	
	virtual ~CFeatureQFunction();
	
/// Calls updateValue from the specified V-Function
/**
Allows direct feature manipulation without the need of state objects.
*/
	void updateValue(CFeature *state, CAction *action, double td, CActionData *data = NULL);
/// Calls setValue from the specified V-Function
/**
Allows direct feature manipulation without the need of state objects.
*/
	void setValue(int state, CAction *action, double qValue, CActionData *data = NULL); 
/// Returns the Value of a feature for a specific actoin.
/**
Allows direct feature manipulation without the need of state objects.
*/
	double getValue(int feature, CAction *action, CActionData *data = NULL);

	void setFeatureCalculator(CStateModifier *discretizer);
	CStateProperties *getFeatureCalculator();


	int getNumFeatures();

/// Saves the Values of the actions for each state in a readable tabular form
/** Tool for debugging and tracing the learning results*/
	void saveFeatureActionValueTable(FILE *stream);
/// Saves the index of the best action for each state in a readable tabular form
/** Tool for debugging and tracing the learning results*/
	void saveFeatureActionTable(FILE *stream);
};

class CComposedQFunction : public  CGradientQFunction
{
protected:
	std::list<CAbstractQFunction *> *qFunctions;

	virtual int getWeightsOffset(CAction *action);
	virtual void updateWeights(CFeatureList *features);

public:
	CComposedQFunction();
	virtual ~CComposedQFunction();

	virtual void saveData(FILE *file);
	virtual void loadData(FILE *file);
	virtual void printValues();

	virtual void getStatistics(CStateCollection *state, CAction *action, CActionSet *actions, CActionStatistics* statistics);

	/// Interface for updating a Q-Value
	virtual void updateValue(CStateCollection *state, CAction *action, double td, CActionData *data = NULL);
	/// Interface for setting a Q-Value
	virtual void setValue(CStateCollection *state, CAction *action, double qValue, CActionData *data = NULL); 
	/// Interface for getting a Q-Value
	virtual double getValue(CStateCollection *state, CAction *action, CActionData *data = NULL);

	void addQFunction(CAbstractQFunction *qFunction);

	std::list<CAbstractQFunction *> *getQFunctions();
	int getNumQFunctions();

	virtual CAbstractQETraces *getStandardETraces();

	//virtual CStateProperties *getGradientCalculator(CAction *action);

	virtual void getGradient(CStateCollection *state, CAction *action, CActionData *data, CFeatureList *gradient);


	virtual int getNumWeights();
	virtual void getWeights(double *weights);
	virtual void setWeights(double *weights);

	virtual void resetData();
};

/*
class CQTable : public CFeatureQFunction
{
	CAbstractStateDiscretizer *discretizer;
	virtual void init(int states);

public:
	CQTable(CActionSet *actions, CAbstractStateDiscretizer *state);
	
	~CQTable();
	
	void setDiscretizer(CAbstractStateDiscretizer *discretizer);
	CAbstractStateDiscretizer *getDiscretizer();
	
	int getNumStates();
};*/
#endif



