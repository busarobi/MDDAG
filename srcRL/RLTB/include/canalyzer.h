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

#ifndef CANALYZER_H
#define CANALYZER_H

#define ANALYZER_MSE 1
#define ANALYZER_MAE 2
#define ANALYZER_MAXERROR 3

#include "cbatchlearning.h"
#include "ctestsuit.h"
#include "crewardfunction.h"
#include "cagentcontroller.h"
#include "cbaseobjects.h"

#include <list>

class CAbstractQFunction;
class CAbstractVFunction;

/// Analyzer for V-Functions
/**
With V-Function Analyzers you can create different tables for showing the shape of your continuous V-Function. These tables can only be saved to a file.
The analyzer supports:
- 1 dimensional Tables
- 2 dimensional Tables
- calculate the Value for each State in a given list

A 1-dimensional Table can be created with the function save1DValues. You can choose the dimension for which the table shall be created and the number of partitions for this dimension. Addtionally you can specify an init-state, this init-state is used for the state values for all other dimensions. This function can only be used for continuous state variables.
A 2-dimensional Table can be created with the function save2DValues. You can choose both dimension, and the number of partitions for each dimension. The initstate is used in the same way as for 1-D Tables. This function can only be used for continuous state variables.
The table for specific states is created saveStateValues, here you just have to specify the states in the state list object. This function can also be used for discrete states.
*/

class CVFunctionAnalyzer
{
protected:
	CAbstractVFunction *vFunction;
	CStateProperties *modelStateProperties;
	CStateCollectionImpl *stateCollection;
public:
/// Create a new V-Function Analyzer
	/**
	The analyzer needs the vFunction, the modelstate and all the modifiers the v-function needs. It is recommended to take the agent's state modifiers (agent->getStateModifiers();).
	*/
	CVFunctionAnalyzer(CAbstractVFunction *vFunction, CStateProperties *modelState, std::list<CStateModifier *> *modifiers);
	virtual ~CVFunctionAnalyzer();

	/// Create a 1 dimensional Table of the Shape of the V-Function
	/**A 1-dimensional Table can be created with the function save1DValues. You can choose the dimension for which the table shall be created and the number of partitions for this dimension. Addtionally you can specify an init-state, this init-state is used for the state values for all other dimensions. This function can only be used for continuous state variables.*/
	void save1DValues(FILE *stream, CState *initstate, int dim1, int part1);

	/// Same as save1DValues, saves the output values in a readable matlab format
	/** Format of a row:
	    s_i v_i \n
	    s_i ... Value of the continuous state variable
	    v_i ... V-Value of the state
	*/
	void save1DMatlab(FILE *stream, CState *initstate, int dim1, int part1);
	
	/// Create a 1 dimensional Table of the Shape of the V-Function
	/**A 1-dimensional Table can be created with the function saveDiscrete1DValues. You can choose the dimension for which the table shall be created. Addtionally you can specify an init-state, this init-state is used for the state values for all other dimensions. This function can only be used for discrete state variables.*/
	void saveDiscrete1DValues(FILE *stream, CState *initstate, int dim1);

	/// Same as saveDiscrete1DValues, saves the output values in a readable matlab format
	/** Format of a row:
	    s_i v_i \n
	    s_i ... Discrete State Index
	    v_i ... V-Value of the state
	*/
	void saveDiscrete1DMatlab(FILE *stream, CState *initstate, int dim1);
	
	
	/// Create a 2 dimensional Table of the Shape of the V-Function
	/**A 2-dimensional Table can be created with the function save2DValues. You can choose both dimension, and the number of partitions for each dimension. The initstate is used in the same way as for 1-D Tables. This function can only be used for continuous state variables.*/
	void save2DValues(FILE *stream, CState *initstate, int dim1, int part1, int dim2, int part2);

	/// Same as save2DValues, saves the output values in a readable matlab format
	/** Format of a row:
	    s_i s_j v_i \n
	    s_i, s_j ... Values of the first and second continuous state variables
	    v_i ... V-Value of the state
	*/
	void save2DMatlab(FILE *stream, CState *initstate, int dim1, int part1, int dim2, int part2);

	/// Create a 2 dimensional Table of the Shape of the V-Function
	/**A 2-dimensional Table can be created with the function saveDiscrete2DValues. You can choose the dimensions for which the table shall be created. The dimension row_dim will be used as the row index, col_dim is the column index in the 2D output. The initstate is used in the same way as for 1-D Tables. This function can only be used for discrete state variables.*/
	void saveDiscrete2DValues(FILE *stream, CState *initstate, int row_dim, int col_dim);

	/// Same as saveDiscrete2DValues, saves the output values in a readable matlab format
	/** Format of a row:
	    s_i s_j v_i \n
	    s_i, s_j ... First and second discrete state index
	    v_i ... V-Value of the state
	*/
	void saveDiscrete2DMatlab(FILE *stream, CState *initstate, int row_dim, int col_dim);


	/// Creates a table of the values of the specified states
	void saveStateValues(FILE *stream, CStateList *states);
	
	/// Create a 1 dimensional Table of the Shape of the V-Function
	/**A 1-dimensional Table can be created with the function save1DValues. You can choose the dimension for which the table shall be created and the number of partitions for this dimension. Addtionally you can specify an init-state, this init-state is used for the state values for all other dimensions. This function can only be used for continuous state variables.*/
	void save1DValues(char *filename, CState *initstate, int dim1, int part1);
	
	/// Create a 1 dimensional Table of the Shape of the V-Function
	/**A 1-dimensional Table can be created with the function saveDiscrete1DValues. You can choose the dimension for which the table shall be created. Addtionally you can specify an init-state, this init-state is used for the state values for all other dimensions. This function can only be used for discrete state variables.*/
	void saveDiscrete1DValues(char *filename, CState *initstate, int dim1);
	
	/// Create a 2 dimensional Table of the Shape of the V-Function
	/**A 2-dimensional Table can be created with the function save2DValues. You can choose both dimension, and the number of partitions for each dimension. The initstate is used in the same way as for 1-D Tables. This function can only be used for continuous state variables.*/
	void save2DValues(char *filename, CState *initstate, int dim1, int part1, int dim2, int part2);

	/// Create a 2 dimensional Table of the Shape of the V-Function
	/**A 2-dimensional Table can be created with the function saveDiscrete2DValues. You can choose the dimensions for which the table shall be created. The dimension row_dim will be used as the row index, col_dim is the column index in the 2D output. The initstate is used in the same way as for 1-D Tables. This function can only be used for discrete state variables.*/
	void saveDiscrete2DValues(char *filename, CState *initstate, int row_dim, int col_dim);


	/// Creates a table of the values of the specified states
	void saveStateValues(char *filename, CStateList *states);

	/// Same as save1DValues, saves the output values in a readable matlab format
	/** Format of a row:
	    s_i v_i \n
	    s_i ... Value of the continuous state variable
	    v_i ... V-Value of the state
	*/
	void save1DMatlab(char *filename, CState *initstate, int dim1, int part1);
	
	/// Same as saveDiscrete1DValues, saves the output values in a readable matlab format
	/** Format of a row:
	    s_i v_i \n
	    s_i ... Discrete State Index
	    v_i ... V-Value of the state
	*/
	void saveDiscrete1DMatlab(char *filename, CState *initstate, int dim1);

	/// Same as save2DValues, saves the output values in a readable matlab format
	/** Format of a row:
	    s_i s_j v_i \n
	    s_i, s_j ... Values of the first and second continuous state variables
	    v_i ... V-Value of the state
	*/
	void save2DMatlab(char *filename, CState *initstate, int dim1, int part1, int dim2, int part2);

	/// Same as saveDiscrete2DValues, saves the output values in a readable matlab format
	/** Format of a row:
	    s_i s_j v_i \n
	    s_i, s_j ... First and second discrete state index
	    v_i ... V-Value of the state
	*/
	void saveDiscrete2DMatlab(char *filename, CState *initstate, int row_dim, int col_dim);
	
	void setVFunction(CAbstractVFunction *vFunction);
};


/// Analyzer for Q-Functions
/**
With Q-Function Analyzers you can create different tables for showing the shape of your continuous Q-Functions for different actions. These tables can only be saved to a file.
The analyzer supports:
- 1 dimensional Tables
- 2 dimensional Tables
- calculate the Value for each State in a given list
The tables are created for each action in the specified action set.

A 1-dimensional Table can be created with the function save1DValues. You can choose the dimension for which the table shall be created and the number of partitions for this dimension. Addtionally you can specify an init-state, this init-state is used for the state values for all other dimensions. This function can only be used for continuous state variables.
A 2-dimensional Table can be created with the function save2DValues. You can choose both dimension, and the number of partitions for each dimension. The initstate is used in the same way as for 1-D Tables. This function can only be used for continuous state variables.
The table for specific states is created saveStateValues, here you just have to specify the states in the state list object. This function can also be used for discrete states.
*/
class CQFunctionAnalyzer
{
protected:
	CAbstractQFunction *qFunction;
	CStateProperties *modelStateProperties;
	CStateCollectionImpl *stateCollection;
public:
	/// Create a new Q-Function Analyzer
	/**
	The analyzer needs the qFunction, the modelstate and all the modifiers the v-function needs. It is recommended to take the agent's state modifiers (agent->getStateModifiers();).
	*/
	CQFunctionAnalyzer(CAbstractQFunction *qFunction, CStateProperties *modelState, std::list<CStateModifier *> *modifiers);
	virtual ~CQFunctionAnalyzer();
	
	void setQFunction(CAbstractQFunction *l_qFunction);

	/// Create a 1 dimensional Table of the Shape of the Q-Function
	/**A 1-dimensional Table can be created with the function save1DValues. You can choose the dimension for which the table shall be created and the number of partitions for this dimension. Addtionally you can specify an init-state, this init-state is used for the state values for all other dimensions. The tables are created for each action in the specified action set.
. This function can only be used for continuous state variables.*/
	void save1DValues(FILE *stream, CActionSet *action, CState *initstate, int dim1, int part1);

	void save1DValues(char *filename, CActionSet *action, CState *initstate, int dim1, int part1);

	/// saves the maximum actions for a grid over one continuous state variable
	void save1DMaxAction(FILE *stream, CActionSet *action, CState *initstate, int dim1, int part1, char *actionSymbols = NULL);

	void save1DMaxAction(char *filename, CActionSet *action, CState *initstate, int dim1, int part1, char *actionSymbols = NULL);


	/// Same as saveDiscrete2DValues, saves the output values in a readable matlab format
	/** Format of a row:
	    s_i q_1 q_2 ... q_n \n
	    s_i ... Value of continuous State variable
	    q_i ... n Q-Values (n... size of the action set)
	*/
	void save1DMatlab(FILE *stream, CActionSet *action, CState *initstate, int dim1, int part1);

	void save1DMatlab(char *filename, CActionSet *action, CState *initstate, int dim1, int part1);




	/// Create a 1 dimensional Table of the Shape of the Q-Function
	/**A 1-dimensional Table can be created with the function save1DValues. You can choose the dimension for which the table shall be created. Addtionally you can specify an init-state, this init-state is used for the state values for all other dimensions. The tables are created for each action in the specified action set.
. This function can only be used for discrete state variables.*/
	void saveDiscrete1DValues(FILE *stream, CActionSet *action, CState *initstate, int dim1);

	void saveDiscrete1DValues(char *filename, CActionSet *action, CState *initstate, int dim1);

	/// Saves the maximum action for all states of one discrete state variable
	void saveDiscrete1DMaxAction(FILE *stream, CActionSet *action, CState *initstate, int dim1, char *actionSymbols = NULL);

	void saveDiscrete1DMaxAction(char *filename, CActionSet *action, CState *initstate, int dim1, char *actionSymbols = NULL);

	/// Same as saveDiscrete2DValues, saves the output values in a readable matlab format
	/** Format of a row:
	    d_i q_1 q_2 ... q_n \n
	    d_i ... Discrete State Number
	    q_i ... n Q-Values (n... size of the action set)
	*/
	void saveDiscrete1DMatlab(FILE *stream, CActionSet *action, CState *initstate, int dim1);

	void saveDiscrete1DMatlab(char *filename, CActionSet *action, CState *initstate, int dim1);


	/// Create a 2 dimensional Table of the Shape of the Q-Function
	/**A 2-dimensional Table can be created with the function save2DValues. You can choose both dimension, and the number of partitions for each dimension. The initstate is used in the same way as for 1-D Tables. This function can only be used for continuous state variables. The tables are created for each action in the specified action set.*/
	void save2DValues(FILE *stream, CActionSet *action, CState *initstate, int dim1, int part1, int dim2, int part2);

	void save2DValues(char *filename, CActionSet *action, CState *initstate, int dim1, int part1, int dim2, int part2);

	/// save maximum action for a grid over 2 continuous state variables
	void save2DMaxAction(FILE *stream, CActionSet *action, CState *initstate, int dim1, int part1, int dim2, int part2, char *actionSymbols = NULL);

	void save2DMaxAction(char *filename, CActionSet *action, CState *initstate, int dim1, int part1, int dim2, int part2, char *actionSymbols = NULL);



	/// Same as saveDiscrete2DValues, saves the output values in a readable matlab format
	/** Format of a row:
	    s_i s_j q_1 q_2 ... q_n \n
	    s_i s_j ... First and second Continuous State Variable
	    q_i ... n Q-Values (n... size of the action set)
	*/
	void save2DMatlab(FILE *stream, CActionSet *action, CState *initstate, int dim1, int part1, int dim2, int part2);

	void save2DMatlab(char *filename, CActionSet *action, CState *initstate, int dim1, int part1, int dim2, int part2);


	/// Create a 2 dimensional Table of the Shape of the Q-Function
	/**A 2-dimensional Table can be created with the function saveDiscrete2DValues. You can choose the dimensions for which the table shall be created. The dimension row_dim will be used as the row index, col_dim is the column index in the 2D output. The initstate is used in the same way as for 1-D Tables. This function can only be used for discrete state variables. The tables are created for each action in the specified action set.*/
	void saveDiscrete2DValues(FILE *stream, CActionSet *action, CState *initstate, int row_dim, int col_dim);

	void saveDiscrete2DValues(char *filename, CActionSet *action, CState *initstate, int row_dim, int col_dim);

	/// Saves maximum action for all states of 2 discrete state variables
	void saveDiscrete2DMaxAction(FILE *stream, CActionSet *action, CState *initstate, int row_dim, int col_dim, char *actionSymbols = NULL);

	void saveDiscrete2DMaxAction(char *filename, CActionSet *action, CState *initstate, int row_dim, int col_dim, char *actionSymbols = NULL);



	/// Same as saveDiscrete2DValues, saves the output values in a readable matlab format
	/** Format of a row:
	    d_i d_j q_1 q_2 ... q_n \n
	    d_i d_j ... First and second Discrete State Variable
	    q_i ... n Q-Values (n... size of the action set)
	*/
	void saveDiscrete2DMatlab(FILE *stream, CActionSet *action, CState *initstate, int row_dim, int col_dim);

	void saveDiscrete2DMatlab(char *filename, CActionSet *action, CState *initstate, int row_dim, int col_dim);

	/// Creates a table of the values of the specified states, the tables are created for each action in the specified action set.

	void saveStateValues(FILE *stream, CActionSet *action, CStateList *states);

	void saveStateValues(char *filename, CActionSet *action, CStateList *states);
};

/// Super class of the V-Function Comperators and Q-Function comperators.
/**
Function comperators calculate the difference between 2 V-Functions or Q-Functions. You have2 possibilities to do this:
- calculate the difference for N random states (with compareFunctionsRandom)
- calculate the difference for a given state list (with compareFunctionsStates)

For each of these 2 functions you can define the errorfunction which should be used. There are 3 different kinds of errorfunctions:
- MSE: calculate the mean squared error between the 2 functions
- MAE: calculate the mean average error between the 2 functions
- MAXERROR : calculate the maximum error between the 2 functions
*/

class CFunctionComperator : public CRewardFunction
{
protected:
	CStateProperties *modelStateProperties;
	CStateCollectionImpl *stateCollection;

	/// Get the value of the numFunc function for the given state
	virtual double getValue(int numFunc, CStateCollection *state) = 0;
	/// get the difference of the 2 functions for the given state, use the given errorfunction
	virtual double getDifference(CStateCollection *state, int errorFunction);

	/// returns a random state 
	void getRandomState(CState *state);
public:
	/// The comperator needs the properties of the model state and all modifiers the 2 functions use.
	/**
It is recommended to take the agent's state modifiers (agent->getStateModifiers();).
	*/
	CFunctionComperator(CStateProperties *modelState, std::list<CStateModifier *> *modifiers);
	virtual ~CFunctionComperator();

	/// Compares the 2 functions for nSamples random states
	/**
	For more details about the errorfunction see the class description.
	*/
	double compareFunctionsRandom(int nSamples, int errorFunction = 1);
	
	/// Compares the 2 functions for every state in the state list.
	/**
	For more details about the errorfunction see the class description.
	*/
	double compareFunctionsStates(CStateList *states, int errorFunction = 1);
	
	double getReward(CStateCollection *oldState, CAction *action, CStateCollection *newState);
};

/// Comperator for 2 V-Functions
/** 
See the description of the super class for more details.
*/
class CVFunctionComperator : public CFunctionComperator
{
protected:
	CAbstractVFunction *vFunction1;
	CAbstractVFunction *vFunction2;

	virtual double getValue(int numFunc, CStateCollection *state);

public:
	CVFunctionComperator(CStateProperties *modelState, std::list<CStateModifier *> *modifiers, CAbstractVFunction *vFunction1, CAbstractVFunction *vFunction2);
	virtual ~CVFunctionComperator(){};
};

/// Comperator for 2 Q-Functions
/** 
The 2 Q-Functions are compared for the given action in the constructor.
See the description of the super class for more details. 
*/
class CQFunctionComperator : public CFunctionComperator
{
protected:
	CAbstractQFunction *qFunction1;
	CAbstractQFunction *qFunction2;
	CAction *action;

	virtual double getValue(int numFunc, CStateCollection *state);

public:
	CQFunctionComperator(CStateProperties *modelState, std::list<CStateModifier *> *modifiers, CAbstractQFunction *qFunction1, CAbstractQFunction *qFunction2, CAction *action);

	virtual ~CQFunctionComperator(){};
};

/// Controller analyzer 
/**
The controller analyzer supports creating a table of the choosed action for every given state in the state list. This can be done by the function saveActions. The function additionally needs all the modifiers used by the controllers, which are usually the modifiers of the used V or Q Functions.
*/
class CControllerAnalyzer : public CActionObject
{
protected:
	CStateList *states;
	CAgentController *controller;

public:
	CControllerAnalyzer(CStateList *states, CAgentController *controller, CActionSet *actions);
	virtual ~CControllerAnalyzer();

	CStateList *getStateList();
	void setStateList(CStateList *states);

	CAgentController *getController();
	void setController(CAgentController *Controller);

	void saveActions(FILE *stream, std::list<CStateModifier *> *modifiers);
};
/*
/// Action Statistic Analyzer
 
Not documented, not needed.

class CActionStatisticAnalyzer : public CAgentStatisticController
{
protected:
	std::map<void *, std::pair<char *, long>*>* controllers;
	CAgentStatisticController * master;

public:
	CActionStatisticAnalyzer(CAgentStatisticController *master);
	virtual ~CActionStatisticAnalyzer();

	void init();

	void addController(CAgentStatisticController *newcontroller, char *name);

	void printStatistics();

	virtual CAction* getNextAction(CStateCollection *state, CActionStatistics *stat);
};
*/

class CFittedQIterationAnalyzer : public CFittedQIteration, public CTestSuiteEvaluatorLogger
{
protected:
	CPolicySameStateEvaluator *evaluator; 

	int numEvaluations;
	FILE *analyzerFile;
	
	CState *buffState2;
	bool useQValues;

	double lastQValue;
	double lastEstimatedQValue;

	virtual double getValue(CStateCollection *stateCollection, CAction *action);
public:
	CFittedQIterationAnalyzer(CQFunction *qFunction, CAgentController *estimationPolicy, CEpisodeHistory *episodeHistory, CRewardHistory *rewardLogger, CSupervisedQFunctionLearner *learner, CStateProperties *residualProperties, CPolicySameStateEvaluator *evaluator);

	virtual ~CFittedQIterationAnalyzer();

	virtual void addResidualInput(CStep *step, CAction *action, double V, double newV, double nearestNeighborDistance, CAction *nextHistoryAction = NULL, double nextReward = 0.0);

	virtual void evaluate(string evaluationDirectory, int trial, int numEpisodes);	
	
	virtual void startNewEvaluation(string evaluationDirectory, CParameters *parameters, int trial);

};


#endif // CQFUNCTIONANALYZER_H


