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

#ifndef __CTransitionFunction_H
#define __CTransitionFunction_H

#include "cparameters.h"
#include "cbaseobjects.h"
#include "crewardfunction.h"
#include "cenvironmentmodel.h"
#include "cqfunction.h"


#define DM_CONTINUOUSMODEL 1
#define DM_DERIVATIONUMODEL 2
#define DM_EXTENDEDACTIONMODEL 4

#define DM_RESET_TYPE_ALL_RANDOM 2
#define DM_RESET_TYPE_RANDOM 1
#define DM_RESET_TYPE_ZERO 0

class CStateCollectionImpl;
class CStateCollectionList;
class CActionDataSet;
class CContinuousActionProperties;
class CContinuousAction;
class CStateList;
class CRegion;
class CAbstractFeatureStochasticModel;
class CVFunctionInputDerivationCalculator;
class CAbstractVFunction;

class CTransitionFunction : public CStateObject, public CActionObject, virtual public CParameterObject
{
protected:
	int type;

	int resetType;
public:
	CTransitionFunction(CStateProperties *properties, CActionSet *actions);

	int getType();
	void addType(int Type);
	bool isType(int type);

	virtual void transitionFunction(CState *oldstate, CAction *action, CState *newState, CActionData *data = NULL) = 0;
	
	virtual void getDerivationU(CState *oldstate, Matrix *derivation);

	virtual bool isResetState(CState *) {return false;};
	virtual bool isFailedState(CState *) {return false;};

	virtual void getResetState(CState *resetState);

	virtual void setResetType(int resetType);
};

class CExtendedActionTransitionFunction : public CTransitionFunction, public CRewardFunction
{
protected:
	CTransitionFunction *dynModel;

	CStateCollectionImpl *intermediateState;
	CStateCollectionImpl *nextState;

	CActionDataSet *actionDataSet;

	CRewardFunction *rewardFunction;
	double lastReward;
public:
	CExtendedActionTransitionFunction(CActionSet *actions, CTransitionFunction *model, std::list<CStateModifier *> *modifiers, CRewardFunction *rewardFunction = NULL) ;
	~CExtendedActionTransitionFunction();

	virtual void transitionFunction(CState *oldstate, CAction *action, CState *newState, CActionData *data = NULL);
	virtual double transitionFunctionAndReward(CState *oldState, CAction *action, CState *newState, CActionData *data, CRewardFunction *reward, double gamma);

	virtual void getDerivationU(CState *oldstate, Matrix *derivation);

	virtual bool isResetState(CState *state);
	virtual bool isFailedState(CState *state);

	virtual void getResetState(CState *resetState);

	virtual void setResetType(int resetType);

	virtual double getReward(CStateCollection *oldState, CAction *action, CStateCollection *newState);
};


class CComposedTransitionFunction : public CTransitionFunction
{
protected:
	std::list<CTransitionFunction *> *TransitionFunction;
public:

	CComposedTransitionFunction(CStateProperties *properties);
	~CComposedTransitionFunction();

	void addTransitionFunction(CTransitionFunction *model);

	virtual void transitionFunction(CState *oldstate, CAction *action, CState *newState, CActionData *data = NULL);
};

class CContinuousTimeTransitionFunction : public CTransitionFunction
{
protected:
	double dt;
	int simulationSteps;

	ColumnVector *derivation;

	virtual void doSimulationStep(CState *oldState, double timeStep, CAction *action, CActionData *data);

public:
	CContinuousTimeTransitionFunction(CStateProperties *properties, CActionSet *actions, double dt);
	virtual ~CContinuousTimeTransitionFunction();

	virtual void transitionFunction(CState *oldstate, CAction *action, CState *newState, CActionData *data = NULL);

	double getTimeIntervall();
	void setTimeIntervall(double dt);

	void setSimulationSteps(int steps);
	int getSimulationSteps();

	virtual void getDerivationX(CState *oldstate, CAction *action, ColumnVector *derivation, CActionData *data = NULL) = 0;
};

class CContinuousAction;
class CContinuousActionData;

class CContinuousTimeAndActionTransitionFunction : public CContinuousTimeTransitionFunction
{
protected:
	CContinuousActionProperties *actionProp;
	CContinuousAction *contAction;
public:
	CContinuousTimeAndActionTransitionFunction(CStateProperties *properties, CContinuousAction *action, double dt);
	virtual ~CContinuousTimeAndActionTransitionFunction();
	
	virtual void getDerivationX(CState *oldState, CAction *action, ColumnVector *derivationX, CActionData *data = NULL);
	virtual void getCADerivationX(CState *oldState, CContinuousActionData *action, ColumnVector *derivationX) = 0;

	
	CContinuousAction *getContinuousAction();
};


class CLinearActionContinuousTimeTransitionFunction : public CContinuousTimeAndActionTransitionFunction
{
protected:

	ColumnVector *A;
	Matrix *B;

public:
	CLinearActionContinuousTimeTransitionFunction(CStateProperties *properties, CContinuousAction *action, double dt);
	~CLinearActionContinuousTimeTransitionFunction();

	virtual void getCADerivationX(CState *oldState, CContinuousActionData *action, ColumnVector *derivationX);

	virtual void getDerivationU(CState *oldstate, Matrix *derivation);
	virtual Matrix *getB(CState *state) = 0;
	virtual ColumnVector *getA(CState *state) = 0;

};

class CDynamicLinearContinuousTimeModel : public CLinearActionContinuousTimeTransitionFunction
{
protected:
	Matrix *B;
	Matrix *AMatrix;

public:
	CDynamicLinearContinuousTimeModel(CStateProperties *properties, CContinuousAction *action, double dt, Matrix *A, Matrix *B);
	~CDynamicLinearContinuousTimeModel();

	virtual Matrix *getB(CState *state);
	virtual ColumnVector *getA(CState *state);
};


class CTransitionFunctionEnvironment : public CEnvironmentModel
{
protected:
	CTransitionFunction *TransitionFunction;
	CState *modelState;
	CState *nextState;

	CStateList *startStates;
	int nEpisode;
	bool createdStartStates;

	CRegion *failedRegion;
	CRegion *sampleRegion;
	CRegion *targetRegion;
public:
	CTransitionFunctionEnvironment(CTransitionFunction *model);
	virtual ~CTransitionFunctionEnvironment();

	virtual void doNextState(CPrimitiveAction *action);
	virtual void doResetModel();

	virtual void getState(CState *state);

	virtual void setState(CState *state);

	virtual void setStartStates(CStateList *startStates);
	virtual void setStartStates(char *filename);

	CTransitionFunction *getTransitionFunction() {return TransitionFunction;};

	void setSampleRegion(CRegion *sampleRegion);
	void setFailedRegion(CRegion *failedRegion);
	void setTargetRegion(CRegion *sampleRegion);
};

class CTransitionFunctionFromStochasticModel : public CTransitionFunction
{
protected:
	CAbstractFeatureStochasticModel *stochasticModel;

	std::list<int> *startStates;
	std::list<double> *startProbabilities;
	std::map<int, double> *endStates;
public:
	CTransitionFunctionFromStochasticModel(CStateProperties *properties, CAbstractFeatureStochasticModel *model);
	~CTransitionFunctionFromStochasticModel();

	virtual void transitionFunction(CState *oldstate, CAction *action, CState *newState, CActionData *data = NULL);	

	void addStartState(int state, double probability);
	void addEndState(int state, double probability);

	virtual bool isResetState(CState *state);
	virtual void getResetState(CState *state);
};

class CQFunctionFromTransitionFunction : public CAbstractQFunction, public CStateModifiersObject 
{
protected:

	/// The given V-Function
	CAbstractVFunction *vfunction;
	/// The model
	CTransitionFunction *model;
	/// feature Reward Function for the learning problem.
	CRewardFunction *rewardfunction;

	/// state buffer
	CStateCollectionImpl *intermediateState;	
	CStateCollectionImpl *nextState;


	CStateCollectionList *stateCollectionList;
	CActionDataSet *actionDataSet;

public:
	/// Creates a new QFunction from VFunction object for the given V-Function and the given model, the discretizer is take nfrom the V-Function

	CQFunctionFromTransitionFunction(CActionSet *actions, CAbstractVFunction *vfunction, CTransitionFunction *model, CRewardFunction *rewardfunction, std::list<CStateModifier *> *modifiers);

	virtual ~CQFunctionFromTransitionFunction();

	/// Writes the Action-Values in the actionValues Array.
	//void getActionValues(CStateCollection *state, double *actionValues, CActionSet *actions);

	/// Does nothing
	virtual void setValue(CStateCollection *, CAction *, double , CActionData * = NULL) {};

	/// Does nothing
	virtual void updateValue(CStateCollection *, CAction *, double , CActionData * = NULL) {}; 


	/// getValue function for state collections
	/** Calls the getValue function for the specific state (retrieved from the collection by the discretizer)*/
	virtual double getValue(CStateCollection *state, CAction *action, CActionData *data = NULL);

	double getValueDepthSearch(CStateCollectionList *state, CAction *action, CActionData *data, int depth);

	virtual CAbstractQETraces *getStandardETraces() {return NULL;};

	virtual void addStateModifier(CStateModifier *modifier);
};

class CContinuousTimeQFunctionFromTransitionFunction : public CAbstractQFunction, public CStateModifiersObject 
{
protected:

	/// The given V-Function
	CVFunctionInputDerivationCalculator *vfunction;
	/// The model
	CContinuousTimeTransitionFunction *model;
	/// feature Reward Function for the learning problem.
	CRewardFunction *rewardfunction;

	CStateCollectionImpl *nextState;

	CState *derivationXModel;
	CState *derivationXVFunction;

	virtual double getValueVDerivation(CStateCollection *state, CAction *action, CActionData *data, ColumnVector *derivationXVFunction);
public:
	/// Creates a new QFunction from VFunction object for the given V-Function and the given model, the discretizer is take nfrom the V-Function

	CContinuousTimeQFunctionFromTransitionFunction(CActionSet *actions, CVFunctionInputDerivationCalculator *vfunction, CContinuousTimeTransitionFunction *model, CRewardFunction *rewardfunction, std::list<CStateModifier *> *modifiers);

	CContinuousTimeQFunctionFromTransitionFunction(CActionSet *actions, CVFunctionInputDerivationCalculator *vfunction, CContinuousTimeTransitionFunction *model, CRewardFunction *rewardfunction);

	virtual ~CContinuousTimeQFunctionFromTransitionFunction();

	virtual void getActionValues(CStateCollection *state, CActionSet *actions, double *actionValues, CActionDataSet *actionDataSet);

	/// Does nothing
	virtual void setValue(CStateCollection *, CAction *, double , CActionData * = NULL) {};

	/// Does nothing
	virtual void updateValue(CStateCollection *, CAction *, double , CActionData * = NULL) {}; 


	/// getValue function for state collections
	/** Calls the getValue function for the specific state (retrieved from the collection by the discretizer)*/
	virtual double getValue(CStateCollection *state, CAction *action, CActionData *data = NULL);


	virtual CAbstractQETraces *getStandardETraces() {return NULL;};

	virtual void addStateModifier(CStateModifier *modifier);

};



#endif

