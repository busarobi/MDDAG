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


#ifndef __CCONTINUOUSRL_H
#define __CCONTINUOUSRL_H




#include "caction.h"
#include "cutility.h"
#include "cqfunction.h"
#include "cqetraces.h"
#include "cagentcontroller.h"
#include "cagentlistener.h"

class CAbstractQFunction;
class CQFunction;

/// class for saving the continuous Values from a ContinuosAction
/**
@see CActionData
*/
class CContinuousActionData : public CMultiStepActionData, public ColumnVector
{
protected:
	
public:
	CContinuousActionData(CContinuousActionProperties *properties);
	virtual ~CContinuousActionData();

	CContinuousActionProperties *properties;

	virtual void setActionValue(int dim, double value);
	double getActionValue(int dim);

//	double *getActionValues();

	void normalizeAction();
	
	double getDistance(ColumnVector *vector);

	virtual void saveASCII(FILE *stream);
	virtual void loadASCII(FILE *stream);

	virtual void saveBIN(FILE *stream);
	virtual void loadBIN(FILE *stream);

	virtual void setData(CActionData *actionData);
	void initData(double initVal);
};

class CContinuousActionProperties
{
protected:
	unsigned int numActionValues;
	double *minValues;
	double *maxValues;
public:
	CContinuousActionProperties(int numActionValues);
	virtual ~CContinuousActionProperties();

	unsigned int getNumActionValues();

	double getMinActionValue(int dim);
	double getMaxActionValue(int dim);

	void setMinActionValue(int dim, double value);
	void setMaxActionValue(int dim, double value);
};

class CContinuousAction : public CPrimitiveAction
{
protected:
	CContinuousActionData *continuousActionData;
	CContinuousActionProperties *properties;

	CContinuousAction(CContinuousActionProperties *properties, CContinuousActionData *actionData);
public:
	CContinuousAction(CContinuousActionProperties *properties);
	virtual ~CContinuousAction();


	CContinuousActionProperties *getContinuousActionProperties();

	virtual CContinuousActionData *getContinuousActionData() {return continuousActionData;};

	virtual CActionData *getNewActionData();

	double getActionValue(int dim);
	unsigned int getNumDimensions();

	virtual void loadActionData(CActionData *data);

	virtual bool equals(CAction *action);
	virtual bool isSameAction(CAction *action, CActionData *data);
};

#define NO_RANDOM_CONTROLLER 0
#define EXTERN_RANDOM_CONTROLLER 1
#define INTERN_RANDOM_CONTROLLER 2

class CContinuousActionRandomPolicy;

class CContinuousActionController : public CAgentController
{
protected:
	CContinuousAction *contAction;

	CContinuousActionRandomPolicy *randomController;
	CContinuousActionData *noise;

	int randomControllerMode;
public:
	CContinuousActionController(CContinuousAction *contAction, int randomControllerMode = 1);
	virtual ~CContinuousActionController();

	virtual CAction *getNextAction(CStateCollection *state, CActionDataSet *data = NULL);
	virtual void getNextContinuousAction(CStateCollection *state, CContinuousActionData *action) = 0;

	virtual CContinuousActionProperties *getContinuousActionProperties() {return contAction->getContinuousActionProperties();};
	virtual CContinuousAction *getContinuousAction() {return contAction;};

	virtual void setRandomController(CContinuousActionRandomPolicy *randomController);
	virtual CContinuousActionRandomPolicy *getRandomController();

	void setRandomControllerMode(int randomControllerMode);
	int getRandomControllerMode();

	virtual void getNoise(CStateCollection *state, CContinuousActionData *action, CContinuousActionData *noise);


};


class CStaticContinuousAction : public CContinuousAction
{
protected:
	CContinuousAction *contAction;

	double maximumDistance;
public:
	CStaticContinuousAction(CContinuousAction *properties, double *actionValues, double maximumDistance = 0.0);
	virtual ~CStaticContinuousAction();

	virtual void setContinuousAction(CContinuousActionData *contAction);
	virtual void addToContinuousAction(CContinuousActionData *contAction, double factor);

	CContinuousAction *getContinuousAction();

	virtual void loadActionData(CActionData *) {};
	virtual void setData(CActionData *) {assert(false);};

	virtual bool equals(CAction *action);
	virtual bool isSameAction(CAction *action, CActionData *data);

	virtual double getMaximumDistance();
};


class CLinearFAContinuousAction : public CStaticContinuousAction
{
protected:
public:
	CLinearFAContinuousAction(CContinuousAction *properties, double *actionValues);
	virtual ~CLinearFAContinuousAction() {};

	virtual double getActionFactor(CContinuousActionData *contAction) = 0;

};

class CContinuousRBFAction : public CLinearFAContinuousAction
{
protected:
	double *rbfSigma;
public:
	CContinuousRBFAction(CContinuousAction *properties, double *rbfCenter, double *rbfSigma);
	virtual ~CContinuousRBFAction();

	virtual double getActionFactor(CContinuousActionData *contAction);
};

class CContinuousActionLinearFA
{
protected:
	CActionSet *contActions;
	CContinuousActionProperties *actionProperties;

public:

	CContinuousActionLinearFA(CActionSet *contActions, CContinuousActionProperties *properties);
	virtual ~CContinuousActionLinearFA();

	void getActionFactors(CContinuousActionData *action, double *actionFactors);
	
	void getContinuousAction(unsigned int index, CContinuousActionData *action);
	void getContinuousAction(CContinuousActionData *action, double *actionFactors);

	int getNumContinuousActionFA();
};


class CCALinearFAQETraces;

class CContinuousActionQFunction : public CGradientQFunction
{
protected:
	CContinuousAction *contAction;
public:
	CContinuousActionQFunction(CContinuousAction *contAction);
	virtual ~CContinuousActionQFunction();

	virtual CAction *getMax(CStateCollection *, CActionSet *availableActions, CActionDataSet *actionDatas);

	virtual void getBestContinuousAction(CStateCollection *state, CContinuousActionData *actionData) = 0;

	virtual void updateValue(CStateCollection *state, CAction *action, double td, CActionData *data = NULL);
	/// Sets the Value of the value function assigned to the given action
	/** Calls the setValue Function of the specified value function.
	*/
	virtual void setValue(CStateCollection *state, CAction *action, double qValue, CActionData *data = NULL); 
	/// Returns the Value of the value function assigned to the given action
	/** Returns the value of  the getValue Function of the specified value function.
	*/
	virtual double getValue(CStateCollection *state, CAction *action, CActionData *data = NULL);


	virtual void updateCAValue(CStateCollection *state, CContinuousActionData *data, double td);
	virtual void setCAValue(CStateCollection *state, CContinuousActionData *data, double qValue); 
	virtual double getCAValue(CStateCollection *state, CContinuousActionData *data) = 0;


	virtual void getGradient(CStateCollection *state, CAction *action, CActionData *data, CFeatureList *gradient);
	virtual void getCAGradient(CStateCollection *state, CContinuousActionData *data, CFeatureList *gradient);

	CContinuousAction *getContinuousActionObject() {return contAction;};

	virtual int getNumWeights() {return 0;};

	virtual void getWeights(double *parameters);
	virtual void setWeights(double *parameters);

	//virtual CAbstractQETraces* getStandardETraces() = 0;
};

class CCALinearFAQFunction : public CContinuousActionQFunction, public CContinuousActionLinearFA
{
protected:
	double *actionFactors;
	double *CAactionValues;
	CQFunction *qFunction;

	CFeatureList *tempGradient;

	virtual void updateWeights(CFeatureList *features);

public:

	CCALinearFAQFunction(CQFunction *qFunction, CContinuousAction *returnAction);

	virtual ~CCALinearFAQFunction();

	virtual void getBestContinuousAction(CStateCollection *state, CContinuousActionData *actionData);

	virtual void updateCAValue(CStateCollection *state, CContinuousActionData *data, double td);
	virtual void setCAValue(CStateCollection *state, CContinuousActionData *data, double qValue); 
	virtual double getCAValue(CStateCollection *state, CContinuousActionData *data);


	CQFunction *getQFunctionForCA();

	virtual CAbstractQETraces* getStandardETraces();
	virtual void getCAGradient(CStateCollection *state, CContinuousActionData *action, CFeatureList *gradient);

	virtual int getNumWeights();

	virtual void getWeights(double *weights);
	virtual void setWeights(double *weights);

	virtual int getWeightsOffset(CAction *) {return 0;};
};

class CCALinearFAQETraces : public CQETraces
{
protected:
	double *actionFactors;
	CCALinearFAQFunction *contQFunc;
	
public:

	CCALinearFAQETraces(CCALinearFAQFunction *qfunction);
	virtual ~CCALinearFAQETraces();

	virtual void addETrace(CStateCollection *State, CAction *action, double factor = 1.0, CActionData *data = NULL);
};

class CActionDistribution;

class CContinuousActionPolicy : public CContinuousActionController
{
protected:
	CActionDistribution *distribution;
	double *actionValues;
	CAbstractQFunction *continuousActionQFunc;

	CActionSet *continuousStaticActions;

public:
	CContinuousActionPolicy(CContinuousAction *contAction, CActionDistribution *distribution, CAbstractQFunction *continuousActionQFunc, CActionSet *continuousStaticActions);
	virtual ~CContinuousActionPolicy();

	virtual void getNextContinuousAction(CStateCollection *state, CContinuousActionData *action);

};

class CContinuousActionRandomPolicy : public CContinuousActionController, public CSemiMDPListener
{
protected:
	ColumnVector *lastNoise;
	ColumnVector *currentNoise;
	
	double sigma;
	double alpha;
public: 
	CContinuousActionRandomPolicy(CContinuousAction *action, double sigma, double alpha);
	virtual ~CContinuousActionRandomPolicy();

	virtual void newEpisode();
	virtual void nextStep(CStateCollection *, CAction *, CStateCollection *);

	virtual void getNextContinuousAction(CStateCollection *state, CContinuousActionData *action);
	
	virtual void onParametersChanged();
	
	ColumnVector *getCurrentNoise();
	ColumnVector *getLastNoise();
};


class CContinuousActionAddController : public CContinuousActionController
{
protected:
	std::list<CContinuousActionController *> *controllers;

	std::map<CContinuousActionController *,double> *controllerWeights;

	ColumnVector *actionValues;
public:
	CContinuousActionAddController(CContinuousAction *action);
	virtual ~CContinuousActionAddController();

	virtual void getNextContinuousAction(CStateCollection *state, CContinuousActionData *action);

	void addContinuousActionController(CContinuousActionController *controller, double weight = 1.0);
	void setControllerWeight(CContinuousActionController *controller, double weight);
	double getControllerWeight(CContinuousActionController *controller);

};


#endif

