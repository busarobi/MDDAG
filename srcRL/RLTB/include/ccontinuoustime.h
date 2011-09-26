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

#ifndef C_CONTINUOUSTIME
#define C_CONTINUOUSTIME


#include "cpolicies.h"
#include "ccontinuousactions.h"
#include "ccontinuousactiongradientpolicy.h"
#include "newmat/newmat.h"



class CVFunctionInputDerivationCalculator;
class CContinuousTimeTransitionFunction;
class CContinuousTimeQFunctionFromTransitionFunction;
class CRewardFunction;
class CTransitionFunction;
class CGradientVFunction;
class CStateCollection;

class CContinuousTimeParameters
{
public:
	static double getGammaFromSgamma(double sgamma, double dt);
	static double getLambdaFromKappa(double kappa, double sgamma, double dt);
};

class CContinuousTimeVMPolicy : public CQStochasticPolicy
{
protected:
	CVFunctionInputDerivationCalculator *vfunction;
	CContinuousTimeTransitionFunction *model;
public:

	CContinuousTimeVMPolicy(CActionSet *actions, CActionDistribution *distribution, CVFunctionInputDerivationCalculator *vFunction, CContinuousTimeTransitionFunction *model, CRewardFunction *rewardFunction);
	~CContinuousTimeVMPolicy();

	CContinuousTimeQFunctionFromTransitionFunction *getQFunctionFromTransitionFunction();

};

class CContinuousTimeAndActionVMPolicy : public CContinuousActionController
{
protected:
	CVFunctionInputDerivationCalculator *dVFunction;
	CTransitionFunction *model;

	ColumnVector *actionValues;
	ColumnVector *derivationX;
	Matrix *derivationU;

	virtual void getActionValues(ColumnVector *actionValues, ColumnVector *noise) = 0;
public:
	CContinuousTimeAndActionVMPolicy(CContinuousAction *action, CVFunctionInputDerivationCalculator *dVFunction, CTransitionFunction *model);
	~CContinuousTimeAndActionVMPolicy();

	virtual void getNextContinuousAction(CStateCollection *state, CContinuousActionData *contAction);

	
};

class CContinuousTimeAndActionSigmoidVMPolicy : public CContinuousTimeAndActionVMPolicy
{
protected:

	ColumnVector *c;

	void getActionValues(ColumnVector *actionValues, ColumnVector *noise);

public:
	CContinuousTimeAndActionSigmoidVMPolicy(CContinuousAction *action, CVFunctionInputDerivationCalculator *vfunction, CTransitionFunction *model);
	~CContinuousTimeAndActionSigmoidVMPolicy();

	void setC(int index, double value);
	double getC(int index);

	ColumnVector *getC() {return c;};

	virtual void getNoise(CStateCollection *state, CContinuousActionData *action, CContinuousActionData *noise);
};

class CContinuousTimeAndActionSigmoidVMGradientPolicy : public CContinuousActionGradientPolicy
{
protected:
	CGradientVFunction *vFunction;
	CStateCollectionImpl *derivationState;

	CFeatureList *gradient1;
	CFeatureList *gradient2;

	virtual void updateWeights(CFeatureList *dParams);

	
	CVFunctionInputDerivationCalculator *dVFunction;
	CTransitionFunction *model;
	ColumnVector *actionValues;
	ColumnVector *derivationX;
	Matrix *derivationU;

	ColumnVector *c;

	void getActionValues(ColumnVector *actionValues, ColumnVector *noise);
	virtual void getGradientActionValues(ColumnVector *, ColumnVector *) {};




public:
	CContinuousTimeAndActionSigmoidVMGradientPolicy(CContinuousAction *action, CGradientVFunction *gradVFunction, CVFunctionInputDerivationCalculator *vfunction, CTransitionFunction *model, std::list<CStateModifier *> *modifiers);
	virtual ~CContinuousTimeAndActionSigmoidVMGradientPolicy();

	virtual int getNumWeights();

	virtual void getWeights(double *parameters);
	virtual void setWeights(double *parameters);

	virtual void getGradient(CStateCollection *inputState, int outputDimension, CFeatureList *gradientFeatures);

	virtual void resetData();

	virtual void getNextContinuousAction(CStateCollection *state, CContinuousActionData *contAction);

	void setC(int index, double value);
	double getC(int index);

	ColumnVector *getC() {return c;};

	virtual void getNoise(CStateCollection *state, CContinuousActionData *action, CContinuousActionData *noise);
};

class CContinuousTimeAndActionBangBangVMPolicy : public CContinuousTimeAndActionVMPolicy
{
protected:
	virtual void getActionValues(ColumnVector *actionValues, ColumnVector *noise);
 
public:
	CContinuousTimeAndActionBangBangVMPolicy(CContinuousAction *action, CVFunctionInputDerivationCalculator *vfunction, CTransitionFunction *model);

	virtual void getNoise(CStateCollection *state, CContinuousActionData *action, CContinuousActionData *noise);
};

class CContinuousActionSmoother : public CContinuousActionController
{
protected:
	CContinuousActionController *policy;
	double *actionValues;

	double alpha;
public:
	CContinuousActionSmoother(CContinuousAction *action, CContinuousActionController *policy, double alpha = 0.3);
	~CContinuousActionSmoother();

	virtual void getNextContinuousAction(CStateCollection *state, CContinuousActionData *contAction);

	void setAlpha(double alpha);
	virtual double getAlpha();

};

#endif

