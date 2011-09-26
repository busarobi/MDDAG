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

#ifndef C_CONTINUOUSACTIONGRADIENTPOLICY__H
#define C_CONTINUOUSACTIONGRADIENTPOLICY__H

#include "cparameters.h"
#include "cbaseobjects.h"
#include "ccontinuousactions.h"
#include "cgradientfunction.h"


#include <list>
#include <map>
#include <vector>

class CStateProperties;
class CStateCollection;
class CStateCollectionImpl;
class CFeatureList;
class CFeatureCalculator;
class CFeatureVFunction;
class CVFunctionInputDerivationCalculator;

class CStateProperties;

class CCAGradientPolicyInputDerivationCalculator : virtual public CParameterObject
{
protected:

public:

	virtual void getInputDerivation(CStateCollection *inputState, Matrix *targetVector) = 0;

};

class CContinuousActionGradientPolicy : public CContinuousActionController, public CGradientFunction, public CStateObject
{
protected:
	CStateProperties *modelState;

	virtual void updateWeights(CFeatureList *dParams) = 0;

public:
	CContinuousActionGradientPolicy(CContinuousAction *contAction, CStateProperties *modelState);
	~CContinuousActionGradientPolicy();

	virtual void getNextContinuousAction(CStateCollection *state, CContinuousActionData *action) = 0;

	virtual int getNumWeights() = 0;

	virtual void getWeights(double *parameters) = 0;
	virtual void setWeights(double *parameters) = 0;

	virtual void getGradient(CStateCollection *inputState, int outputDimension, CFeatureList *gradientFeatures) = 0;
	virtual void getGradientPre(ColumnVector *input, ColumnVector *outputErrors, CFeatureList *gradientFeatures);

	virtual void getFunctionValuePre(ColumnVector *input, ColumnVector *output);

	virtual void resetData() = 0;
};

class CContinuousActionPolicyFromGradientFunction : public CContinuousActionGradientPolicy, public CCAGradientPolicyInputDerivationCalculator
{
protected:
	CGradientFunction *gradientFunction;
	
	ColumnVector *outputError;
	

	virtual void updateWeights(CFeatureList *dParams);

//	virtual void getInnerContinuousAction(CStateCollection *state, ColumnVector *action);
//	virtual void getInnerInputDerivation(CStateCollection *inputState, Matrix *targetVector);
//	virtual void getInnerGradient(CStateCollection *inputState, int outputDimension, CFeatureList *gradientFeatures);

public:
	CContinuousActionPolicyFromGradientFunction(CContinuousAction *contAction, CGradientFunction *gradientFunction, CStateProperties *modelState);
	~CContinuousActionPolicyFromGradientFunction();

	virtual void getNextContinuousAction(CStateCollection *state, CContinuousActionData *action);
	virtual void getGradient(CStateCollection *inputState, int outputDimension, CFeatureList *gradientFeatures);
	virtual void getInputDerivation(CStateCollection *inputState, Matrix *targetVector);

	virtual int getNumWeights();

	virtual void getWeights(double *parameters);
	virtual void setWeights(double *parameters);


	virtual void resetData();
};

class CContinuousActionFeaturePolicy : public CContinuousActionGradientPolicy, public CCAGradientPolicyInputDerivationCalculator
{
protected:
	std::list<CFeatureCalculator *> *featureCalculators;
	std::list<CFeatureVFunction *> *featureFunctions;

	virtual void updateWeights(CFeatureList *dParams);

	int numWeights;
	CFeatureList *localGradient;

	ColumnVector *inputDerivation;

	std::map<CFeatureVFunction *, CVFunctionInputDerivationCalculator *> *inputDerivationFunctions;

	//virtual void getInnerContinuousAction(CStateCollection *state, ColumnVector *action);
	//virtual void getInnerInputDerivation(CStateCollection *inputState, Matrix *targetVector);
	//virtual void getInnerGradient(CStateCollection *inputState, int outputDimension, CFeatureList *gradientFeatures);

public:
	CContinuousActionFeaturePolicy(CContinuousAction *contAction, CStateProperties *modelState, std::list<CFeatureCalculator *> *featureCalcualtors);
	~CContinuousActionFeaturePolicy();

	virtual int getNumWeights();

	virtual void getWeights(double *parameters);
	virtual void setWeights(double *parameters);

	virtual void resetData();

	virtual void getNextContinuousAction(CStateCollection *state, CContinuousActionData *action);
	virtual void getGradient(CStateCollection *inputState, int outputDimension, CFeatureList *gradientFeatures);
	virtual void getInputDerivation(CStateCollection *inputState, Matrix *targetVector);
};

class CContinuousActionSigmoidPolicy : public CContinuousActionGradientPolicy, public CCAGradientPolicyInputDerivationCalculator
{
protected:
	CContinuousActionGradientPolicy *policy;
	CCAGradientPolicyInputDerivationCalculator *inputDerivation;

	CContinuousActionData *contData;

	virtual void updateWeights(CFeatureList *dParams);

public:
	CContinuousActionSigmoidPolicy(CContinuousActionGradientPolicy *policy, CCAGradientPolicyInputDerivationCalculator *inputDerivation);
	~CContinuousActionSigmoidPolicy();


	virtual int getNumWeights();

	virtual void getWeights(double *parameters);
	virtual void setWeights(double *parameters);

	virtual void resetData();

	virtual void getNextContinuousAction(CStateCollection *state, CContinuousActionData *action);
	virtual void getGradient(CStateCollection *inputState, int outputDimension, CFeatureList *gradientFeatures);
	virtual void getInputDerivation(CStateCollection *inputState, Matrix *targetVector);

	virtual void getNoise(CStateCollection *state, CContinuousActionData *action, CContinuousActionData *l_noise);

};

class CCAGradientPolicyNumericInputDerivationCalculator : public CCAGradientPolicyInputDerivationCalculator
{
protected:
	CContinuousActionGradientPolicy *policy;

	CContinuousActionData *contDataPlus;
	CContinuousActionData *contDataMinus;

	CStateCollectionImpl *stateBuffer;
public:
	CCAGradientPolicyNumericInputDerivationCalculator(CContinuousActionGradientPolicy *policy, double stepSize,  std::list<CStateModifier *> *modifiers);
	~CCAGradientPolicyNumericInputDerivationCalculator();

	virtual void getInputDerivation(CStateCollection *inputState, Matrix *targetVector);
};

#endif

