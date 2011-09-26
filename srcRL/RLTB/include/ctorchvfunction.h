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

#ifndef CVTORCHFUNCTION_H
#define CVTORCHFUNCTION_H

#include "ConnectedMachine.h"
#include "MLP.h"
#include "cvfunction.h"
#include "ccontinuousactions.h"


using Torch::Sequence;
using Torch::Parameters;
using Torch::Machine;
using Torch::GradientMachine;

/// Interface to integrate Torch Machines in the learning systems
/** This class only handels the output of the torch machines. It provides a function
for transforming a RIL toolbox state object in a torch input sequence. Therefore the continuous and then the discrete state variables,
are written in the sequence. When the function getValue is called the state is converted into the torch sequence
and then passed to the torch machine, and the ouput of the machine is returned. Learning (i.e. setting and updating the Values) can 
only be used with the class CVFunctionFromGradientFunction.
@see CVFunctionFromGradientFunction
*/


class CTorchFunction
{
protected:
	/// Pointer to the torch machine
	Machine *machine;
		
	/// The input sequence, needed to feed the torch machines
	Sequence *input;

public:
	/// Initialises the input sequence with one frame of size |continuousStates| + |discreteStates| of the properties object.
	CTorchFunction(Machine *machine);
	virtual ~CTorchFunction();

	

	///Returns the torch machine
	virtual Machine *getMachine();

	virtual double getValueFromMachine(Sequence *state);
};

/// Class for learning with Torch-Gradient machines
/**
Extends the ability from CTorchVFunction, to learn with a torch gradient machine. The parameters of the machine 
are updated by adding the current gradient of the parameters  multplied with the difference
given by updateValue. 

*/
class CTorchGradientFunction : public CTorchFunction, public CGradientFunction
{
protected:
	Sequence *alpha;
	/// Pointer to the gradient Machine
	GradientMachine *gradientMachine;

	CAdaptiveEtaCalculator *localEtaCalc;
public:
	/// Creates a new value function learning with a torch gradient machine	
	CTorchGradientFunction(int numInputs, int numOutputs);
	CTorchGradientFunction(GradientMachine *machine);
	virtual ~CTorchGradientFunction();

	/// Resets the parameters of the gradient machine
	virtual void resetData();

	virtual void updateWeights(CFeatureList *gradientFeatures);

	virtual int getNumWeights();

	virtual void getInputDerivationPre(ColumnVector *input, Matrix *targetVector);
	virtual void getFunctionValuePre(ColumnVector *input, ColumnVector *output);


	virtual void getWeights(double *parameters);
	virtual void setWeights(double *parameters);

	virtual void getGradientPre(ColumnVector *input, ColumnVector *outputErrors, CFeatureList *gradientFeatures);

	void setGradientMachine(GradientMachine *gradientMachine);
	GradientMachine *getGradientMachine();
};

class CTorchGradientEtaCalculator : public CIndividualEtaCalculator
{
public:
	CTorchGradientEtaCalculator(GradientMachine *gradientMachine);
};

class CTorchVFunction :  public CAbstractVFunction
{
protected:
	/// Converts the state in an torch sequence
	/** the sequence has to have the frame size |continuousStates| + |discreteStates| of the state.
	*/
	void getInputSequence(CState *state, Sequence *input);

	CTorchFunction *torchFunction;

	Sequence *input;

public:
	/// Initialises the input sequence with one frame of size |continuousStates| + |discreteStates| of the properties object.
	CTorchVFunction(CTorchFunction *torchFunction, CStateProperties *properties);
	virtual ~CTorchVFunction();

	/// Converts the state into an input sequence, tansfers the sequence to the machine and returns its output.
	virtual double getValue(CState *state);

};


/// Class for learning with Torch-Gradient machines
/**
Extends the ability from CTorchVFunction, to learn with a torch gradient machine. The parameters of the machine 
are updated by adding the current gradient of the parameters  multplied with the difference
given by updateValue. 

*/
class CVFunctionFromGradientFunction : public CGradientVFunction, public CVFunctionInputDerivationCalculator
{
protected:
	/// Pointer to the gradient Machine
	CGradientFunction *gradientFunction;

	ColumnVector *input;
	ColumnVector *outputError;
	Matrix *inputDerivation;

	virtual void updateWeights(CFeatureList *gradientFeatures);
	void getInputSequence(CState *state, ColumnVector *sequence);

public:
/// Creates a new value function learning with a torch gradient machine	
	CVFunctionFromGradientFunction(CGradientFunction *gradientFunction, CStateProperties *properties);
	virtual ~CVFunctionFromGradientFunction();

/// Calls update value with "value" - currentValue as parameter.
/** For learning only updateValue should be used.*/
	virtual void setValue(CState *state, double value);
/// Updates the parameters of the machine by adding the gradient to the parameters.


/// Resets the parameters of the gradient machine
	virtual void resetData();

	/// Converts the state into an input sequence, tansfers the sequence to the machine and returns its output.
	virtual double getValue(CState *state);

	//virtual CStateProperties *getGradientCalculator();

	virtual void getGradient(CStateCollection *originalState, CFeatureList *modifiedState);

	virtual int getNumWeights();

	virtual CAbstractVETraces *getStandardETraces();
	
	void getInputDerivation(CStateCollection *originalState, ColumnVector *targetVector);

	virtual void getWeights(double *parameters);
	virtual void setWeights(double *parameters); 

};

class CQFunctionFromGradientFunction : public CContinuousActionQFunction, CStateObject
{
protected:
	CGradientFunction *gradientFunction;
	ColumnVector *input;
	ColumnVector *outputError;

	CActionSet *staticActions;

	void getInputSequence(ColumnVector *input, CState *state, CContinuousActionData *data);
	virtual void updateWeights(CFeatureList *gradientFeatures);

public:
	CQFunctionFromGradientFunction(CContinuousAction *contAction, CGradientFunction *torchGradientFunction, CActionSet *actions, CStateProperties *properties);
	virtual ~CQFunctionFromGradientFunction();

	virtual void getBestContinuousAction(CStateCollection *state, CContinuousActionData *actionData);

	virtual void updateCAValue(CStateCollection *state, CContinuousActionData *data, double td);
	virtual void setCAValue(CStateCollection *state, CContinuousActionData *data, double qValue); 
	virtual double getCAValue(CStateCollection *state, CContinuousActionData *data);


	virtual void getCAGradient(CStateCollection *state, CContinuousActionData *data, CFeatureList *gradient);
	virtual int getNumWeights();

	virtual void getWeights(double *parameters);
	virtual void setWeights(double *parameters);

	virtual void resetData();
};



#endif
