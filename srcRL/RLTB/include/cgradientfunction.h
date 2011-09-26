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

#ifndef C_GRADIENTFUNCTION__H
#define C_GRADIENTFUNCTION__H

#include "cparameters.h"
#include "clearndataobject.h"
#include <newmat/newmat.h>

class CFeatureList;

/// Adaptive Learning Rate Calculator Interface class
/** 
Adaptive Learning Rate (Eta) Calculators calculate the learning rate for the weights of a gradientfunction. This learning rate can be calculated for each weight individually, so it gives you more freedom to update your gradientfunction. For every gradient function you can specify a eta calculator, before updating the weights of the gradient function, the eta calculator can modify the gradient feature list for updating (i.e. multiply the features of the gradient with the learning rates). This is done in the method getWeightUpdates(CFeatureList), which has to be implemented by all subclass. If no eta calculator is defined for a gradient function, the gradient isn't changed (default for the most gradient functions).
There are 2 different eta calculators at your disposal:
- CIndividualEtaCalculator: You can set a constant, but for each weight individual learning rate. Used for example for Neural Networks.
- CVarioEta: Calculates the learning rate of a weight according to its standard deviation. For more details see (Coulom, 1) of the paper section.
*/

class CAdaptiveEtaCalculator : virtual public CParameterObject
{
public:
	/// Multiply the update gradient list with the learning rates
	virtual void getWeightUpdates(CFeatureList *updates) = 0;
};

/// Eta Calculator, which allows you to set individual learning rates for each weight.
class CIndividualEtaCalculator : public CAdaptiveEtaCalculator
{
protected:
	int numWeights;
	double *etas;
public:
	CIndividualEtaCalculator(int numWeights, double *etas = NULL);
	virtual ~CIndividualEtaCalculator();

	/// Multiply the update gradient list with the fixed learning rates
	virtual void getWeightUpdates(CFeatureList *updates);

	/// Set Learning Rate for indexth weight
	virtual void setEta(int index, double value);
};

/// Vario Eta Learning Rate Calculator
/** 
Calculates the learning rate according to the weights standard deviation. Used by Coulum, more theoretical use to find out good fixed learning rates for a given function approximator.
Parameters of CVarioEta:
"VarioEtaLearningRate" : Base Learning Rate
"VarioEtaBeta" : Update factor
"VarioEtaEpsilon" :

For a more detailed description of the parameters see [Coulom, 1]
*/
class CVarioEta : public CAdaptiveEtaCalculator
{
protected:
	double *eta_i;
	double *v_i;

	/*double beta;
	double eta;
	double epsilon;*/
	unsigned int numParams;
public:
	CVarioEta(unsigned int numParams, double eta, double beta = 0.01, double epsilon = 0.0001);
	~CVarioEta();

	virtual void getWeightUpdates(CFeatureList *updates);
};

/// Interface class for all function which gradient update
/** 
Gradient update function only support updating the weights of the function, given the gradient. The datastructure of the gradient is always a feature list, because for RBF networks many parts of the gradient are zero. Additionally you can retrieve and set the weights directly with the functions getWeights(double *parameters) and setWeights(double *parameters). Gradient update functions don't implement the functions itself, so you can't calculate the output of the function given an input, you also can't calculate the gradient itself. This interface is just for the  update of the weights!
Each subclass of CGradientUpdateFunction has to implement the functoins:
- updateWeights(CFeatureList *gradient): Update the weights according to the gradient.
- getWeights(double *parameters), write all weights in the double array
- setWeights(double *parameters), set the weights according to the double array
- resetData(): reset all weights, needed when a new learning process is started
- getNumWeights(): return the number of weights.

When the weights are updated, the function updateGradient(CFeatureList *gradientFeatures, double factor = 1.0) is called. The gradient is at first saved in the local gradient buffer and then multiplied by the specified factor. If an eta calculator exists the eta calulator is applied to the gradient update, after that the user-interface function updateWeights is called.
Parameters of CGradientUpdateFunction:
The gradient update function inherits all Parameters from its eta calculator.
*/

class CGradientUpdateFunction : virtual public CParameterObject, virtual public CLearnDataObject
{
protected:
	CFeatureList *localGradientFeatureBuffer;

	
	CAdaptiveEtaCalculator *etaCalc;
public:
	CGradientUpdateFunction();
	virtual ~CGradientUpdateFunction();

	/// Does the preprocessing for the gradient update
	/** 
	When the weights are updated, the function updateGradient(CFeatureList *gradientFeatures, double factor = 1.0) is called. The gradient is at first saved in the local gradient buffer and then multiplied by the specified factor. If an eta calculator exists the eta calulator is applied to the gradient update, after that the user-interface function updateWeights is called.

	*/
	void updateGradient(CFeatureList *gradientFeatures, double factor = 1.0);

	/// Interface for updating the weights
	virtual void updateWeights(CFeatureList *dParams) = 0;

	///  Returns the number of weights.
	virtual int getNumWeights() = 0;

	/// get the eta calculator for this gradient update function
	virtual CAdaptiveEtaCalculator* getEtaCalculator();
	/// set the eta calculator for this gradient update function
	virtual void setEtaCalculator(CAdaptiveEtaCalculator *etaCalc);

	/// Function for getting all weights
	/** 
	The double array is assumed to be large enough. This isn't checked!
	*/
	virtual void getWeights(double *parameters) = 0;
	/// Function for setting all weights
	/** 
	The double array is assumed to be large enough. This isn't checked!
	*/
	virtual void setWeights(double *parameters) = 0;

	/// Save weights coming from getWeights
	virtual void saveData(FILE *stream);
	/// Load weights and set them with setWeights
	virtual void loadData(FILE *stream);

	/// Interface for resetting the weights
	virtual void resetData() = 0;

	virtual void copy(CLearnDataObject *gradientFuntion);
};

/*
/// Encapsulates the given gradientFunction and stores the weight updates
* the weight updates are transmitted to the original gradient function at every call of "updateOriginalGradientFunction", so the updates can be delayed to an arbitrary time.

class CGradientDelayedUpdateFunction : virtual public CGradientUpdateFunction
{
protected:
	CGradientUpdateFunction *gradientFunction;

	double *weightsUpdate;
	
	
public:
	virtual void updateWeights(CFeatureList *dParams);


	CGradientDelayedUpdateFunction(CGradientUpdateFunction *gradientFunction);
	virtual ~CGradientDelayedUpdateFunction();

	///  Returns the number of weights.
	virtual int getNumWeights();

	virtual void getWeights(double *parameters);
	virtual void setWeights(double *parameters);

	/// Interface for resetting the weights
	virtual void resetData();

	virtual void updateOriginalGradientFunction();
};

class CDelayedFunctionUpdater : public CSemiMDPListener
{
protected:
	int nUpdateEpisodes;
	int nUpdateSteps;

	CGradientDelayedUpdateFunction * updateFunction;

	int nEpisodes;
	int nSteps;
public:

	CDelayedFunctionUpdater(CGradientDelayedUpdateFunction * updateFunction, int nUpdateEpisodes, int nUpdateSteps);
	virtual ~CDelayedFunctionUpdater();

	virtual void newEpisode();
	virtual void nextStep(CStateCollection *oldState, CAction *action, CStateCollection *nextState);

};
*/

/// Interface for all functions which support gradient update and gradient and output calculation 
/** 
The class represents a n-input, m-output function approximator, which depends on numWeights weights. Gradient functions can always calculate the gradient with respect to the weights given an input and the m-dimensional output given an n-dimensional input .
This interface extends CGradientUpdateFunction, therefore all subclasses have to implement the same functions as for the super class. Due to the additional functionality of CGradientFunction the subclasses have to implement additionally to the functions from CGradientUpdateFunction following functions:
- getGradient(ColumnVector *input, ColumnVector *outputErrors, CFeatureList *gradientFeatures) : Calculate the gradient given the input and mthe output error (which is used for backpropagation). Save the gradient in the given feature list, which is supposed to be empty (but its recommended to be sure and clear it before use).
- getFunctionValue(ColumnVector *input, ColumnVector *output) : Calculate the m-dimensonal output given a n-dimensional input. The vectors are expected to have the correct size! 
- getNumInputs()
- getNumOutputs()

*/
class CGradientFunction : public CGradientUpdateFunction
{
protected:
	int num_inputs;
	int num_outputs;

	ColumnVector *input_mean;
	ColumnVector *input_std;
	
	ColumnVector *output_mean;
	ColumnVector *output_std;
	
	
	virtual void preprocessInput(ColumnVector *input, ColumnVector *norm_input);
	virtual void postprocessOutput(Matrix *norm_output, Matrix *output);
public:
	CGradientFunction(int n_input, int n_output);
	virtual ~CGradientFunction();

	virtual void getGradient(ColumnVector *input, ColumnVector *outputErrors, CFeatureList *gradientFeatures);
	/// Interface for calculating the output value
	virtual void getFunctionValue(ColumnVector *input, ColumnVector *output);

	/// Interface function for calculating the input gradient, only optional and not implemented by all classes
	virtual void getInputDerivation(ColumnVector *input, Matrix *targetVector);



	/// Interface for calculating the gradient given the input and the outputerror
	virtual void getGradientPre(ColumnVector *input, ColumnVector *outputErrors, CFeatureList *gradientFeatures) = 0;
	/// Interface for calculating the output value
	virtual void getFunctionValuePre(ColumnVector *input, ColumnVector *output) = 0;

	/// Interface function for calculating the input gradient, only optional and not implemented by all classes
	virtual void getInputDerivationPre(ColumnVector *, Matrix *) {};


	/// Return the dimension of the input
	virtual int getNumInputs();
	/// Return the dimension of the output
	virtual int getNumOutputs();

	void setInputMean(ColumnVector *input_mean);
	void setOutputMean(ColumnVector *output_mean);
	
	void setInputStd(ColumnVector *input_std);
	void setOutputStd(ColumnVector *output_std);
};


#endif

