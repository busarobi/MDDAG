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

#include "ril_debug.h"
#include "ctorchvfunction.h"
#include "cstatecollection.h"
#include "cstate.h"
#include "cstateproperties.h"
#include "caction.h"
#include "cvetraces.h"

#include <assert.h>
#include <math.h>

CTorchFunction::CTorchFunction(Machine *machine) 
{
	this->machine = machine;
}

CTorchFunction::~CTorchFunction()
{
}

double CTorchFunction::getValueFromMachine(Sequence *input)
{
	machine->forward(input);
	return (**machine->outputs->frames);
}

Machine *CTorchFunction::getMachine()
{
	return machine;
}


	/// Creates a new value function learning with a torch gradient machine	
CTorchGradientFunction::CTorchGradientFunction(GradientMachine *machine) : CTorchFunction(machine), CGradientFunction(machine->n_inputs, machine->n_outputs)
{
	localEtaCalc = NULL;

	input = new Sequence(1, machine->n_inputs);
	alpha = new Sequence(1, machine->n_outputs);	

	setGradientMachine(machine);
	
	//addParameter("InitWeightVarianceFactor", 1.0);
	//addParameter("TorchNormalizeWeights", 0.0);
}

CTorchGradientFunction::CTorchGradientFunction(int numInputs, int numOutputs) : CTorchFunction(NULL), CGradientFunction(numInputs, numOutputs)
{

	localEtaCalc = NULL;

	input = new Sequence(1, numInputs);
	alpha = new Sequence(1, numOutputs);	

	gradientMachine = NULL;
}

CTorchGradientFunction::~CTorchGradientFunction()
{
	delete alpha;
	delete input;

	delete localEtaCalc;
}

void CTorchGradientFunction::getFunctionValuePre(ColumnVector *inputVector, ColumnVector *output)
{
	if (gradientMachine == NULL)
	{
		*output = 0;
		return;
	}

	for (int i = 0; i < getNumInputs(); i++)
	{
		input->frames[0][i] = inputVector->element(i);
	}

	gradientMachine->forward(input);

	for (int i = 0; i < getNumOutputs(); i++)
	{
		output->element(i) = gradientMachine->outputs->frames[0][i];
	}
}

void CTorchGradientFunction::updateWeights(CFeatureList *gradientFeatures)
{
	if (gradientMachine == NULL)
	{
		return;
	}


	Parameters *params = (gradientMachine)->params;

	if (DebugIsEnabled('v'))
	{
		DebugPrint('v', "Updating Torch Function, Gradient: ");
		gradientFeatures->saveASCII(DebugGetFileHandle('v'));
		DebugPrint('v',"\n");
	}

	if(params)
	{
		CFeatureList::iterator it = gradientFeatures->begin();

		for(; it != gradientFeatures->end(); it++)
		{
			assert((*it)->featureIndex < (unsigned int) params->n_params);

			int param_index = 0;
			int param_offset = 0;

			while ((unsigned int) ((*it)->featureIndex - param_offset) > (unsigned int) (params->size[param_index]))
			{
				assert(param_index + 1 < params->n_data);
				param_offset += params->size[param_index];
				param_index += 1;
			}

			//DebugPrint('v', "\n Parameter: %d, Params Number: %d Params Size: %d\n", (*it)->featureIndex, param_index, params->size[param_index]);

			//DebugPrint('v', "(%f %f)", params->data[param_index][(*it)->featureIndex - param_offset], (*it)->factor);

			params->data[param_index][(*it)->featureIndex - param_offset] += (*it)->factor;

			//DebugPrint('v', "\n");
		}
	}
	/*if (getParameter("TorchNormalizeWeights") > 0.5)
	{
		for (int i = 0; i < params->n_data - 1; i ++)
		{
			double sum = 0;

			for (int j = 0; j < params->size[i]; j++)
			{
				sum += pow(params->data[i][j], 2);
			}
			sum = sqrt(sum);
			
			for (int j = 0; j < params->size[i]; j++)
			{
				params->data[i][j] /= 2.0;
			}
		}
	}*/
}

void CTorchGradientFunction::resetData()
{
	if (gradientMachine == NULL)
	{
		return;
	}

	Parameters *params = gradientMachine->params;

	int inputs = getNumInputs() + 1;
	DebugPrint('v', "Torch Gradient Function InitValues:\n");

	double sigma = 1.0;// getParameter("InitWeightVarianceFactor");

	if (params)
	{
		for(int i = 0; i < params->n_data; i++)
		{
			for(int j = 0; j < params->size[i]; j++)
			{
				params->data[i][j] = CDistributions::getNormalDistributionSample(0.0, (1.0 / inputs) * sigma);
				DebugPrint('v', "%f ", params->data[i][j]);

			}
			DebugPrint('v',  "\n");

			inputs = params->size[i] / inputs + 1;
		}
	}
}

void CTorchGradientFunction::setGradientMachine(GradientMachine *machine)
{
	this->gradientMachine = machine;

	if (localEtaCalc)
	{
		delete localEtaCalc;
		localEtaCalc = NULL;
	}

	if (gradientMachine)
	{
		localEtaCalc = new CTorchGradientEtaCalculator(gradientMachine);
	}

}

GradientMachine *CTorchGradientFunction::getGradientMachine()
{
	return gradientMachine;
}

void CTorchGradientFunction::getGradientPre(ColumnVector *inputVector, ColumnVector *outputErrors, CFeatureList *gradientFeatures)
{
	if (gradientMachine == NULL)
	{
		return;
	}


	Parameters *params = (gradientMachine)->params;
	Parameters *der_params = (gradientMachine)->der_params;

	for (int i = 0; i < getNumInputs(); i ++)
	{
		input->frames[0][i] = inputVector->element(i);
	}

	for (int i = 0; i < getNumOutputs(); i ++)
	{
		alpha->frames[0][i] = outputErrors->element(i);
	}

	gradientMachine->iterInitialize();
	if (der_params)
	{
		for(int i = 0; i < der_params->n_data; i++)
		{
			for(int j = 0; j < params->size[i]; j++)
			{
				der_params->data[i][j] = 0.0; 
			}
		}
		//memset(der_params->data[i], 0, sizeof(real)*der_params->size[i]);
	}

	gradientMachine->forward(input);
	gradientMachine->backward(input, alpha);

	DebugPrint('v', "\n Getting Torch Gradient Params Size: %d\n", getNumWeights());

	if(params)
	{
		int param_offset = 0;
		for(int i = 0; i < params->n_data; i++)
		{
			real *ptr_der_params = der_params->data[i];
			real *ptr_params = params->data[i];

			for(int j = 0; j < params->size[i]; j++)
			{
				gradientFeatures->set(param_offset + j, ptr_der_params[j]);
				DebugPrint('v', "%f (%f)", ptr_der_params[j], ptr_params[j]);
			}
			DebugPrint('v',  "\n");

		
			param_offset += params->size[i];
		}
	}
}

int CTorchGradientFunction::getNumWeights()
{
	if (gradientMachine == NULL)
	{
		return 0;
	}

	return gradientMachine->params->n_params;
}

void CTorchGradientFunction::getInputDerivationPre(ColumnVector *inputVector, Matrix *targetMatrix)
{
	if (gradientMachine == NULL)
	{
		return;
	}

	*targetMatrix = 0;

	Sequence *beta = (gradientMachine)->beta;

	for (int i = 0; i < getNumInputs(); i ++)
	{
		input->frames[0][i] = inputVector->element(i);
	}


	for (int nout = 0; nout < getNumOutputs(); nout ++)
	{
		for (int i = 0; i < getNumOutputs(); i ++)
		{
			alpha->frames[0][i] = 0.0;
		}

		alpha->frames[0][nout] = 1.0;

		gradientMachine->iterInitialize();

		gradientMachine->forward(input);
		gradientMachine->backward(input, alpha);

		if(beta)
		{
			for (int i = 0; i < getNumInputs(); i++)
			{
				targetMatrix->element(nout, i)  = beta->frames[0][i];
			}
		}
	}
	
}

void CTorchGradientFunction::getWeights(double *parameters)
{
	if (gradientMachine == NULL)
	{
		return;
	}

	Parameters *params = (gradientMachine)->params;

	if(params)
	{
		int paramIndex = 0;
		for(int i = 0; i < params->n_data; i++)
		{
			real *ptr_params = params->data[i];

			for(int j = 0; j < params->size[i]; j++)
			{
				parameters[paramIndex] = ptr_params[j];
				paramIndex ++;
			}
		}
	}
}

void CTorchGradientFunction::setWeights(double *parameters)
{
	if (gradientMachine == NULL)
	{
		return;
	}

	Parameters *params = (gradientMachine)->params;

	if(params)
	{
		int paramIndex = 0;
		for(int i = 0; i < params->n_data; i++)
		{
			real *ptr_params = params->data[i];

			for(int j = 0; j < params->size[i]; j++)
			{
				ptr_params[j] = parameters[paramIndex]; 
				paramIndex ++;
			}
		}
	}
	if (DebugIsEnabled('t'))
	{
		DebugPrint('t', "Setting Torch Weights: ");

		saveData(DebugGetFileHandle('t'));
	}
}

CTorchGradientEtaCalculator::CTorchGradientEtaCalculator(GradientMachine *gradientMachine) : CIndividualEtaCalculator(gradientMachine->params->n_params)
{
	Parameters *params = gradientMachine->params;

	int inputs = gradientMachine->n_inputs + 1;
	int neurons = 1;
	int parameterIndex = 0;
	double factor = 1.0;
	if (params)
	{
		for(int i = 0; i < params->n_data; i++)
		{
			for(int j = 0; j < params->size[i]; j++)
			{
				this->etas[parameterIndex] = factor;
				parameterIndex ++;
			}
			inputs = params->size[i] / inputs + 1;
			neurons = inputs - 1;
			factor = 1 / sqrt((double) neurons);
		}
	}
}

CTorchVFunction::CTorchVFunction(CTorchFunction *torchFunction, CStateProperties *properties) : CAbstractVFunction(properties)
{
	input = new Sequence(1, properties->getNumContinuousStates() + properties->getNumDiscreteStates());

	this->torchFunction = torchFunction;
}

CTorchVFunction::~CTorchVFunction()
{
	delete input;
}

void CTorchVFunction::getInputSequence(CState *state, Sequence *sequence)
{
	for (unsigned int i = 0; i < state->getNumActiveDiscreteStates(); i ++)
	{
		sequence->frames[0][i] = state->getContinuousState(i);
	}
	for (unsigned int i = 0; i < state->getNumActiveDiscreteStates(); i++)
	{
		sequence->frames[0][i + state->getNumContinuousStates()] = state->getDiscreteState(i);
	}
}

double CTorchVFunction::getValue(CState *state)
{
	getInputSequence(state, input);
	double value = torchFunction->getValueFromMachine(input);
	
	state->saveASCII( stdout);
	
	if (!mayDiverge && (value < - DIVERGENTVFUNCTIONVALUE || value > DIVERGENTVFUNCTIONVALUE))
	{
		throw new CDivergentVFunctionException("Torch VFunction", this, state, value);
	}
	
	return value;
}

CVFunctionFromGradientFunction::CVFunctionFromGradientFunction(CGradientFunction *l_gradientFunction, CStateProperties *properties) : CGradientVFunction(properties) , CVFunctionInputDerivationCalculator(properties)
{
	this->gradientFunction = l_gradientFunction;

	printf("%d %d %d %d\n", properties->getNumContinuousStates(), properties->getNumDiscreteStates(), gradientFunction->getNumInputs(), gradientFunction->getNumOutputs());
	assert(properties->getNumContinuousStates() + properties->getNumDiscreteStates() == (unsigned int) gradientFunction->getNumInputs() && gradientFunction->getNumOutputs() == 1);

	input = new ColumnVector(properties->getNumContinuousStates() + properties->getNumDiscreteStates());
	outputError = new ColumnVector(1);
	outputError->element(0) = 1.0;

	this->inputDerivation = new Matrix(1, properties->getNumContinuousStates() + properties->getNumDiscreteStates());

	addParameters(l_gradientFunction);
}

CVFunctionFromGradientFunction::~CVFunctionFromGradientFunction()
{
	delete input;
	delete outputError;
	delete inputDerivation;
}

void CVFunctionFromGradientFunction::getInputSequence(CState *state, ColumnVector *sequence)
{
	for (unsigned int i = 0; i < state->getNumActiveContinuousStates(); i ++)
	{
		sequence->element(i) = state->getContinuousState(i);
	}
	for (unsigned int i = 0; i < state->getNumActiveDiscreteStates(); i++)
	{
		sequence->element(i + state->getNumContinuousStates()) = state->getContinuousState(i);
	}
}

void CVFunctionFromGradientFunction::setValue(CState *state, double value)
{
	updateValue(state, value - getValue(state));
}

void CVFunctionFromGradientFunction::resetData()
{
	gradientFunction->resetData();
}


double CVFunctionFromGradientFunction::getValue(CState *state)
{
	getInputSequence(state, input);
	
	gradientFunction->getFunctionValue(input, outputError);

	double value = outputError->element(0);

	if (!mayDiverge && (value < - DIVERGENTVFUNCTIONVALUE || value > DIVERGENTVFUNCTIONVALUE))
	{
		throw new CDivergentVFunctionException("Torch VFunction", this, state, value);
	}

	return value;
}
	
void CVFunctionFromGradientFunction::updateWeights(CFeatureList *gradientFeatures)
{
	gradientFunction->updateWeights(gradientFeatures);
}

int CVFunctionFromGradientFunction::getNumWeights()
{
	return gradientFunction->getNumWeights();
}

void CVFunctionFromGradientFunction::getGradient(CStateCollection *originalState, CFeatureList *modifiedState)
{
	CState *state = originalState->getState(this->getStateProperties());

	getInputSequence(state, input);
	outputError->element(0) = 1.0;

	gradientFunction->getGradient(input, outputError, modifiedState);
}

void CVFunctionFromGradientFunction::getInputDerivation(CStateCollection *originalState, ColumnVector *targetVector)
{
	CState *state = originalState->getState(this->getStateProperties());

	getInputSequence(state, input);

	gradientFunction->getInputDerivation(input, targetVector);

//	memcpy(targetVector->getData(), inputDerivation->getRow(0), sizeof(double) * gradientFunction->getNumInputs());
}

CAbstractVETraces *CVFunctionFromGradientFunction::getStandardETraces()
{
	return new CGradientVETraces(this);
}

void CVFunctionFromGradientFunction::getWeights(double *parameters)
{
	gradientFunction->getWeights(parameters);
}

void CVFunctionFromGradientFunction::setWeights(double *parameters)
{
	gradientFunction->setWeights(parameters);
}

CQFunctionFromGradientFunction::CQFunctionFromGradientFunction(CContinuousAction *contAction, CGradientFunction *gradientFunction, CActionSet *actions, CStateProperties *properties) : CContinuousActionQFunction(contAction), CStateObject(properties)
{
	assert(properties->getNumContinuousStates() + properties->getNumDiscreteStates() + contAction->getNumDimensions() == (unsigned int) gradientFunction->getNumInputs() && gradientFunction->getNumOutputs() == 1);

	input = new ColumnVector(properties->getNumContinuousStates() + properties->getNumDiscreteStates() + contAction->getNumDimensions());
	outputError = new ColumnVector(1);
	outputError->element(0) = 1.0;

	this->gradientFunction = gradientFunction;

	staticActions = actions;
}


CQFunctionFromGradientFunction::~CQFunctionFromGradientFunction()
{
	delete input;
	delete outputError;
}

void CQFunctionFromGradientFunction::getInputSequence(ColumnVector *sequence, CState *state, CContinuousActionData *data)
{
	for (unsigned int i = 0; i < state->getNumContinuousStates(); i ++)
	{
		sequence->element(i ) = state->getContinuousState(i);
	}
	for (unsigned int i = 0; i < state->getNumDiscreteStates(); i++)
	{
		sequence->element(i + state->getNumContinuousStates()) = state->getDiscreteState(i);
	}
	for (int i = 0; i < data->nrows(); i++)
	{
		double min =  contAction->getContinuousActionProperties()->getMinActionValue(i);
		double width = contAction->getContinuousActionProperties()->getMaxActionValue(i) - min;

		sequence->element(i + state->getNumContinuousStates() + state->getNumDiscreteStates()) = ((data->getActionValue(i) - min) / width) * 2  - 1.0;
	}
}

void CQFunctionFromGradientFunction::getBestContinuousAction(CStateCollection *state, CContinuousActionData *actionData)
{
	CAction *staticAction = CAbstractQFunction::getMax(state, staticActions);
	actionData->setData(staticAction->getActionData());
}

void CQFunctionFromGradientFunction::updateCAValue(CStateCollection *state, CContinuousActionData *data, double td)
{
	this->localGradientFeatureBuffer->clear();

	getCAGradient(state, data, localGradientFeatureBuffer);

	updateGradient(localGradientFeatureBuffer, td);
}

void CQFunctionFromGradientFunction::setCAValue(CStateCollection *state, CContinuousActionData *data, double qValue)
{
	updateCAValue(state, data, qValue - getCAValue(state, data));
}

double CQFunctionFromGradientFunction::getCAValue(CStateCollection *state, CContinuousActionData *data)
{
	getInputSequence(input, state->getState(properties), data);
	
	gradientFunction->getFunctionValue(input, outputError);

	return outputError->element(0);
}


void CQFunctionFromGradientFunction::getCAGradient(CStateCollection *state, CContinuousActionData *data, CFeatureList *gradient)
{
	getInputSequence(input, state->getState(properties), data);
	outputError->element(0) = 1.0;

	gradientFunction->getGradient(input, outputError, gradient);
}

void CQFunctionFromGradientFunction::updateWeights(CFeatureList *gradientFeatures)
{
	gradientFunction->updateWeights(gradientFeatures);
}

int CQFunctionFromGradientFunction::getNumWeights()
{
	return gradientFunction->getNumWeights();
}

void CQFunctionFromGradientFunction::resetData()
{
	gradientFunction->resetData();
}


void CQFunctionFromGradientFunction::getWeights(double *weights)
{
	gradientFunction->getWeights(weights);
}

void CQFunctionFromGradientFunction::setWeights(double *parameters)
{
	gradientFunction->setWeights(parameters);
}
