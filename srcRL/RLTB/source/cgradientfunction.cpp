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
#include "cgradientfunction.h"
#include "cfeaturefunction.h"
#include "cutility.h"

#include <newmat/newmat.h>

#include <math.h>


CIndividualEtaCalculator::CIndividualEtaCalculator(int numWeights, double *l_etas)
{
	this->numWeights = numWeights;
	this->etas = new double[numWeights];

	if (l_etas != NULL)
	{
		memcpy(this->etas, l_etas, sizeof(double) * numWeights);
	}
	else
	{
		for (int i = 0; i < numWeights; i++)
		{
			etas[i] = 1.0;
		}
	}
}

CIndividualEtaCalculator::~CIndividualEtaCalculator()
{
	delete etas;
}


void CIndividualEtaCalculator::getWeightUpdates(CFeatureList *updates)
{
	CFeatureList::iterator it = updates->begin();
	for (; it != updates->end(); it ++)
	{
		(*it)->factor *= etas[(*it)->featureIndex];
	}
}

void CIndividualEtaCalculator::setEta(int index, double value)
{
	assert(index >= 0 && index < numWeights);
	etas[index] = value;
}

CVarioEta::CVarioEta(unsigned int numParams, double eta, double beta, double epsilon)
{
	/*this->beta = beta;
	this->epsilon = epsilon;
	this->eta = eta;*/
	addParameter("VarioEtaLearningRate", eta);
	addParameter("VarioEtaBeta", beta);
	addParameter("VarioEtaEpsilon",epsilon);

	this->numParams = numParams;

	eta_i = new double[numParams];
	v_i = new double[numParams];

	for (unsigned int i = 0; i < numParams; i++)
	{
		v_i[i] = 1;
		eta_i[i] = eta / (sqrt(v_i[i]) + epsilon);
	}
}

CVarioEta::~CVarioEta()
{
	delete eta_i;
	delete v_i;
}

void CVarioEta::getWeightUpdates(CFeatureList *Updates)
{
	double epsilon = getParameter("VarioEtaEpsilon");
	double beta = getParameter("VarioEtaBeta");
	double eta = getParameter("VarioEtaLearningRate");

	for (unsigned int i = 0; i < numParams; i ++)
	{
		eta_i[i] = eta / (sqrt(v_i[i]) + epsilon);
		v_i[i] *= (1 - beta); 
	}

	/*for (unsigned int i = 0; i < numParams; i ++)
	{
	v_i[i] = v_i[i] * (1 - beta);
	}*/

	CFeatureList::iterator it = Updates->begin();
	for (;it != Updates->end(); it ++)
	{
		v_i[(*it)->featureIndex] += beta * pow((*it)->factor / eta, 2);
		DebugPrint('v', "Vario Eta: Updating Feature %d with Eta %f (v_i: %f)\n", (*it)->featureIndex,eta_i[(*it)->featureIndex], v_i[(*it)->featureIndex]);
		(*it)->factor = (*it)->factor * eta_i[(*it)->featureIndex];
	}
}

CGradientUpdateFunction::CGradientUpdateFunction()
{
	localGradientFeatureBuffer = new CFeatureList();

	etaCalc = NULL;
}

CGradientUpdateFunction::~CGradientUpdateFunction()
{
	delete localGradientFeatureBuffer;
}


void CGradientUpdateFunction::updateGradient(CFeatureList *gradientFeatures, double factor)
{
	if (gradientFeatures != this->localGradientFeatureBuffer)
	{
		DebugPrint('g', "1...");
		
		this->localGradientFeatureBuffer->clear();

		CFeatureList::iterator it1 = gradientFeatures->begin();

		for (; it1 != gradientFeatures->end(); it1 ++)
		{
			this->localGradientFeatureBuffer->update((*it1)->featureIndex, factor * (*it1)->factor);
			
			DebugPrint('g', "%d : %f %f \n", (*it1)->featureIndex, (*it1)->factor, factor);
		}
		
		
		if (DebugIsEnabled('g'))
		{
			DebugPrint('g', "localGradientFeatures: ");
			localGradientFeatureBuffer->saveASCII(DebugGetFileHandle('g'));
		}
		
	}
	else
	{
		localGradientFeatureBuffer->multFactor(factor);
		DebugPrint('g', "2...");
	}

	if (etaCalc)
	{
		etaCalc->getWeightUpdates(this->localGradientFeatureBuffer);
		DebugPrint('g', "3...");
	}

	if (DebugIsEnabled('g'))
	{
		DebugPrint('g', "gradientFeatures: ");
		gradientFeatures->saveASCII(DebugGetFileHandle('g'));
		DebugPrint('g', "localGradientFeatures: ");
		localGradientFeatureBuffer->saveASCII(DebugGetFileHandle('g'));
	}
	
	updateWeights(this->localGradientFeatureBuffer);
}

CAdaptiveEtaCalculator* CGradientUpdateFunction::getEtaCalculator()
{
	return getEtaCalculator();	
}

void CGradientUpdateFunction::setEtaCalculator(CAdaptiveEtaCalculator *etaCalc)
{
	addParameters(etaCalc);
	this->etaCalc = etaCalc;
}


void CGradientUpdateFunction::copy(CLearnDataObject *l_gradientUpdateFunction)
{
	CGradientUpdateFunction *gradientUpdateFunction = dynamic_cast<CGradientUpdateFunction *>(l_gradientUpdateFunction);

	assert(gradientUpdateFunction->getNumWeights() == getNumWeights());	

	double *weights = new double[getNumWeights()];

	getWeights(weights);

	gradientUpdateFunction->setWeights( weights);	

	delete [] weights;
}


void CGradientUpdateFunction::saveData(FILE *stream)
{
	double *parameters = new double[getNumWeights()];
	getWeights(parameters);

	fprintf(stream, "Gradient Function\n");
	fprintf(stream, "Parameters: %d\n", getNumWeights());

	for (int i = 0; i < getNumWeights(); i ++)
	{
		fprintf(stream, "%f ", parameters[i]);
	}
	fprintf(stream, "\n");
	delete [] parameters;
}

void CGradientUpdateFunction::loadData(FILE *stream)
{
	double *parameters = new double[getNumWeights()];
	int bufNumParam;

	fscanf(stream, "Gradient Function\n");
	fscanf(stream, "Parameters: %d\n", &bufNumParam);

	assert(bufNumParam == getNumWeights());

	for (int i = 0; i < getNumWeights(); i ++)
	{
		fscanf(stream, "%lf ", &parameters[i]);
	}
	fscanf(stream, "\n");
	setWeights(parameters);

	delete [] parameters;
}

/*
CGradientDelayedUpdateFunction::CGradientDelayedUpdateFunction(CGradientUpdateFunction *gradientFunction)
{
	this->gradientFunction = gradientFunction;

	weightsUpdate = new double[gradientFunction->getNumWeights()];
}

CGradientDelayedUpdateFunction::~CGradientDelayedUpdateFunction()
{
	delete weightsUpdate;
}

void CGradientDelayedUpdateFunction::updateWeights(CFeatureList *dParams)
{
	CFeatureList::iterator it = dParams->begin();
	for (; it != dParams->end(); it ++)
	{
		weightsUpdate[(*it)->featureIndex] += (*it)->factor;
	}
}

int CGradientDelayedUpdateFunction::getNumWeights()
{
	return gradientFunction->getNumWeights();
}

void CGradientDelayedUpdateFunction::getWeights(double *parameters)
{
	return gradientFunction->getWeights(parameters);
}

void CGradientDelayedUpdateFunction::setWeights(double *parameters)
{
	gradientFunction->setWeights(parameters);
}

void CGradientDelayedUpdateFunction::resetData()
{
	memset(weightsUpdate, 0, sizeof(int) * getNumWeights());
}

void CGradientDelayedUpdateFunction::updateOriginalGradientFunction()
{
	gradientFunction->setWeights(weightsUpdate);
}


CDelayedFunctionUpdater::CDelayedFunctionUpdater(CGradientDelayedUpdateFunction * updateFunction, int nUpdateEpisodes, int nUpdateSteps)
{
	this->updateFunction = updateFunction;

	this->nUpdateEpisodes = nUpdateEpisodes;
	this->nUpdateSteps = nUpdateSteps;

	nEpisodes = 0;
	nSteps = 0;
}

CDelayedFunctionUpdater::~CDelayedFunctionUpdater()
{

}

void CDelayedFunctionUpdater::newEpisode()
{
	nEpisodes ++;
	if (nUpdateEpisodes > 0 && (nEpisodes % nUpdateEpisodes == 0))
	{
		updateFunction->updateOriginalGradientFunction();
	}
}

void CDelayedFunctionUpdater::nextStep(CStateCollection *oldState, CAction *action, CStateCollection *nextState)
{
	nSteps ++;
	if (nUpdateSteps > 0 && (nSteps % nUpdateSteps == 0))
	{
		updateFunction->updateOriginalGradientFunction();
	}
}
*/

CGradientFunction::CGradientFunction(int l_num_inputs, int l_num_outputs)
{
	num_inputs = l_num_inputs;
	num_outputs = l_num_outputs;
	
	input_mean = new ColumnVector(num_inputs);
	input_std = new ColumnVector(num_inputs);
	
	output_mean = new ColumnVector(num_outputs);
	output_std = new ColumnVector(num_outputs);
	
	*input_mean = 0;
	*output_mean = 0;
	
	*output_std = 1;
	*input_std = 1;
}

CGradientFunction::~CGradientFunction()
{
	delete input_mean;
	delete output_mean;
	delete input_std;
	delete output_std;
}

void CGradientFunction::preprocessInput(ColumnVector *input, ColumnVector *norm_input)
{
	*norm_input = (*input - *input_mean);
	
	for (int i = 0; i < num_inputs; i ++)
	{
		norm_input->element(i) = norm_input->element(i)  / input_std->element(i);
	}
	
}

void CGradientFunction::postprocessOutput(Matrix *norm_output, Matrix *output)
{
	for (int i = 0; i < num_outputs; i ++)
	{
		for (int j = 0; j < output->ncols(); j ++)
		{
			output->element(i, j) = (norm_output->element(i, j)  * output_std->element(i)) + output_mean->element(i);;
		}
	}

}


void CGradientFunction::getGradient(ColumnVector *input, ColumnVector *outputErrors, CFeatureList *gradientFeatures)
{
	ColumnVector norm_input(num_inputs);
	preprocessInput(input, &norm_input);
	getGradientPre(&norm_input, outputErrors, gradientFeatures);

}

void CGradientFunction::getFunctionValue(ColumnVector *input, ColumnVector *output)
{
	ColumnVector norm_input(num_inputs);
	preprocessInput(input, &norm_input);
	
//	printf("PreProc Input : ");
//	for (int i = 0; i < num_inputs; i ++)
//	{
//		printf("%f ", norm_input.element(i));
//	}
//	printf("\n");
	
	getFunctionValuePre(&norm_input, output);
	
//	printf("Output : ");
//	for (int i = 0; i < num_outputs; i ++)
//	{
//		printf("%f ", output->element(i));
//	}
//	printf("\n");
	
	postprocessOutput(output, output);
	
//	printf("PostProc Output : ");
//	for (int i = 0; i < num_outputs; i ++)
//	{
//		printf("%f ", output->element(i));
//	}
//	printf("\n");
}

void CGradientFunction::getInputDerivation(ColumnVector *input, Matrix *targetVector)
{
	ColumnVector norm_input(num_inputs);
	preprocessInput(input, &norm_input);
	getInputDerivation(&norm_input, targetVector);
	postprocessOutput(targetVector, targetVector);
}

void CGradientFunction::setInputMean(ColumnVector *l_input_mean)
{
	*input_mean = *l_input_mean;
}

void CGradientFunction::setOutputMean(ColumnVector *l_output_mean)
{
	*output_mean = *l_output_mean;
}
	
void CGradientFunction::setInputStd(ColumnVector *l_input_std)
{
	*input_std = *l_input_std;
}

void CGradientFunction::setOutputStd(ColumnVector *l_output_std)
{
	*output_std = *l_output_std;
}

int CGradientFunction::getNumInputs()
{
	return num_inputs;
}

int CGradientFunction::getNumOutputs()
{
	return num_outputs;
}


/*
CComposedGradientFunction::CComposedGradientFunction(CGradientFunction *gradientFunction1, CGradientFunction *gradientFunction2)
{
	this->gradientFunction1 = gradientFunction1;
	this->gradientFunction2 = gradientFunction2;
}

CComposedGradientFunction::~CComposedGradientFunction();
{

}

void CComposedGradientFunction::getGradient(ColumnVector *input, ColumnVector *outputErrors, CFeatureList *gradientFeatures)
{

	localGradientFeatureBuffer->clear();

	gradientFunction1->getFunctionValue(input, )
	gradientFunction1->getGradient()
}

void CComposedGradientFunction::getFunctionValue(ColumnVector *input, ColumnVector *output)
{
}

void CComposedGradientFunction::getInputDerivation(ColumnVector *input, Matrix *targetVector)
{
}

int CComposedGradientFunction::getNumInputs()
{
}

int CComposedGradientFunction::getNumOutputs()
{
}

void CComposedGradientFunction::updateWeights(CFeatureList *dParams)
{
}

int CComposedGradientFunction::getNumWeights()
{
}

void CComposedGradientFunction::getWeights(double *parameters)
{
}

void CComposedGradientFunction::setWeights(double *parameters)
{
}

void CComposedGradientFunction::resetData()
{
}*/
