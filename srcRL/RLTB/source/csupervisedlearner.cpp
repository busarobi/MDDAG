
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
#include "csupervisedlearner.h"

#include <newmat/newmat.h>
#include "cutility.h"
#include "ctorchvfunction.h"
#include "cgradientfunction.h"
#include "cinputdata.h"
#include "cfeaturefunction.h"

#include "ConnectedMachine.h"
#include "MLP.h"
#include "StochasticGradient.h"
#include "WeightedMSECriterion.h"
#include "MSECriterion.h"
#include "MemoryDataSet.h"
#include "DataSet.h"

using  namespace Torch;

using Torch::MLP;

CLeastSquaresLearner::CLeastSquaresLearner(CGradientUpdateFunction *l_featureFunc, int numData)
{
	this->featureFunc = l_featureFunc;
		
	int numFeatures = featureFunc->getNumWeights();
	A = new Matrix(numData, numFeatures); 
	b = new ColumnVector(numData);
	
	*A = 0;
	*b = 0;
	
	A_pinv = new Matrix(numFeatures, numData);
	
	*A_pinv = 0;
	
	addParameter("SVD_Damping", 0.01);

}

CLeastSquaresLearner::~CLeastSquaresLearner()
{
	delete A;
	delete b;
	delete A_pinv;
}

double CLeastSquaresLearner::doOptimization()
{
	int numFeatures = featureFunc->getNumWeights();
	ColumnVector w(numFeatures);
	double error = doOptimization(A, A_pinv, b, &w, getParameter("SVD_Damping"));
	
	double weights[numFeatures];
	
	for (int i = 0; i < numFeatures; i ++)
	{
		weights[i] = w.element(i);
	}
	featureFunc->setWeights(weights);
	
	return error;
}
		
double CLeastSquaresLearner::doOptimization(Matrix *A, Matrix *A_pinv, ColumnVector *b, ColumnVector *w, double lambda)
{
	int numFeatures = w->nrows();
//	double lambda = getParameter("SVD_Damping");

	assert(A->ncols() == numFeatures && A->nrows() == b->nrows());

	getPseudoInverse(A, A_pinv, lambda);
	
	(*w) = (*A_pinv) * (*b);
	
	
	double error = 0;
	
	ColumnVector b_error(b->nrows());
	b_error = (*A) * (*w);
	b_error = b_error - *b;
	
	error = b_error.norm_Frobenius() / b_error.nrows();
	
	return error;
}


CSupervisedGradientCalculator::CSupervisedGradientCalculator(CGradientFunction *l_gradientFunction, CDataSet *l_inputData, CDataSet *l_outputData)
{
	this->inputData = l_inputData;
	this->outputData = l_outputData;

	outputData1D = NULL;	

	this->gradientFunction = l_gradientFunction;
	
	//assert(gradientFunction->getNumInputs() == inputData->getNumDimensions());
	//assert(gradientFunction->getNumOutputs() == inputData->nrows());
}

CSupervisedGradientCalculator::~CSupervisedGradientCalculator()
{
}
		
void CSupervisedGradientCalculator::getGradient(CFeatureList *gradient)
{
	ColumnVector *input;

	ColumnVector output(gradientFunction->getNumOutputs());
	
	gradient->clear();
	
	for (unsigned int i = 0; i < inputData->size(); i++)
	{
		input = (*inputData)[i];
		
		gradientFunction->getFunctionValue(input, &output);
		
		if (outputData1D != NULL)
		{
			output.element(0) = output.element(0)  - (*outputData1D)[i];
		}
		else
		{
			output = output  - *(*outputData)[i];
		}

    		gradientFunction->getGradient(input, &output, gradient);
	}
	gradient->multFactor( 1.0 / inputData->size());
}

double CSupervisedGradientCalculator::getFunctionValue()
{
	double squaredE = 0;

	ColumnVector *input;

	ColumnVector output(gradientFunction->getNumOutputs());
	
	for (unsigned int i = 0; i < inputData->size(); i++)
	{
		input = (*inputData)[i];
		
		gradientFunction->getFunctionValue(input, &output);
		
		if (outputData1D != NULL)
		{
			output.element(0) = output.element(0)  - (*outputData1D)[i ];
		}
		else
		{
			output = output  - *(*outputData)[i];
		}

    		squaredE += dotproduct(output, output); 		
	}
	return squaredE / inputData->size() / 2;
}

CSupervisedFeatureGradientCalculator::CSupervisedFeatureGradientCalculator(CFeatureFunction *l_featureFunction) : CSupervisedGradientCalculator(NULL, NULL, NULL)
{
	featureFunction = l_featureFunction;

	featureList = new CFeatureList();
}

CSupervisedFeatureGradientCalculator::~CSupervisedFeatureGradientCalculator()
{
	delete featureList;
}

CFeatureList *CSupervisedFeatureGradientCalculator::getFeatureList(ColumnVector *input)
{
	featureList->clear();
	for (int i = 0; i < input->nrows() / 2; i ++)
	{
		featureList->set((int) input->element(i), input->element(i +  input->nrows() / 2)); 
	}
	return featureList;
}


void CSupervisedFeatureGradientCalculator::getGradient(CFeatureList *gradient)
{
	ColumnVector *input;
	
	gradient->clear();
	
	for (unsigned int i = 0; i < inputData->size(); i++)
	{
		input = (*inputData)[i];
	
		CFeatureList *featureList = getFeatureList(input);
	
		double output = featureFunction->getFeatureList(featureList);
		double error = 0;
		if (outputData1D != NULL)
		{
			error = output - (*outputData1D)[i];
		}
		else
		{
			assert(false);
		}
		gradient->add(featureList, error);
	}
	gradient->multFactor( 1.0 / inputData->size());
}

double CSupervisedFeatureGradientCalculator::getFunctionValue()
{

	ColumnVector *input;
	
	double squaredE = 0;
	for (unsigned int i = 0; i < inputData->size(); i++)
	{
		input = (*inputData)[i];
	
		CFeatureList *featureList = getFeatureList(input);
	
		double output = featureFunction->getFeatureList(featureList);
		double error = 0;
		if (outputData1D != NULL)
		{
			error = output - (*outputData1D)[i];
		}
		else
		{
			assert(false);
		}
		squaredE += pow(error, 2.0);
	}
	return squaredE / 2.0 / inputData->size();
}



void CSupervisedGradientCalculator::setData(CDataSet *l_inputData, CDataSet1D *l_outputData1D)
{
	inputData = l_inputData;
	outputData =  NULL;
	outputData1D = l_outputData1D;
}

void CSupervisedGradientCalculator::setData(CDataSet *l_inputData, CDataSet *l_outputData)
{
	inputData = l_inputData;
	outputData =  l_outputData;
	outputData1D = NULL;
}

CSupervisedGradientLearner::CSupervisedGradientLearner(CGradientLearner *l_gradientLearner, CSupervisedGradientCalculator *l_gradientCalculator, int episodes)
{
	addParameter("SupervisedGradientLearnEpisodes", episodes);

	gradientLearner = l_gradientLearner;
	gradientCalculator = l_gradientCalculator;
}

CSupervisedGradientLearner::~CSupervisedGradientLearner()
{
}

void CSupervisedGradientLearner::resetLearner()
{
	gradientLearner->resetOptimization();
}
		

void CSupervisedGradientLearner::learnFA(CDataSet *inputData, CDataSet1D *outputData)
{	
	gradientCalculator->setData( inputData, outputData);
	

	int numEpisodes = (int) getParameter("SupervisedGradientLearnEpisodes");
	gradientLearner->doOptimization(numEpisodes);	 
}

CSupervisedQFunctionLearnerFromLearners::CSupervisedQFunctionLearnerFromLearners(std::map<CAction *, CSupervisedLearner *> *l_learnerMap)
{
	learnerMap = l_learnerMap;
	std::map<CAction *, CSupervisedLearner *>::iterator it = learnerMap->begin();
	for (; it != learnerMap->end(); it ++)
	{
		addParameters((*it).second);
	}
}

CSupervisedQFunctionLearnerFromLearners::~CSupervisedQFunctionLearnerFromLearners()
{
	
}


void CSupervisedQFunctionLearnerFromLearners::resetLearner()
{
	std::map<CAction *, CSupervisedLearner *>::iterator it = learnerMap->begin();
	for (; it != learnerMap->end(); it ++)
	{
		(*it).second->resetLearner();
	}
}

void CSupervisedQFunctionLearnerFromLearners::learnQFunction(CAction *action, CDataSet *inputData, CDataSet1D *outputData)
{
	(*learnerMap)[action]->learnFA(inputData, outputData);
}

CSupervisedQFunctionWeightedLearnerFromLearners::CSupervisedQFunctionWeightedLearnerFromLearners(std::map<CAction *, CSupervisedWeightedLearner *> *l_learnerMap)
{
	learnerMap = l_learnerMap;
	std::map<CAction *, CSupervisedWeightedLearner *>::iterator it = learnerMap->begin();
	for (; it != learnerMap->end(); it ++)
	{
		addParameters((*it).second);
	}
}

CSupervisedQFunctionWeightedLearnerFromLearners::~CSupervisedQFunctionWeightedLearnerFromLearners()
{
	
}


void CSupervisedQFunctionWeightedLearnerFromLearners::resetLearner()
{
	std::map<CAction *, CSupervisedWeightedLearner *>::iterator it = learnerMap->begin();
	for (; it != learnerMap->end(); it ++)
	{
		(*it).second->resetLearner();
	}
}

void CSupervisedQFunctionWeightedLearnerFromLearners::learnQFunction(CAction *action, CDataSet *inputData, CDataSet1D *outputData, CDataSet1D *weightData)
{
	(*learnerMap)[action]->learnWeightedFA(inputData, outputData, weightData);
}

CGradientFunctionUpdater::CGradientFunctionUpdater(CGradientUpdateFunction *updateFunction)
{
	this->updateFunction = updateFunction;
}

void CGradientFunctionUpdater::addRandomParams(double randSize)
{
	double *weights = new double[updateFunction->getNumWeights()];
	updateFunction->getWeights(weights);

	double normWeights = 0;
	for (int i = 0; i < updateFunction->getNumWeights(); i++)
	{
		normWeights += pow(weights[i], 2);
	}
	normWeights = sqrt(normWeights);

	for (int i = 0; i <updateFunction->getNumWeights(); i ++)
	{
		weights[i] += CDistributions::getNormalDistributionSample(0, normWeights * randSize / 2);
	}
	updateFunction->setWeights(weights);
	delete weights;
}

CConstantGradientFunctionUpdater::CConstantGradientFunctionUpdater(CGradientUpdateFunction *updateFunction, double learningRate) : CGradientFunctionUpdater(updateFunction)
{
	addParameter("GradientLearningRate", learningRate);
}

void CConstantGradientFunctionUpdater::updateWeights(CFeatureList *gradient)
{
	updateFunction->updateGradient(gradient, getParameter("GradientLearningRate"));
}

void CLineSearchGradientFunctionUpdater::setWorkingParamters(CFeatureList *gradient, double stepSize, double *startParameters, double *workParameters)
{
	DebugPrint('l', "Applying StepSize: %f\n", stepSize);

	memcpy(workParameters, startParameters, sizeof(double) * updateFunction->getNumWeights());

	CFeatureList::iterator it = gradient->begin();
	for (; it != gradient->end(); it ++)
	{
		workParameters[(*it)->featureIndex] += stepSize * (*it)->factor;
	}
}

CLineSearchGradientFunctionUpdater::CLineSearchGradientFunctionUpdater(CGradientCalculator *l_gradientCalculator, CGradientUpdateFunction *updateFunction, int maxSteps) : CGradientFunctionUpdater(updateFunction)
{
	gradientCalculator = l_gradientCalculator;
	
	startParameters = new double[updateFunction->getNumWeights()];
	workParameters = new double[updateFunction->getNumWeights()];

	this->maxSteps = maxSteps;

	addParameter("LineSearchStepSizeScale", 1.0);
	precision_treshold = 0.0001;
}

CLineSearchGradientFunctionUpdater::~CLineSearchGradientFunctionUpdater()
{
	delete startParameters;
	delete workParameters;
}

double CLineSearchGradientFunctionUpdater::getFunctionValue(double *startParameters, CFeatureList *gradient, double stepSize)
{
	setWorkingParamters(gradient, stepSize, startParameters, workParameters);
	updateFunction->setWeights(workParameters);
		
	double value = gradientCalculator->getFunctionValue();
	return value;
}

void CLineSearchGradientFunctionUpdater::bracketMinimum(double *startParameters, CFeatureList *gradient, double fa, double &a, double &b, double &c)
{
	// Value of golden section (1 + sqrt(5))/2.0
	const double phi = 1.6180339887;

	// Initialise count of number of function evaluations
	int num_evals = 0;

	// A small non-zero number to avoid dividing by zero in quadratic interpolation
	const double TINY = 1.e-10;

	double max_step = 10.0;

	double fb = getFunctionValue(startParameters, gradient, b);
		
	num_evals ++;

    bool bracket_found = false;
	// Assume that we know going from a to b is downhill initially 
	// (usually because gradf(a) < 0).
	
	double fu = 0;
	double fc = 0;
	double u = 0;
	
	if (fb > fa)
  	{
  		// Minimum must lie between a and b: do golden section until we find point
  		// low enough to be middle of bracket
	    do
	    {
		    c = b;
			b = a + (c-a)/phi;
			fb = getFunctionValue(startParameters, gradient, b);
			num_evals ++;
		    printf("Bracketing1: (%f %f) (%f, %f) (%f %f)\n", a, fa, b, fb ,c, fc);
	    }
		while (fb > fa && b > a + TINY);
  	}
	else  
	{
		  // There is a valid bracket upper bound greater than b
		  c = b + phi*(b-a);
		  fc = getFunctionValue(startParameters, gradient, c);
		  num_evals = num_evals + 1;

  		  while (fb > fc)
		  {
		  	//Do a quadratic interpolation (i.e. to minimum of quadratic)
		    double r = (b-a)*(fb-fc);
    		double q = (b-c)*(fb-fa);
    		double sign_qr = 1.0;
    		if (q - r < 0)
    		{
    			sign_qr = -1.0;
    		} 
    		
    		
		    u = b - ((b-c)*q - (b-a)*r) / (2.0*(sign_qr*max(fabs(q-r), TINY)));
		    double ulimit = b + max_step*(c-b);
            if ((b-u)*(u-c) > 0.0)
            {
		      // Interpolant lies between b and c
		      fu = getFunctionValue(startParameters, gradient, u);
		      num_evals = num_evals + 1;
		      if (fu < fc)
			  {
				 //Have a minimum between b and c
			     a = b;
				 b = u;
			 	 c = c;
				 return;
			  }
		      else
		      {
		      	if (fu > fb)
		      	{
					// Have a minimum between a and u
					a = a;
					b = c;
					c = u;
					return;
		      	}
		      }
		      // Quadratic interpolation didn't give a bracket, so take a golden step
			  u = c + phi*(c-b);
            }
			else
			{
				if ((c-u)*(u-ulimit) > 0.0)
				{
					// Interpolant lies between c and limit
				    fu = getFunctionValue(startParameters, gradient, u);
				    num_evals = num_evals + 1;
				    if (fu < fc)
				    {
						// Move bracket along, and then take a golden section step
						b = c;
						c = u;
						u = c + phi*(c-b);
				    }
				    else
				    {
						bracket_found = 1;
				    }
				}
			    else
			    {
			    	if ((u-ulimit)*(ulimit-c) >= 0.0)
			    	{
				      // Limit parabolic u to maximum value
				      u = ulimit;
			    	}
				    else
				    {
				      // Reject parabolic u and use golden section step
				      u = c + phi*(c-b);
				    }
			    }
			}
    		if (!bracket_found)
    		{
			   fu = getFunctionValue(startParameters, gradient, u);
			   printf("1\n");
			   num_evals = num_evals + 1;
    		}
			a = b; b = c; c = u;
		    fa = fb; fb = fc; fc = fu;
		    
		    printf("Bracketing: (%f, %f) (%f, %f) (%f, %f)\n", a, fa, b, fb ,c, fc);
		  
		} // while loop
	} // bracket found

    printf("Bracketing finished: (%e %f) (%e, %f) (%e %f)\n", a, fa, b, fb ,c, fc);
	if (a > c)
	{
	  double temp = c;
	  c = a;
	  a = temp;
	}
}

void CLineSearchGradientFunctionUpdater::updateWeights(CFeatureList *gradient)
{
	updateFunction->getWeights(startParameters);
	
	double fpt = getFunctionValue(startParameters, gradient, 0);
	double lmin = 0.0;
	updateWeights(gradient, fpt, lmin);
}


double CLineSearchGradientFunctionUpdater::updateWeights(CFeatureList *gradient, double fpt, double &lmin)
{
//	int maxIndex = 0;
//	double maxValue = 0.0;
//	double maxLearnRate = 0.0;
//
//	double *values = new double[numStepSizes];
//	double searchValues[3];
//	double searchStepSizes[3];
//	
//	updateFunction->getWeights(startParameters);
//
//	printf("Searching in Gradient Direction, %d start points\n", numStepSizes);
//
//	//evaluator->getNewStartStates();
//
//	int i = 0;
//
//	DebugPrint('l', "Beginning Line Search\n");
//	DebugPrint('l', "Gradient: ");
//
//	if (DebugIsEnabled('l'))
//	{
//		gradient->saveASCII(DebugGetFileHandle('l'));
//		DebugPrint('l', "Gradient Norm: %f\n", gradient->multFeatureList(gradient));
//		DebugPrint('l', "\n");
//	}
//
//	for (i = 0; i < numStepSizes; i++)
//	{
//		setWorkingParamters(gradient, startStepSizes[i] * getParameter("LineSearchStepSizeScale"), startParameters, workParameters);
//		updateFunction->setWeights(workParameters);
//		
//		//double newValue = 0.0;
//		try
//		{
//			values[i] = evaluator->evaluatePolicy();
//		}
//		catch (CMyException *E) 
//		{
//			values[i] = - 100000000;
//		}
//		printf("StepSize %f : %f\n", startStepSizes[i] * getParameter("LineSearchStepSizeScale"), values[i]);
//		DebugPrint('l', "Finished Evaluation of StepSize %f : Value %f\n", startStepSizes[i], values[i]);
//
//
//		if (i == 0 || values[i] > maxValue + (fabs(maxValue) * 0.0001))
//		{
//			maxIndex = i;
//			maxValue = values[i];
//			maxLearnRate = startStepSizes[i] * getParameter("LineSearchStepSizeScale");
//			printf("Found New Maximum\n");
//		}
//	}
//	if (i < maxSteps)
//	{
//		if (maxIndex == 0 || maxIndex == numStepSizes - 1)
//		{
//			maxIndex ++;
//			printf("Maximum outside the start step intervall, not searching further\n");
//		}
//		else
//		{
//			for (int j = 0; j < 3; j ++)
//			{
//				searchValues[j] = values[maxIndex + j - 1];
//				searchStepSizes[j] = startStepSizes[maxIndex + j - 1] * getParameter("LineSearchStepSizeScale");
//			}
//			while (i < maxSteps)
//			{
//				i ++;
//				if (searchValues[0] / (searchStepSizes[1] - searchStepSizes[0]) > searchValues[2] / (searchStepSizes[2] - searchStepSizes[1]))
//				{
//					double newStepSize = (searchStepSizes[0] + searchStepSizes[1]) * 0.5;
//
//					setWorkingParamters(gradient, newStepSize, startParameters, workParameters);
//					updateFunction->setWeights(workParameters);
//
//					double newValue = 0.0;
//					try
//					{
//						newValue = evaluator->evaluatePolicy();
//					}
//					catch (CMyException *E) 
//					{
//						newValue = - 100000000;
//					}
//					
//					printf("StepSize %f : %f\n", newStepSize, newValue);
//					DebugPrint('l', "Finished Evaluation of StepSize %f : Value %f\n", newStepSize, newValue);
//
//					if (newValue > searchValues[1])
//					{
//						searchValues[2] = searchValues[1];
//						searchValues[1] = newValue;
//
//						searchStepSizes[2] = searchStepSizes[1];
//						searchStepSizes[1] = newStepSize;
//						printf("Found New Maximum\n");
//
//					}
//					else
//					{
//						searchValues[0] = newValue;
//
//						searchStepSizes[0] =newStepSize;
//					}
//				}
//				else
//				{
//					double newStepSize = (searchStepSizes[2] + searchStepSizes[1]) * 0.5;
//					setWorkingParamters(gradient, newStepSize, startParameters, workParameters);
//					updateFunction->setWeights(workParameters);
//
//					double newValue = evaluator->evaluatePolicy();
//
//					printf("StepSize %f : %f\n", newStepSize, newValue);
//					DebugPrint('l', "Finished Evaluation of StepSize %f : Value %f\n", newStepSize, newValue);
//					if (newValue > searchValues[1])
//					{
//						searchValues[0] = searchValues[1];
//						searchValues[1] = newValue;
//
//						searchStepSizes[0] = searchStepSizes[1];
//						searchStepSizes[1] = newStepSize;
//						printf("Found New Maximum\n");
//
//					}
//					else
//					{
//						searchValues[2] = newValue;
//						searchStepSizes[2] = newStepSize;
//					}
//				}
//			}
//			maxLearnRate = searchStepSizes[1];
//
//		}
//		
//	}
//	delete [] values;
//
//	DebugPrint('l', "End Line Search, applying step Size %f\n", maxLearnRate);
//
//	printf("Applying maximum stepsize %f\n", maxLearnRate);
//	setWorkingParamters(gradient, maxLearnRate, startParameters, workParameters);
//	updateFunction->setWeights(workParameters);

	// Value of golden section (1 + sqrt(5))/2.0
	const double phi = 1.6180339887499;
	const double cphi = 1 - 1/phi;
	const double TOL = 1.0e-10;	// Maximal fractional precision
	const double TINY = 1.0e-10;    // Can't use fractional precision when minimum is at 0

	// Bracket the minimum
	
	double br_min = 0.0;
	double br_max = getParameter("LineSearchStepSizeScale");
	double br_mid = br_max;
	
	updateFunction->getWeights(startParameters);
	
	
	printf("Bracketing Minimum\n");
	bracketMinimum(startParameters, gradient, fpt, br_min, br_mid, br_max);
	printf("Done...\n");

// Use Brent's algorithm to find minimum
// Initialise the points and function values
	double w = br_mid;   	// Where second from minimum is
	double v = br_mid;   	// Previous value of w
	double x = v;   	// Where current minimum is
	double e = 0.0; 	// Distance moved on step before last
	double fx = getFunctionValue(startParameters, gradient, x);

	double fv = fx; 
	double fw = fx;
	double fu = 0.0;

	for (int n = 1; n < maxSteps; n ++)
	{
		double xm = 0.5 * (br_min+br_max);  // Middle of bracket
	  	// Make sure that tolerance is big enough
		double tol1 = TOL * (fabs(x)) + TINY;
	  	// Decide termination on absolute precision required by options(2)
		if (fabs(x - xm) <= precision_treshold && br_max-br_min < 4*precision_treshold)
	    {
    		printf("Exiting Line Search x = %f - PrecisionTreshold\n", x);
    		break;
	    }
	  
  // Check if step before last was big enough to try a parabolic step.
  // Note that this will fail on first iteration, which must be a golden
  // section step.
	  
	  double r = 0.0;
   	  double q = 0.0;
      double p = 0.0;
	  double d = 0.0;
	  double u = 0.0;
	    
	  if (fabs(e) > tol1)
      {
      	 // Construct a trial parabolic fit through x, v and w
	     r = (fx - fv) * (x - w);
    	 q = (fx - fw) * (x - v);
    	 p = (x - v)*q - (x - w)*r;
    	 q = 2.0 * (q - r);
	     if (q > 0.0)
	     { 
	    	p = -p; 
	     }
	    q = fabs(q);
    	// Test if the parabolic fit is OK
	    if (fabs(p) >= fabs(0.5*q*e) || p <= q*(br_min-x) || p >= q*(br_max-x))
		{
        // No it isn't, so take a golden section step
	      if (x >= xm)
    	  {
    	  	e = br_min-x;
    	  }
      	  else
      	  {
		    e = br_max-x;
      	  }
          d = cphi*e;
		}
	     else
	     {
	      // Yes it is, so take the parabolic step
	       e = d;
	       d = p/q;
	       u = x+d;
	       if (u-br_min < 2*tol1 || br_max-u < 2*tol1)
	       {
	       	  if (xm - x < 0)
	       	  {
		       	  d = -tol1;
	       	  }
	       	  else
	       	  {
		       	  d = tol1;
	       	  }
	       }  
	     }
      }
   	  else
   	  {
	  // Step before last not big enough, so take a golden section step
		if (x >= xm)
		{
		   e = br_min - x;
		}
	    else
    	{ 
    	   e = br_max - x;
    	}

    	d = cphi*e;
   	  }
	  // Make sure that step is big enough
	  if (fabs(d) >= tol1)
	  {
	    u = x + d;
	  }
	  else
	  {
	  	if (d > 0)
	  	{
	    	u = x + tol1;
	  	}
	  	else
	  	{
	    	u = x - tol1;
	  	}
	  }
	
  	  // Evaluate function at u
	  fu = getFunctionValue(startParameters, gradient, u);
	  
 	  // Reorganise bracket
	  if (fu <= fx)
      {	
    	if (u >= x)
	    {
	      br_min = x;
	    }
    	else
    	{
	      br_max = x;
    	}
        v = w; 
        w = x; 
        x = u;
      
        fv = fw; 
        fw = fx; 
        fx = fu;
      }
  	  else
	  {
	    if (u < x)
	    {
    	    br_min = u;   
	    }
	    else
      	{
	      	br_max = u;
      	}
      	
	    if (fu <= fw || w == x)
	    {
	      v = w; w = u;
    	  fv = fw; fw = fu;
	    }
    	else
    	{
    		if (fu <= fv || v == x || v == w)
    		{
		      v = u;
		      fv = fu;
    		}
    	}
	  }
      printf("Cycle %d  Error %f, br_min: %f, br_max: %f, x: %f\n", n, fx, br_min, br_max, x);
	}	  
	setWorkingParamters(gradient, x, startParameters, workParameters);
	updateFunction->setWeights(workParameters);
	lmin = x;	
	return fx;
}

CGradientLearner::CGradientLearner(CGradientCalculator *l_gradientCalculator)
{
	gradientCalculator = l_gradientCalculator;
	addParameters(gradientCalculator);
}

CBatchGradientLearner::CBatchGradientLearner(CGradientCalculator *gradientCalculator, CGradientFunctionUpdater *l_updater) : CGradientLearner(gradientCalculator)
{
	updater = l_updater;
	addParameters(updater);
	treshold_f = 0.0001;
	
	gradient = new CFeatureList();
}

CBatchGradientLearner::~CBatchGradientLearner()
{
	delete gradient;
}
		
double CBatchGradientLearner::doOptimization(int maxSteps)
{
	double fold = gradientCalculator->getFunctionValue();
	double fnew = fold;
	for (int i = 0; i < maxSteps; i ++)
	{
		gradient->clear();
		
		printf("Getting gradient...\n");
		gradientCalculator->getGradient(gradient);
		printf("done... multiplying with -1 \n");
		gradient->multFactor(-1.0);
		printf("Updating weights...");
		updater->updateWeights(gradient);
		printf("done");
		
		fnew = gradientCalculator->getFunctionValue();
		printf("New Function Value (%d): %f\n", i, fnew);
		
		if (fabs(fnew - fold) < treshold_f)
		{
			printf("Change in Functionvalue below treshold - Exiting\n");
			break;
		}
		fold = fnew;
	}
	return fold;
}

CConjugateGradientLearner::CConjugateGradientLearner(CGradientCalculator *gradientCalculator, CLineSearchGradientFunctionUpdater *updater) : CGradientLearner(gradientCalculator)
{
	this->gradientUpdater = updater;
	
	addParameters(gradientUpdater);

	gradnew = new CFeatureList();
	gradold = new CFeatureList();
	d = new CFeatureList();


	treshold_x = 0.0001;
	treshold_f = 0.0001;
	
	fnew = 0.0;
	
	exiting = 0;
}

CConjugateGradientLearner::~CConjugateGradientLearner()
{
	delete gradnew;
	delete gradold;
	delete d;
}

double CConjugateGradientLearner::doOptimization(int maxGradientUpdates)
{
	int niters = maxGradientUpdates;

	int maxExits = 10;
	
	double fold = 0.0;

//	double br_min = 0;
//	double br_max = 1.0;	// Initial value for maximum distance to search along
	
//	const double tol = 1.0e-10;;

	int j = 1;

//	double *parameters = new double[gradientFunction->getNumWeights()];
//	double *oldParameters = new double[gradientFunction->getNumWeights()];
	
	if (exiting > maxExits)
	{
		printf("Finished Optimization\n");
		return fnew;
	}
	
	while (j <= niters)
	{
//		memcpy(oldParameters, parameters, sizeof(double) * gradientFunction->getNumWeights());

		gradnew->clear();
		gradientCalculator->getGradient(gradnew);
		    
//	    gradnew = feval(gradf, x, varargin{:});
//  options(11) = options(11) + 1;

		// Use Polak-Ribiere formula to update search direction
		double gg = gradnew->multFeatureList(gradnew);
	    if (gg == 0.0)
    	{
			printf("Gradient is 0 - Exiting\n");
    		// If the gradient is zero then we are done.
	    	break;
    	}

		
		if (d->size() == 0 || gradold->size() == 0)
		{
			
			fnew =  gradientCalculator->getFunctionValue();
			printf("Calculated new value: %f \n", fnew);
				
			d->clear();
			d->add(gradnew, -1.0); // Initial search direction		
		}
		else
		{	
			//gamma = ((gradnew - gradold)*(gradnew)')/gg;
		    gradold->multFactor(-1.0);
		    gradold->add(gradnew);
	    
		    double gamma = gradold->multFeatureList(gradnew) / gg;
		    d->multFactor(gamma);
		    d->add(gradnew, -1.0);
		}
		  
    	fold = fnew;
    	gradold->clear();
		gradold->add(gradnew);


  		// This shouldn't occur, but rest of code depends on d being downhill
		double prod_d = gradnew->multFeatureList(d);
		if (prod_d > 0)
		{
		    d->clear();
		    d->add(gradnew, -1.0);
            printf("search direction uphill in conjgrad\n");
		}
		double norm_d = sqrt(d->multFeatureList(d));  
			    
	    double lmin = 0.0;
	    
   		d->multFactor(1.0 / norm_d);
   		
	    printf("Starting LineSearch\n");
	    
	    fnew = gradientUpdater->updateWeights(d, fold, lmin);
	    
	    printf("result: %e %f %f\n", lmin, fnew, fold);
        // Set x and fnew to be the actual search point we have found
		// x = xold + lmin * line_sd;
		// fnew = line_options(8);

		d->multFactor(norm_d);
		// Check for termination
	    if (lmin * norm_d < treshold_x && fabs(fnew - fold) < treshold_f)
		{
			printf("Updates are smaller than treshold - Exiting\n");
			break;
		}
	       
//	    d = (d .* gamma) - gradnew;
	    printf("Cycle %d  Function %f\n", j, fnew);

		j = j + 1;
	}

// If we get here, then we haven't terminated in the given number of 
// iterations.
	if (j > niters)
	{
		printf("Maximum number of iterations reached\n");
		exiting = 0;
	}
	else
	{
		printf("Finished Optimization\n");
		d->clear();
		exiting ++;
	}
	return fnew;
}

void CConjugateGradientLearner::resetOptimization()
{
	CGradientLearner::resetOptimization();
	
	exiting = 0;
	
	d->clear();
	gradold->clear();
	
}

CSupervisedNeuralNetworkMatlabLearner::CSupervisedNeuralNetworkMatlabLearner(CTorchGradientFunction *l_mlpFunction, int numHidden)
{
	mlpFunction = l_mlpFunction;
	addParameter("MatlabNeuralNetNumNeurons", numHidden);
}

CSupervisedNeuralNetworkMatlabLearner::~CSupervisedNeuralNetworkMatlabLearner()
{
	if (mlpFunction->getMachine() != NULL)
	{
		delete mlpFunction->getMachine();
	}
}
		
void CSupervisedNeuralNetworkMatlabLearner::learnFA(CDataSet *inputData, CDataSet1D *outputData)
{
	/*
	FILE *inputFile = fopen("InputData.csv", "w");
	inputData->saveCSV(inputFile);
	fclose(inputFile);

	FILE *targetFile = fopen("TargetData.csv", "w");

	outputData->saveCSV(targetFile);
	fclose(targetFile);

	int numHidden = (int)  getParameter("MatlabNeuralNetNumNeurons");

	char syscall[250];
	sprintf(syscall, "matlab -r \"learnNeuralNet(%d)\"", numHidden);
	
	printf("System Call: %s\n", syscall);
	system(syscall);

	if (mlpFunction->getMachine() != NULL)
	{
		delete mlpFunction->getMachine();
	}

	MLP *mlp = new MLP(3, inputData->getNumDimensions(), "linear", numHidden, "tanh", numHidden, "linear", 1);

	mlpFunction->setGradientMachine( mlp);
	mlpFunction->loadDataFromFile("NetworkWeights.csv");

	char *systemCall = "rm NetworkWeights.csv";

	system(systemCall);

	char *filename = "NetworkWeights_config.data";


	FILE *configFile = fopen(filename, "r");
		
	if (configFile == NULL)
	{
		printf("Configuration file %s not found\n", filename);
		exit(-1);
	}
				
	ColumnVector inputMean(3);
	ColumnVector inputStd(3);
	ColumnVector outputMean(1);
	ColumnVector outputStd(1);
	
	inputMean = 0;
	inputStd = 1;
	
	outputMean = 0;
	outputStd = 1;
		
	// load configuration
	while (!feof(configFile))
	{
		char buffer[100];
		fscanf(configFile, "%s ", buffer);
		
		if (strcmp(buffer, "NUMHIDDEN:") == 0)
		{
			fscanf(configFile, "%d", &numHidden);
	//		printf("NumHidden: %d\n", numHidden);
		}
			
		if (strcmp(buffer, "PREPROCESSING_INPUTMEAN:") == 0)
		{
			double buf1, buf2 , buf3;
			fscanf(configFile, "%lf, %lf, %lf", &buf1, &buf2, &buf3);
			inputMean << buf1 << buf2 << buf3;
			
	//		printf("Input Mean : %lf, %lf, %lf\n", buf1, buf2, buf3);
		}
		if (strcmp(buffer, "PREPROCESSING_INPUTSTD:") == 0)
		{
			double buf1, buf2 , buf3;
			fscanf(configFile, "%lf, %lf, %lf", &buf1, &buf2, &buf3);
			inputStd << buf1 << buf2 << buf3;
	//		printf("Input Std : %lf, %lf, %lf\n", buf1, buf2, buf3);
		}
		if (strcmp(buffer, "PREPROCESSING_OUTPUTMEAN:") == 0)
		{
			double buf1;
			fscanf(configFile, "%lf ", &buf1);
			outputMean << buf1;
				
	//		printf("Output Mean : %lf\n", buf1);
		}
		if (strcmp(buffer, "PREPROCESSING_OUTPUTSTD:") == 0)
		{
			double buf1;
			fscanf(configFile, "%lf ", &buf1);
			outputStd << buf1;
	//		printf("Output Std : %lf\n", buf1);
		}
	}

	fclose(configFile);

	systemCall = "rm NetworkWeights_config.data";
	system(systemCall);

	mlpFunction->setInputMean(&inputMean);
	mlpFunction->setOutputMean(&outputMean);
	mlpFunction->setInputStd(&inputStd);
	mlpFunction->setOutputStd(&outputStd);
	 */
}

void CSupervisedNeuralNetworkMatlabLearner::resetLearner()
{
	if (mlpFunction->getMachine() != NULL)
	{	
		delete mlpFunction->getMachine();
		mlpFunction->setGradientMachine(NULL);
	}

}


		

CSupervisedNeuralNetworkTorchLearner::CSupervisedNeuralNetworkTorchLearner(CTorchGradientFunction *l_mlpFunction)
{
	mlpFunction = l_mlpFunction;

	addParameter("TorchLearningRate", 0.01);
	addParameter("TorchLearningRateDecay", 0.0);
	addParameter("TorchMaxIterations", -1);
	addParameter("TorchAccuracy", 0.0001);
}

CSupervisedNeuralNetworkTorchLearner::~CSupervisedNeuralNetworkTorchLearner()
{
}
		

void CSupervisedNeuralNetworkTorchLearner::learnFA(CDataSet *inputData, CDataSet1D *outputData)
{
	learnWeightedFA(inputData, outputData, NULL);
}

void CSupervisedNeuralNetworkTorchLearner::learnWeightedFA(CDataSet *inputData, CDataSet1D *outputData, CDataSet1D *weightingSet)
{
	/*
	double *weighting = NULL;

	ColumnVector inputMean(inputData->getNumDimensions());
	ColumnVector inputStd(inputData->getNumDimensions());

	inputData->getMean(NULL, &inputMean);
	inputData->getVariance(NULL, &inputStd);

	for (int i = 0; i < inputStd.nrows(); i++)
	{
		inputStd.element(i) = sqrt(inputStd.element(i));
	}


	mlpFunction->setInputMean(&inputMean);
	mlpFunction->setInputStd(&inputStd);

	CMeanStdPreprocessor preProc(inputData);
	
	preProc.preprocessDataSet(inputData);
	

	if (weightingSet)
	{
		weighting = new double[inputData->size()];

		for (int i = 0; i < weightingSet->size(); i++)
		{
			weighting[i] = (*weightingSet)[i];
		}
	}

	//Sequence *inputSequence = new Sequence(inputData->size(), inputData->getNumDimensions());

	//Sequence *outputSequence = new Sequence(inputData->size(), 1);
	Sequence **inputSequence = new Sequence *[inputData->size()];
	Sequence **outputSequence = new Sequence *[inputData->size()];


	for (int i = 0; i < inputData->size(); i ++)
	{
		ColumnVector *inputVector = (*inputData)[i];

		inputSequence[i] = new Sequence(1, inputVector->nrows());

		for (int j = 0; j < inputData->getNumDimensions(); j ++)
		{
			inputSequence[i]->frames[0][j] = inputVector->element(j);
		}
		
		outputSequence[i] = new Sequence(1, 1);
		outputSequence[i]->frames[0][0] = (*outputData)[i];	
	}

	
	MemoryDataSet *torchDataSet = new MemoryDataSet();

	torchDataSet->setInputs(inputSequence, inputData->size());
	torchDataSet->setTargets(outputSequence, inputData->size());

	Criterion *torchCriterion;

	if (weighting)
	{
		torchCriterion = new WeightedMSECriterion(torchDataSet);
		//torchCriterion = new WeightedMSECriterion(torchDataSet, weighting);
	}
	else
	{
		torchCriterion = new WeightedMSECriterion(torchDataSet);
	}

	mlpFunction->getGradientMachine()->reset();
	StochasticGradient stochGrad(mlpFunction->getGradientMachine(), torchCriterion);

	stochGrad.setROption("learning rate", getParameter("TorchLearningRate"));
	stochGrad.setROption("learning rate decay", getParameter("TorchLearningRateDecay"));

	stochGrad.setIOption("max iter", getParameter("TorchMaxIterations"));
	stochGrad.setROption("end accuracy", getParameter("TorchAccuracy"));

	stochGrad.train(torchDataSet, NULL);

	//mlpFunction->saveData(stdout);

	delete torchCriterion;
	delete torchDataSet;

	for (int i = 0; i < inputData->size(); i ++)
	{
		delete inputSequence[i];
		delete outputSequence[i];
	} 

	delete [] inputSequence;
	delete [] outputSequence;
	
	if (weighting)
	{
		delete weighting;
	}
	 */
}

void CSupervisedNeuralNetworkTorchLearner::resetLearner()
{

}

/*
CSupervisedLearner::CSupervisedLearner(int nInputs, int nOutputs)
{
	outputError = new ColumnVector(nOutputs);
}


CSupervisedLearner::~CSupervisedLearner()
{
	delete outputError;
}


void CSupervisedLearner::learnExample(ColumnVector *input, ColumnVector *target)
{
	testExample(input, outputError);
	outputError->multScalar(-1.0);
	outputError->addVector(target);

	learnExample(input, target, outputError);
}

CSupervisedGradientFunctionLearner::CSupervisedGradientFunctionLearner(CGradientFunction *gradientFunction) : CSupervisedLearner(gradientFunction->getNumInputs(), gradientFunction->getNumOutputs())
{
	this->gradientFunction = gradientFunction;
	this->gradient = new CFeatureList();
	this->localGradient = new CFeatureList();

	addParameter("SGLLearningRate", 0.01);
	addParameter("SGMLMomentum",0.1);

	addParameters(gradientFunction);
}

CSupervisedGradientFunctionLearner::~CSupervisedGradientFunctionLearner()
{
	delete gradient;
}

void CSupervisedGradientFunctionLearner::learnExample(ColumnVector *input, ColumnVector *target, ColumnVector *outputError)
{
	localGradient->clear();
	gradient->multFactor(getParameter("SGMLMomentum"));

	gradientFunction->getGradient(input, outputError, localGradient);
	gradient->add(localGradient);

	gradientFunction->updateGradient(gradient, getParameter("SGLLearningRate"));
}



void CSupervisedGradientFunctionLearner::testExample(ColumnVector *input, ColumnVector *output)
{
	gradientFunction->getFunctionValue(input, output);
}

int CSupervisedGradientFunctionLearner::getNumInputs()
{
	return gradientFunction->getNumInputs();
}

int CSupervisedGradientFunctionLearner::getNumOutputs()
{
	return gradientFunction->getNumOutputs();
}

void CSupervisedGradientFunctionLearner::saveData(FILE *stream)
{
	gradientFunction->saveData(stream);
}

void CSupervisedGradientFunctionLearner::loadData(FILE *stream)
{
	gradientFunction->loadData(stream);
}

void CSupervisedGradientFunctionLearner::resetData()
{
	gradientFunction->resetData();
}
*/
