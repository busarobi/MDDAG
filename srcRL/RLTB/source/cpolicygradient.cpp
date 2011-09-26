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
#include "cpolicygradient.h"

#include "cagent.h"
#include "creinforce.h"
#include "ccontinuousactiongradientpolicy.h"
#include "cparameters.h"
#include "cfeaturefunction.h"
#include "caction.h"
#include "cpolicies.h"
#include "cevaluator.h"

#include <math.h>

CPolicyGradientCalculator::CPolicyGradientCalculator(CAgentController *policy, CPolicyEvaluator *evaluator)
{
	this->policy = policy;
	this->evaluator = evaluator;
}

double CPolicyGradientCalculator::getFunctionValue()
{
	return - evaluator->evaluatePolicy();
}

CGPOMDPGradientCalculator::CGPOMDPGradientCalculator(CRewardFunction *reward, CStochasticPolicy *policy, CPolicyEvaluator *evaluator, CAgent *agent, CReinforcementBaseLineCalculator *baseLine, int TStepsPerEpsiode, int Episodes, double beta) : CPolicyGradientCalculator(policy, evaluator), CSemiMDPRewardListener(reward)
{
	this->agent = agent;
	this->baseLine = baseLine;

	addParameters(baseLine);
	
	addParameter("GradientEstimationStepsPerEpisode", TStepsPerEpsiode);
	addParameter("GradientEstimationEpisodes", Episodes);
	addParameter("GPOMDPBeta", beta);

	localGradient = new CFeatureList();

	localZTrace = new CFeatureList();

	globalGradient = NULL;

	stochPolicy = policy;
}

CGPOMDPGradientCalculator::~CGPOMDPGradientCalculator()
{
	delete localGradient;
	delete localZTrace;
}

void CGPOMDPGradientCalculator::nextStep(CStateCollection *oldState, CAction *action, double reward, CStateCollection *)
{
	if (globalGradient)
	{
		localZTrace->multFactor(getParameter("GPOMDPBeta"));
		localGradient->clear();

		stochPolicy->getActionProbabilityLnGradient(oldState, action, action->getActionData(),localGradient);
		localZTrace->add(localGradient, 1.0);
		
		CFeatureList::iterator it = localZTrace->begin();
		if (DebugIsEnabled('g'))
		{
			DebugPrint('g', "reward: %f, baseline %f, -> factor %f\n", reward, baseLine->getReinforcementBaseLine((*it)->featureIndex));
			DebugPrint('g', "Z-trace: ");
			localZTrace->saveASCII(DebugGetFileHandle('g'));
		}

		for (;it != localZTrace->end(); it ++)
		{
			globalGradient->update((*it)->featureIndex, (reward - baseLine->getReinforcementBaseLine((*it)->featureIndex)) * (*it)->factor);
		}
	}
}

void CGPOMDPGradientCalculator::newEpisode()
{
	localZTrace->clear();
}

void CGPOMDPGradientCalculator::getGradient(CFeatureList *gradient)
{
	setGlobalGradient(gradient);

	int TSteps = my_round(getParameter("GradientEstimationStepsPerEpisode"));
	int nEpisodes = my_round(getParameter("GradientEstimationEpisodes"));

	agent->startNewEpisode();
	
	bool bListen = agent->isListenerAdded(this);

	if (!bListen)
	{
		agent->addSemiMDPListener(this);
	}

	printf("Calculating PGradient with %d steps and %d Episodes\n", TSteps,nEpisodes);
	
	int oldSteps = 0;
	int gradientSteps = 0;

	oldSteps = agent->getTotalSteps();

	for (int i = 0; i < nEpisodes; i++)
	{
		agent->startNewEpisode();
		agent->doControllerEpisode(1, TSteps);
		printf("Finished %d Episode\n", i);
	}
	gradientSteps = agent->getTotalSteps() - oldSteps;

	if (!bListen)
	{
		agent->removeSemiMDPListener(this);
	}
	assert(gradientSteps > 0);

	gradient->multFactor(- 1.0 / gradientSteps); // minus because of minimization

	if (DebugIsEnabled('g'))
	{
		DebugPrint('g', "Calculated GPOMDP Gradient (%d steps)\n", TSteps);
		gradient->saveASCII(DebugGetFileHandle('g'));
		DebugPrint('g', "\n");
	}

	setGlobalGradient(NULL);
}


CFeatureList* CGPOMDPGradientCalculator::getGlobalGradient()
{
	return globalGradient;
}

void CGPOMDPGradientCalculator::setGlobalGradient(CFeatureList *globalGradient)
{
	this->globalGradient = globalGradient;
}

CNumericPolicyGradientCalculator::CNumericPolicyGradientCalculator(CAgent *agent, CContinuousActionGradientPolicy *policy, CTransitionFunctionEnvironment *dynModel, CRewardFunction *l_rewardFunction, double stepSize, CPolicyEvaluator *evaluator) : CPolicyGradientCalculator(policy, evaluator)
{
	weights = new double[policy->getNumWeights()];

	this->rewardFunction = l_rewardFunction;
	this->agent = agent;
	this->dynModel = dynModel;

	addParameter("PolicyGradientNumericStepSize", stepSize);
	gradientPolicy = policy;
}

CNumericPolicyGradientCalculator::~CNumericPolicyGradientCalculator()
{
	delete [] weights;
}

void CNumericPolicyGradientCalculator::getGradient(CFeatureList *gradientFeatures)
{
	gradientPolicy->getWeights(weights);

	agent->setController(policy);

	double stepSize = getParameter("PolicyGradientNumericStepSize");

	double value = evaluator->evaluatePolicy();

	for (int i = 0; i < gradientPolicy->getNumWeights(); i ++)
	{
		weights[i] -= stepSize;
		gradientPolicy->setWeights(weights);
		double vMinus = evaluator->evaluatePolicy();
		weights[i] += 2 * stepSize;
		gradientPolicy->setWeights(weights);
		double vPlus = evaluator->evaluatePolicy();

		weights[i] -= stepSize;
		
		if (vMinus > value || vPlus > value)
		{
			gradientFeatures->set(i, (vPlus - vMinus) / (2 * stepSize));
		}
		else
		{
			gradientFeatures->set(i, 0);
		}

		printf("%f %f %f\n", stepSize, vPlus, vMinus);
		printf("Calculated derivation for weight %d : %f\n", i, gradientFeatures->getFeatureFactor(i));
	}
	gradientPolicy->setWeights(weights);
	
	gradientFeatures->multFactor(-1.0);
}


CRandomPolicyGradientCalculator::CRandomPolicyGradientCalculator(CContinuousActionGradientPolicy *policy, CPolicyEvaluator *evaluator, int numEvaluations, double stepSize) : CPolicyGradientCalculator(policy, evaluator)
{
	addParameter("RandomGradientNumEvaluations", numEvaluations);
	addParameter("RandomGradientStepSize", stepSize);
	
	gradientPolicy = policy;
	 
	stepSizes = new double[gradientPolicy->getNumWeights()];
	minWeights = new double[gradientPolicy->getNumWeights()];
	nullWeights = new double[gradientPolicy->getNumWeights()];
	plusWeights = new double[gradientPolicy->getNumWeights()];
	
	numMinWeights = new int[gradientPolicy->getNumWeights()];
	numMaxWeights = new int[gradientPolicy->getNumWeights()];
	numNullWeights = new int[gradientPolicy->getNumWeights()];
	
	memset(stepSizes, 0, sizeof(double) * gradientPolicy->getNumWeights());
}

CRandomPolicyGradientCalculator::~CRandomPolicyGradientCalculator()
{
	delete [] stepSizes;
	
	delete [] minWeights;
	delete [] nullWeights;
	delete [] plusWeights;
	
	delete [] numMinWeights;
	delete [] numMaxWeights;
	delete [] numNullWeights;
	
}

void CRandomPolicyGradientCalculator::getGradient(CFeatureList *gradient)
{
	gradient->clear();
	
	memset(minWeights, 0, sizeof(double) * gradientPolicy->getNumWeights());
	memset(nullWeights, 0, sizeof(double) *  gradientPolicy->getNumWeights());
	memset(plusWeights, 0, sizeof(double) * gradientPolicy->getNumWeights());
	
	memset(numMinWeights, 0, sizeof(int) * gradientPolicy->getNumWeights());
	memset(numMaxWeights, 0, sizeof(int) *  gradientPolicy->getNumWeights());
	memset(numNullWeights, 0, sizeof(int) * gradientPolicy->getNumWeights());
	
	int numEvals = (int) getParameter("RandomGradientNumEvaluations");
	double stepSize = getParameter("RandomGradientStepSize");
	
	double *parameters = new double[gradientPolicy->getNumWeights()];
	double *oldParameters = new double[gradientPolicy->getNumWeights()];
	int *step = new int[gradientPolicy->getNumWeights()];
	
	gradientPolicy->getWeights(oldParameters);
	
	int *numWeights[3];
	double *weights[3];
	
	weights[0] = minWeights;
	weights[1] = nullWeights;
	weights[2] = plusWeights;
	
	numWeights[0] = numMinWeights;
	numWeights[1] = numNullWeights;
	numWeights[2] = numMaxWeights;
	
	printf("Calculating gradient with random samples\n");	
	for (int i = 0; i < numEvals; i ++)
	{
		
		for (int j = 0; j < gradientPolicy->getNumWeights(); j++)
		{
			if (i > 0)
			{
				step[j] = rand() % 3 ;
			}
			else
			{
				step[j] = 1;	
			}
			
			if (stepSizes[j] == 0)
			{
				parameters[j] = oldParameters[j] + (step[j] - 1) * stepSize;
			}
			else
			{
				parameters[j] = oldParameters[j] + (step[j] - 1) * stepSizes[j];
			}		
		}
		gradientPolicy->setWeights(parameters);
		double value = evaluator->evaluatePolicy();
		
		printf("Evaluation %d: %f\n", i, value);
		
		for (int j = 0; j < gradientPolicy->getNumWeights(); j++)
		{
			numWeights[step[j]][j] ++;
			weights[step[j]][j] += value;
		}
	}
	
	for (int j = 0; j <  gradientPolicy->getNumWeights(); j++)
	{
		double l_stepSize = stepSize;
		if (stepSizes[j] > 0)
		{
			l_stepSize = stepSizes[j];
		}
		
		if (numWeights[0][j] > 0 && weights[0][j] / numWeights[0][j] > weights[1][j] / numWeights[1][j])
		{
			if (numWeights[2][j] > 0 && weights[2][j] / numWeights[2][j] > weights[0][j] / numWeights[0][j])
			{
				gradient->set(j, (weights[2][j] / numWeights[2][j] - weights[1][j] / numWeights[1][j]) / l_stepSize);
			}		
			else
			{
				gradient->set(j, (weights[0][j] / numWeights[0][j] - weights[1][j] / numWeights[1][j]) / ( - l_stepSize));
			}
		}
		else
		{
			if (numWeights[2][j] > 0 && weights[2][j] / numWeights[2][j] > weights[1][j] / numWeights[1][j])
			{
				gradient->set(j, (weights[2][j] / numWeights[2][j] - weights[1][j] / numWeights[1][j]) / l_stepSize);
			}
			else
			{
				gradient->set(j, 0.0);
//				stepSizes[j] = l_stepSize;
//				printf("No Uphill direction found, decreasing step size of parameter %d (%f)\n", j, stepSizes[j]);
			}
		}

	}

	gradient->multFactor(-1.0);	
	gradientPolicy->setWeights(oldParameters);
	
	delete [] step;
	delete [] parameters;
	delete [] oldParameters;
}

void CRandomPolicyGradientCalculator::setStepSize(int index, double stepSize)
{
	stepSizes[index] = stepSize;
}


CRandomMaxPolicyGradientCalculator::CRandomMaxPolicyGradientCalculator(CContinuousActionGradientPolicy *policy, CPolicyEvaluator *evaluator, int numEvaluations, double stepSize) : CPolicyGradientCalculator(policy, evaluator)
{
	addParameter("RandomGradientNumEvaluations", numEvaluations);
	addParameter("RandomGradientStepSize", stepSize);
	
	addParameter("RandomGradientStepSizeFactorNoImprove", 0.95);
	addParameter("RandomGradientStepSizeFactorImprove", 1.1);
	
	gradientPolicy = policy;
	 
	stepSizes = new double[gradientPolicy->getNumWeights()];
	workStepSizes = new double[gradientPolicy->getNumWeights()];
	
	memset(stepSizes, 0, sizeof(double) * gradientPolicy->getNumWeights());
}

CRandomMaxPolicyGradientCalculator::~CRandomMaxPolicyGradientCalculator()
{
	delete [] stepSizes;
	delete [] workStepSizes;

}

void CRandomMaxPolicyGradientCalculator::resetGradientCalculator()
{
	memcpy(workStepSizes, stepSizes, sizeof(double) * gradientPolicy->getNumWeights());
}

void CRandomMaxPolicyGradientCalculator::getGradient(CFeatureList *gradient)
{
	gradient->clear();
	
	int numEvals = (int) getParameter("RandomGradientNumEvaluations");
	double stepSize = getParameter("RandomGradientStepSize");
	
	double *parameters = new double[gradientPolicy->getNumWeights()];
	double *oldParameters = new double[gradientPolicy->getNumWeights()];
	
	double *bestParameters = new double[gradientPolicy->getNumWeights()];
	
	gradientPolicy->getWeights(oldParameters);
	
	
	printf("Calculating gradient with random samples, stepSize %f\n", workStepSizes[0]);	
	double bestValue = 0;
	
	bool newBest = false;
	
	for (int i = 0; i < numEvals; i ++)
	{
		
		for (int j = 0; j < gradientPolicy->getNumWeights(); j++)
		{
			int step = 1;
			if (i > 0)
			{
				step = rand() % 3 ;
			}
			
			if (workStepSizes[j] == 0)
			{
				parameters[j] = oldParameters[j] + (step - 1) * stepSize;
			}
			else
			{
				parameters[j] = oldParameters[j] + (step - 1) * workStepSizes[j];
			}		
		}
		
		gradientPolicy->setWeights(parameters);
		double value = evaluator->evaluatePolicy();
		
		printf("Evaluation %d: %f\n", i, value);
		
		if (i == 0 || value > bestValue)
		{
			printf("Found new Maximum %f -> %f\n", value, bestValue);
			memcpy(bestParameters, parameters, sizeof(double) * gradientPolicy->getNumWeights());
			bestValue = value; 
			
			if (i > 0)
			{
				newBest = true;
			}
		}
				
	}
	
	printf("Setting Gradient...\n");
	for (int j = 0; j <  gradientPolicy->getNumWeights(); j++)
	{
		gradient->set(j, -(bestParameters[j] - oldParameters[j]));
	}
	
	gradientPolicy->setWeights(oldParameters);
	
	double factor = getParameter("RandomGradientStepSizeFactorImprove");
	
	if (!newBest)
	{
		factor = getParameter("RandomGradientStepSizeFactorNoImprove");;
	}
	
	for (int i = 0; i < gradientPolicy->getNumWeights(); i ++)
	{
		if (workStepSizes[i] == 0)
		{
			workStepSizes[i] = stepSize * factor;
		}
		else
		{
			workStepSizes[i] = workStepSizes[i] * factor;
		}
	}
		
	printf("Finished calculating gradient, best value : %f\n", bestValue);
	
	delete [] parameters;
	delete [] oldParameters;
	delete [] bestParameters;
}

void CRandomMaxPolicyGradientCalculator::setStepSize(int index, double stepSize)
{
	stepSizes[index] = stepSize;
}


CGSearchPolicyGradientUpdater::CGSearchPolicyGradientUpdater(CGradientUpdateFunction *updateFunction, CPolicyGradientCalculator *gradientCalculator, double s0, double epsilon) : CGradientFunctionUpdater(updateFunction)
{
	this->gradientCalculator = gradientCalculator;

	startParameters = new double[updateFunction->getNumWeights()];
	workParameters = new double[updateFunction->getNumWeights()];

	addParameters(gradientCalculator, "GSearch");
	addParameter("GSearchStartStepSize", s0);
	addParameter("GSearchEpsilon",epsilon);
	addParameter("GSearchUseLastStepSize", 0.0);

	addParameter("GSearchMinStepSize", s0 / 256);
	addParameter("GSearchMaxStepSize", s0 * 16);

	lastStepSize = s0;
}

CGSearchPolicyGradientUpdater::~CGSearchPolicyGradientUpdater()
{
	delete [] startParameters;
	delete [] workParameters;
}

void CGSearchPolicyGradientUpdater::setWorkingParamters(CFeatureList *gradient, double stepSize, double *startParameters, double *workParameters)
{
	memcpy(workParameters, startParameters, sizeof(double) * updateFunction->getNumWeights());

	CFeatureList::iterator it = gradient->begin();
	for (; it != gradient->end(); it ++)
	{
		workParameters[(*it)->featureIndex] += stepSize * (*it)->factor;
	}
}

void CGSearchPolicyGradientUpdater::updateWeights(CFeatureList *gradient)
{

	double s = getParameter("GSearchStartStepSize");

//	double norm = sqrt(gradient->multFeatureList(gradient));

	if (getParameter("GSearchUseLastStepSize") > 0.5)
	{
		s = lastStepSize;
	}
	printf("Beginning GSEARCH with stepSize %f\n", s);

	double epsilon = getParameter("GSearchEpsilon");

	updateFunction->getWeights(startParameters);
	setWorkingParamters(gradient, s,startParameters, workParameters);

	updateFunction->setWeights(workParameters);
	CFeatureList *newGradient = new CFeatureList();
	gradientCalculator->getGradient(newGradient);

	double newGradientNorm = sqrt(newGradient->multFeatureList(newGradient));

	double prod = gradient->multFeatureList(newGradient);// * 1 / newGradientNorm;;
	double tempProd = prod;
	double sPlus = 0;
	double sMinus = 0;
	double pPlus = 0;
	double pMinus = 0;

	double sMin = getParameter("GSearchMinStepSize");
	double sMax = getParameter("GSearchMaxStepSize");

	printf("gradient * newgradient: %f\n", tempProd);

	if (prod < 0)
	{
		sPlus = s; 

		while(tempProd < - epsilon && s > sMin)
		{
			sPlus = s;
			pPlus = tempProd;
			s = s / 2;

			printf("GSearch StepSize: %f ", s);
			
			setWorkingParamters(gradient, s, startParameters, workParameters);
			updateFunction->setWeights(workParameters);
			newGradient->clear();
			gradientCalculator->getGradient(newGradient);

			newGradientNorm = sqrt(newGradient->multFeatureList(newGradient));
			tempProd = gradient->multFeatureList(newGradient);// * 1 / newGradientNorm;
			
			printf("GSearch StepSize: %f, gradient * newGradient: %f\n", s,tempProd);

		} 
		sMinus = s;
		pMinus = tempProd;
		if (s < sMin)
		{
			s = sMin;
		}
	}
	else
	{
		sMinus = s;
		while(tempProd > epsilon && s < sMax)
		{
			sMinus = s;
			pMinus = tempProd;

			s = 2 * s;

			setWorkingParamters(gradient, s, startParameters, workParameters);
			updateFunction->setWeights(workParameters);
			newGradient->clear();

			gradientCalculator->getGradient(newGradient);

			newGradientNorm = sqrt(newGradient->multFeatureList(newGradient));
			tempProd = gradient->multFeatureList(newGradient);// * 1 / newGradientNorm;

			printf("GSearch StepSize: %f, gradient * newGradient: %f\n", s,tempProd);
		}
		sPlus = s;
		pPlus = tempProd;

		if (s > sMax)
		{
			s = sMax;
		}
	}


	if (pMinus > 0 && pPlus < 0)
	{
		s = (pPlus * sMinus - pMinus * sPlus) / (pPlus - pMinus);
	}
	else
	{
		s = (sPlus + sMinus) / 2;
	}

	printf("GSearch: s: %f, s+ %f, s- %f, p+ %f, p- %f\n",s, sPlus, sMinus, pPlus, pMinus);

	DebugPrint('g',"GSearch: s: %f, s+ %f, s- %f, p+ %f, p- %f\n",s, sPlus, sMinus, pPlus, pMinus);

	setWorkingParamters(gradient, s, startParameters, workParameters);

	if (DebugIsEnabled('g'))
	{
		DebugPrint('g',"GSearch: Calculated StepSize %f\n", s);
		DebugPrint('g', "GSearch: New calculated Parameters\n");
		updateFunction->saveData(DebugGetFileHandle('g'));
	}
	
	lastStepSize = s;


	updateFunction->setWeights(workParameters);

	double normWeights = 0;
	
	for (int i = 0; i < updateFunction->getNumWeights(); i ++)
	{
		normWeights += workParameters[i] * workParameters[i];
	}
	normWeights = sqrt(normWeights);
	printf("Weights Norm after Update %f\n", normWeights);
}
 
/*
CPolicyGradientLearner::CPolicyGradientLearner(CPolicyGradientCalculator *gradientCalculator, CGradientFunctionUpdater *gradientUpdater, double epsilon)
{
	addParameters(gradientCalculator);
	addParameters(gradientUpdater);

	addParameter("GradientResolution", epsilon);
	addParameter("PolicyGradientWeightDecay", 0.0);

	gradient = new CFeatureList();
	hGradient = new CFeatureList();
	gGradient = new CFeatureList();

	this->gradientCalculator = gradientCalculator;
	this->gradientUpdater = gradientUpdater;
}

CPolicyGradientLearner::~CPolicyGradientLearner()
{
	delete gradient;
	delete hGradient;
	delete gGradient;
}

void CPolicyGradientLearner::doUpdate(CFeatureList *gradient)
{
	double gamma = getParameter("PolicyGradientWeightDecay");

	double *oldParameters = new double[gradientUpdater->getUpdateFunction()->getNumWeights()];
	double *newParameters = new double[gradientUpdater->getUpdateFunction()->getNumWeights()];
	gradientUpdater->getUpdateFunction()->getWeights(oldParameters);
	gradientUpdater->updateWeights(gradient);
	if (gamma > 0.0)
	{
		gradientUpdater->getUpdateFunction()->getWeights(newParameters);

		printf("Updating Gradient with weight decay %f\n", gamma);
		for (int i = 0; i < gradientUpdater->getUpdateFunction()->getNumWeights(); i++)
		{
			newParameters[i] -=  gamma * oldParameters[i];
		}
		gradientUpdater->getUpdateFunction()->setWeights(newParameters);

	}
	delete [] oldParameters;
	delete [] newParameters;
}

double CPolicyGradientLearner::learnPolicy(int maxGradientUpdates, CPolicyEvaluator *evaluator, bool useOldGradient)
{
	double epsilon = getParameter("GradientResolution");
	gradient->clear();

	if (!useOldGradient)
	{
		hGradient->clear();
		gGradient->clear();
	}

	

	double normG = gGradient->multFeatureList(gGradient);
	DebugPrint('g', "Gradient-Norm: %f\n", normG);

	printf("Gradient-Norm: %f\n", normG);

	int gradientUpdates = 0;

	double value = 0.0;

	do
	{
		
		if (evaluator)
		{
			value = evaluator->evaluatePolicy();
			printf("Value after %d Gradient Update: %f\n", gradientUpdates, value);
		}
		
		gradient->clear();
		gradientCalculator->getGradient(gradient);

		if (gGradient->size() > 0)
		{
			gGradient->add(gradient);
			normG = gGradient->multFeatureList(gGradient);

			gGradient->multFactor(-1.0);
			gGradient->add(gradient, 1.0);


			double gamma = gGradient->multFeatureList(gradient) / normG;
			DebugPrint('g', "Calculated Gradient :\n");
			if (DebugIsEnabled('g'))
			{
				gradient->saveASCII(DebugGetFileHandle('g'));
			}
			DebugPrint('g', "PGLearner: Gamma %f", gamma);

			hGradient->multFactor(gamma);
			hGradient->add(gradient);

			if (hGradient->multFeatureList(gradient) < 0)
			{
				hGradient->clear();
				hGradient->add(gradient);
			}

			DebugPrint('g', "Update-Gradient :\n");
			if (DebugIsEnabled('g'))
			{
				hGradient->saveASCII(DebugGetFileHandle('g'));
			}
		}
		else
		{
			hGradient->add(gradient);
			normG = hGradient->multFeatureList(hGradient);

		}
		
		gGradient->clear();
		gGradient->add(gradient);

		
		DebugPrint('g', "Gradient-Norm: %f\n", normG);
		printf("Gradient-Norm: %f\n", normG);

		printf("Updating Gradient...");


		doUpdate(hGradient);
		gradientUpdates ++;
		
	}
	while (normG > epsilon && gradientUpdates < maxGradientUpdates);
	
	if (gradientUpdates < maxGradientUpdates)
	{
		printf("Updating Gradient...");
		doUpdate(hGradient);
		gradientUpdates ++;

		if (evaluator)
		{
			double value = evaluator->evaluatePolicy();
			printf("Value after %d Gradient Update: %f\n", gradientUpdates, value);
		}
	}
	return value;
}*/




CPolicyGradientWeightDecayListener::CPolicyGradientWeightDecayListener(CGradientUpdateFunction *updateFunction, double weightdecay)
{
	addParameter("PolicyGradientWeightDecay", weightdecay);
	this->updateFunction = updateFunction;
	parameters = new double[updateFunction->getNumWeights()]; 
}

CPolicyGradientWeightDecayListener::~CPolicyGradientWeightDecayListener()
{
	delete [] parameters;
}

void CPolicyGradientWeightDecayListener::newEpisode()
{
	updateFunction->getWeights(parameters);

	double factor = 1 - getParameter("PolicyGradientWeightDecay");
	for (int i = 0; i < updateFunction->getNumWeights(); i++)
	{
		parameters[i] = factor *  parameters[i];
	}

	updateFunction->setWeights(parameters);
}

