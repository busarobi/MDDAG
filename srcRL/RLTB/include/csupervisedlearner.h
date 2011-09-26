
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

#ifndef C_SUPERVISEDTRAINER__H
#define C_SUPERVISEDTRAINER__H

#include "cparameters.h"

class Matrix;
class ColumnVector;

class CGradientFunction;
class CGradientUpdateFunction;
class CDataSet;
class CDataSet1D;
class CAction;
class CTorchGradientFunction;
class CFeatureList;
class CFeatureFunction;

class CSupervisedLearner : virtual public CParameterObject
{
	protected:
	public:
		CSupervisedLearner() {};
		virtual ~CSupervisedLearner() {};
		

		virtual void learnFA(CDataSet *inputData, CDataSet1D *outputData) = 0;

		virtual void resetLearner() {};
};

class CSupervisedWeightedLearner : virtual public CParameterObject
{
protected:

public:
	CSupervisedWeightedLearner() {};
	virtual ~CSupervisedWeightedLearner() {};

	virtual void learnWeightedFA(CDataSet *inputData, CDataSet1D *outputData, CDataSet1D *weighting) = 0;

	virtual void resetLearner() {};

};

class CSupervisedQFunctionLearner : virtual public CParameterObject
{
	protected:
	public:
		CSupervisedQFunctionLearner() {};
		virtual ~CSupervisedQFunctionLearner() {};

		virtual void learnQFunction(CAction *action, CDataSet *inputData, CDataSet1D *outputData) = 0;

		virtual void resetLearner() {};
};


class CSupervisedQFunctionWeightedLearner : virtual public CParameterObject
{
	protected:
	public:
		CSupervisedQFunctionWeightedLearner() {};
		virtual ~CSupervisedQFunctionWeightedLearner() {};

		virtual void learnQFunction(CAction *action, CDataSet *inputData, CDataSet1D *outputData, CDataSet1D *weightingData) = 0;

		virtual void resetLearner() {};
};


class CLeastSquaresLearner : virtual public CParameterObject
{
	protected:
		Matrix *A;
		Matrix *A_pinv;
		ColumnVector *b;
					
		CGradientUpdateFunction *featureFunc;
	public:
		CLeastSquaresLearner(CGradientUpdateFunction *featureFunc, int numData);
		virtual ~CLeastSquaresLearner();
		
		virtual double doOptimization();
	    static double doOptimization(Matrix *A, Matrix *A_pinv, ColumnVector *b, ColumnVector *w, double lambda);
	
};


class CGradientCalculator : virtual public CParameterObject
{
protected:
public:
	virtual ~CGradientCalculator() {};

	virtual void getGradient(CFeatureList *gradient) = 0;
	virtual double getFunctionValue() = 0;
	
	virtual void resetGradientCalculator() {};
};


class CSupervisedGradientCalculator : public CGradientCalculator
{
	protected:
		CDataSet *inputData;
		
		CDataSet1D *outputData1D;
		CDataSet *outputData;
		
		CGradientFunction *gradientFunction;
	public:
		CSupervisedGradientCalculator(CGradientFunction *gradientFunction, CDataSet *inputData, CDataSet *outputData);
		virtual ~CSupervisedGradientCalculator();
		
		virtual void getGradient(CFeatureList *gradient);
		virtual double getFunctionValue();

		virtual void setData(CDataSet *inputData, CDataSet1D *outputData1D);
		virtual void setData(CDataSet *inputData, CDataSet *outputData);
};


class CSupervisedFeatureGradientCalculator : public CSupervisedGradientCalculator
{
protected:
	CFeatureFunction *featureFunction;
	CFeatureList *featureList;

public: 
	CSupervisedFeatureGradientCalculator(CFeatureFunction *featureFunction);
	virtual ~CSupervisedFeatureGradientCalculator();	

	CFeatureList *getFeatureList(ColumnVector *input);

	virtual void getGradient(CFeatureList *gradient);
	virtual double getFunctionValue();

};

class CGradientFunctionUpdater : virtual public CParameterObject
{
protected:
	CGradientUpdateFunction *updateFunction;
public:
	CGradientFunctionUpdater(CGradientUpdateFunction *updateFunction);
	virtual ~CGradientFunctionUpdater() {};
	
	virtual void updateWeights(CFeatureList *gradient) = 0;
	void addRandomParams(double randSize);

	CGradientUpdateFunction *getUpdateFunction() {return updateFunction;};
};

class CConstantGradientFunctionUpdater : public CGradientFunctionUpdater
{
public:
	CConstantGradientFunctionUpdater(CGradientUpdateFunction *updateFunction, double learningRate);
	virtual ~CConstantGradientFunctionUpdater() {};

	virtual void updateWeights(CFeatureList *gradient);
};

class CLineSearchGradientFunctionUpdater : public CGradientFunctionUpdater
{
protected:
	double *startParameters;
	double *workParameters;

	int maxSteps;

	CGradientCalculator *gradientCalculator;

	double precision_treshold;

	void setWorkingParamters(CFeatureList *gradient, double stepSize, double *startParameters, double *workParameters);
	virtual double getFunctionValue(double *startParameters, CFeatureList *gradient, double stepSize);
	
	void bracketMinimum(double *startParameters, CFeatureList *gradient, double fa, double &a, double &b, double &c);
public:

	CLineSearchGradientFunctionUpdater(CGradientCalculator *gradientCalculator, CGradientUpdateFunction *updateFunction, int maxSteps);
	virtual ~CLineSearchGradientFunctionUpdater();

	virtual void updateWeights(CFeatureList *gradient);
	virtual double updateWeights(CFeatureList *gradient, double fold, double &lmin);
};

class CGradientLearner : virtual public CParameterObject
{
	protected:
		CGradientCalculator *gradientCalculator;
	public:
		CGradientLearner(CGradientCalculator *gradientCalculator);
		virtual ~CGradientLearner() {};
		
		virtual double doOptimization(int maxSteps) = 0;
		
		virtual void resetOptimization() {gradientCalculator->resetGradientCalculator();};
};

class CSupervisedGradientLearner : public CSupervisedLearner
{
	protected:
		CGradientLearner *gradientLearner;
		CSupervisedGradientCalculator *gradientCalculator;
	public:
		CSupervisedGradientLearner(CGradientLearner *gradientLearner,		CSupervisedGradientCalculator *gradientCalculator, int episodes);
		virtual ~CSupervisedGradientLearner();
		

		virtual void learnFA(CDataSet *inputData, CDataSet1D *outputData);

		virtual void resetLearner();
};

class CSupervisedQFunctionLearnerFromLearners : public CSupervisedQFunctionLearner
{
	protected:
		std::map<CAction *, CSupervisedLearner *> *learnerMap;
	public:
		CSupervisedQFunctionLearnerFromLearners(std::map<CAction *, CSupervisedLearner *> *learnerMap);

		virtual ~CSupervisedQFunctionLearnerFromLearners();

		virtual void learnQFunction(CAction *action, CDataSet *inputData, CDataSet1D *outputData);

		virtual void resetLearner();
};

class CSupervisedQFunctionWeightedLearnerFromLearners : public CSupervisedQFunctionLearner
{
	protected:
		std::map<CAction *, CSupervisedWeightedLearner *> *learnerMap;
	public:
		CSupervisedQFunctionWeightedLearnerFromLearners(std::map<CAction *, CSupervisedWeightedLearner *> *learnerMap);

		virtual ~CSupervisedQFunctionWeightedLearnerFromLearners();

		virtual void learnQFunction(CAction *action, CDataSet *inputData, CDataSet1D *outputData, CDataSet1D *weightData);

		virtual void resetLearner();
};


class CBatchGradientLearner : public CGradientLearner
{
	protected:
		CGradientFunctionUpdater *updater;
		CFeatureList *gradient;
		
		double treshold_f;
	public:
		CBatchGradientLearner(CGradientCalculator *gradientCalculator, CGradientFunctionUpdater *updater);
		~CBatchGradientLearner();
		
		virtual double doOptimization(int maxSteps);
};

class CConjugateGradientLearner : public CGradientLearner
{
protected:
	CLineSearchGradientFunctionUpdater *gradientUpdater;

	CFeatureList *gradnew;
	CFeatureList *gradold;
	CFeatureList *d;
	
	double treshold_x;
	double treshold_f;
	
	double fnew;
	
	int exiting;

public:
	CConjugateGradientLearner(CGradientCalculator *gradientCalculator, CLineSearchGradientFunctionUpdater *updater);
	virtual ~CConjugateGradientLearner();

	virtual double doOptimization(int maxGradientUpdates);
	
	virtual void resetOptimization();
};


class CSupervisedNeuralNetworkMatlabLearner : public CSupervisedLearner
{
	protected:
		CTorchGradientFunction *mlpFunction;
		
	public:
		CSupervisedNeuralNetworkMatlabLearner(CTorchGradientFunction *mlpFunction, int numHidden);
		virtual ~CSupervisedNeuralNetworkMatlabLearner();
		

		virtual void learnFA(CDataSet *inputData, CDataSet1D *outputData);

		virtual void resetLearner();
};

class CSupervisedNeuralNetworkTorchLearner : public CSupervisedLearner, public CSupervisedWeightedLearner
{
	protected:
		CTorchGradientFunction *mlpFunction;
		
	public:
		CSupervisedNeuralNetworkTorchLearner(CTorchGradientFunction *mlpFunction);
		virtual ~CSupervisedNeuralNetworkTorchLearner();
		

		virtual void learnFA(CDataSet *inputData, CDataSet1D *outputData);
		virtual void learnWeightedFA(CDataSet *inputData, CDataSet1D *outputData, CDataSet1D *weighting);

		virtual void resetLearner();
};



#endif

