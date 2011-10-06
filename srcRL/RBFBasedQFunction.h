/*
 *  RBFBasedQFunction.h
 *  RLTools
 *
 *  Created by Robert Busa-Fekete on 10/5/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */
#ifndef __RBFBASEDQFUNCTION_H
#define __RBFBASEDQFUNCTION_H

#include "cqfunction.h"
#include "cqetraces.h"
#include "cgradientfunction.h"
#include "cstatemodifier.h"
#include "RBFStateModifier.h"
#include "RBFQETraces.h"
#include "AdaBoostMDPClassifierAdv.h"
#include "cfeaturefunction.h"

class RBF {
protected:
	double _mean;
	double _sigma;
	double _alpha;
	
public:
	RBF() : _mean(0), _sigma(0), _alpha(0) {}
	virtual ~RBF() {}
	
	virtual double getMean() { return _mean; }
	virtual double getSigma() { return _sigma; }
	virtual double getAlpha() { return _alpha; }
	
	virtual void setMean( double m ) { _mean=m; }
	virtual void setSigma( double s ) { _sigma = s; }
	virtual void setAlpha( double a ) { _alpha = a; }
	
	virtual double getValue( double x ) 
	{ 
		double retVal = _alpha * exp( - (( x-_mean )*( x-_mean ))/(2*_sigma*_sigma));
		return retVal;
	}
};


class RBFBasedQFunctionBinary : public CGradientQFunction // CAbstractQFunction
{
protected:
	vector< map< CAction*, vector<RBF> > >	_rbfs;
	int _featureNumber;	
	CActionSet* _actions;
	int _numberOfActions;
    
    double _muAlpha;
    double _muMean;
    double _muSigma;
public:
	RBFBasedQFunctionBinary(CActionSet *actions, CStateModifier* statemodifier ) : CGradientQFunction(actions)
	{
        //CGradientQFunction ancestor init
        addType(GRADIENTQFUNCTION);        
        this->localGradientQFunctionFeatures = new CFeatureList();
        
		// the statemodifier must be RBFStateModifier
		RBFStateModifier* smodifier = dynamic_cast<RBFStateModifier*>( statemodifier );
		
		const int iterationNumber = smodifier->getNumOfIterations();
		const int featureNumber = smodifier->getNumOfRBFsPerIteration();
		const int numOfClasses = smodifier->getNumOfClasses();
		
		_featureNumber = featureNumber;
		
		assert(numOfClasses==1);
		
		_actions = actions;
		_numberOfActions = actions->size();
		
		_rbfs.resize( iterationNumber );
		for( int i=0; i<iterationNumber; ++i)
		{
			CActionSet::iterator it=(*actions).begin();
			for(;it!=(*actions).end(); ++it )
			{				
				vector<RBF> tmpVec(_featureNumber);
				_rbfs[i].insert(pair<CAction*, vector<RBF> >(*it, tmpVec) );
			}
		}
	}
    
    virtual double getMuAlpha() { return _muAlpha; }
	virtual double getMuMean() { return _muMean; }
	virtual double getMuSigma() { return _muSigma; }
	
	virtual void setMuMean( double m ) { _muMean=m; }
	virtual void setMuSigma( double s ) { _muSigma = s; }
	virtual void setMuAlpha( double a ) { _muAlpha = a; }
    
		
	virtual ~RBFBasedQFunctionBinary() {}
    
    virtual void resetData() {}
    
    virtual int getNumWeights()
    {
        int num = 0;
        
		int iterationNumber = _rbfs.size();
		for( int i=0; i<iterationNumber; ++i)
		{
			CActionSet::iterator it=(*actions).begin();
			for(;it!=(*actions).end(); ++it )
			{				
				num += _rbfs[i][*it].size();
			}
		}
        return num;
    }
    
	
	/// Interface for getting a Q-Value
	virtual double getValue(CStateCollection *state, CAction *action, CActionData *data = NULL) 
	{
		CState* currState = state->getState();
		
		int currIter = currState->getDiscreteState(0);
		double margin = currState->getContinuousState(0);
		
		double retVal = 0.0;
		
		for( int i=0; i<_featureNumber; ++i )
		{
			retVal += _rbfs[currIter][action][i].getValue(margin);  // lehet, hogy eggyet hozza kell adni a currIter-hez
		}		
		return retVal;
	}

//	virtual void updateValue(CStateCollection *state, CAction *action, double td, CActionData * = NULL)
//	{
//		CState* currState = state->getState();
//		
//		int currIter = currState->getDiscreteState(0);
//		double margin = currState->getContinuousState(0);
//		
//	}
	
//    void adaptCenters(CStateCollection *state, CAction *action) {
	virtual void updateValue(CStateCollection *state, CAction *action, double td, CActionData * = NULL)
    {
        CState* currState = state->getState();
        int currIter = currState->getDiscreteState(0);
		double margin = currState->getContinuousState(0);

        vector<RBF>& rbfs = _rbfs[currIter][action];
        int numCenters = rbfs.size();//_rbfs[currIter][action].size();
        
        for (int i = 0; i < numCenters; ++i) {
            
            double alpha = rbfs[i].getAlpha();
            double mean = rbfs[i].getMean();
            double sigma = rbfs[i].getSigma();

            double distance = margin - mean;
            double rbfValue = rbfs[i].getValue(margin);
//            double qValue = this->getValue(state, action);
            
            double alphaGrad = rbfValue / sigma;
            double meanGrad = rbfValue * alpha * distance / (sigma*sigma);
            double sigmaGrad = rbfValue * alpha * distance * distance / (sigma*sigma*sigma);        
            
            //update the center and shape
            rbfs[i].setAlpha(alpha + _muAlpha * alphaGrad);
            rbfs[i].setMean(mean + _muMean * meanGrad);
            rbfs[i].setSigma(sigma + _muSigma * sigmaGrad);
        }
    }
    
//	CAbstractQETraces* getStandardETraces()
//	{
//		return new RBFQETraces(this);
//	}
	
	virtual void saveQTable( const char* fname )
	{
		FILE* outFile = fopen( fname, "w" );
		
		for (CActionSet::iterator it=_actions->begin(); it != _actions->end(); ++it)
		{
//TMP			fprintf(outFile,"%d ", dynamic_cast<CAdaBoostAction>(*it)->getMode() );
		}
		fprintf(outFile,"\n");
		
		for(int i=0; i<_rbfs.size(); ++i)
		{
			for (CActionSet::iterator it=_actions->begin(); it != _actions->end(); ++it)
			{
			}
		}
				
		fclose( outFile );
	}
    
    void getWeights(double *weights)
    {
        double *offset = weights;

        int iterationNumber = _rbfs.size();
		for( int i=0; i<iterationNumber; ++i)
		{
			CActionSet::iterator it=(*actions).begin();
			for(;it!=(*actions).end(); ++it )
			{				
                int numFeat = _rbfs[i][*it].size();
				for (int j = 0; j < numFeat; ++j) {
                    offset[j] = _rbfs[i][*it][j].getAlpha();
                }
                offset += numFeat;
			}
		}
    }
    
    void setWeights(double *weights)
    {
        double *offset = weights;
        
        int iterationNumber = _rbfs.size();
		for( int i=0; i<iterationNumber; ++i)
		{
			CActionSet::iterator it=(*actions).begin();
			for(;it!=(*actions).end(); ++it )
			{				
                int numFeat = _rbfs[i][*it].size();
				for (int j = 0; j < numFeat; ++j) {
                    _rbfs[i][*it][j].setAlpha(offset[j]);
                }
                offset += numFeat;
			}
		}        
    }
    
    void getGradient(CStateCollection *state, CAction *action, CActionData *data, CFeatureList *gradientFeatures)
    {    
        return;
    }
    
    void updateWeights(CFeatureList *features)
    {
        return;
    }
    
	
};

#endif