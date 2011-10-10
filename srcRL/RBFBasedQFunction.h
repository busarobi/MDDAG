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
		double retVal = _alpha * getActivationFactor(x);
		return retVal;
	}
    
    virtual double getActivationFactor( double x ) 
	{ 
		double retVal = exp( - (( x-_mean )*( x-_mean ))/(2*_sigma*_sigma));
		return retVal;
	}
    
};


class RBFBasedQFunctionBinary : public CAbstractQFunction // CAbstractQFunction
{
protected:

	map< CAction*, vector<vector<RBF> > > 	_rbfs;
	int _featureNumber;	
	CActionSet* _actions;
	int _numberOfActions;
	double _gradCoeff;
    
public:
    //-----------------------------------------------------
    //-----------------------------------------------------			
	RBFBasedQFunctionBinary(CActionSet *actions, CStateModifier* statemodifier ) : CAbstractQFunction(actions)
	{
		// the statemodifier must be RBFStateModifier
		RBFStateModifier* smodifier = dynamic_cast<RBFStateModifier*>( statemodifier );
		
		const int iterationNumber = smodifier->getNumOfIterations();
		const int featureNumber = smodifier->getNumOfRBFsPerIteration();
		const int numOfClasses = smodifier->getNumOfClasses();
		
		_featureNumber = featureNumber;
		
		assert(numOfClasses==1);
		
		_actions = actions;
		_numberOfActions = actions->size();
		
        CActionSet::iterator it=(*actions).begin();
        for(;it!=(*actions).end(); ++it )
        {
            _rbfs[*it].resize( iterationNumber );
            for( int i=0; i<iterationNumber; ++i)
            {
                _rbfs[*it][i].resize(_featureNumber);
			}
		}
		
		_gradCoeff = 0.3;
	}
    //-----------------------------------------------------
    //-----------------------------------------------------	
    void uniformInit()
    {
        CActionSet::iterator it=_actions->begin();
        for(;it!=_actions->end(); ++it )
        {				
            int iterationNumber = _rbfs[*it].size();
            for( int i=0; i<iterationNumber; ++i)
            {                
                int numFeat = _rbfs[*it][i].size();
				for (int j = 0; j < numFeat; ++j) {
//                    if (numFeat % 2 == 0) {
                        _rbfs[*it][i][j].setMean((j+1) * 1./(numFeat+1));
//                    }
//                    else {
//                        _rbfs[*it][i][j].setMean(j * 1./numFeat);   
//                    }

                    
                    _rbfs[*it][i][j].setSigma(1.0 / (2*numFeat));
                }
            }
        }           
    }
    //-----------------------------------------------------
    //-----------------------------------------------------	        		
	virtual ~RBFBasedQFunctionBinary() {}
    
    //-----------------------------------------------------
    //-----------------------------------------------------	    
    virtual int getNumWeights()
    {
        int num = 0;
        
		int iterationNumber = _rbfs.size();
		for( int i=0; i<iterationNumber; ++i)
		{
			CActionSet::iterator it=(*actions).begin();
			for(;it!=(*actions).end(); ++it )
			{				
				num += _rbfs[*it][i].size();
			}
		}
        return num;
    }
    
    //-----------------------------------------------------
    //-----------------------------------------------------		
	/// Interface for getting a Q-Value
	virtual double getValue(CStateCollection *state, CAction *action, CActionData *data = NULL) 
	{
		CState* currState = state->getState();
		
		int currIter = currState->getDiscreteState(0);
		double margin = currState->getContinuousState(0);
		
		double retVal = 0.0;
		
		for( int i=0; i<_featureNumber; ++i )
		{
			retVal += _rbfs[action][currIter][i].getValue(margin);  // lehet, hogy eggyet hozza kell adni a currIter-hez
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
	
    //-----------------------------------------------------
    //-----------------------------------------------------		
	virtual void updateValue(CStateCollection *state, CAction *action, double td, CActionData * = NULL)
    {
        CState* currState = state->getState();
        int currIter = currState->getDiscreteState(0);
		double margin = currState->getContinuousState(0);

        vector<RBF>& rbfs = _rbfs[action][currIter];
        int numCenters = rbfs.size();//_rbfs[currIter][action].size();
        
        
        for (int i = 0; i < numCenters; ++i) {
            
            double alpha = rbfs[i].getAlpha();
            double mean = rbfs[i].getMean();
            double sigma = rbfs[i].getSigma();

            double distance = margin - mean;
            double rbfValue = rbfs[i].getActivationFactor(margin);
//            double qValue = this->getValue(state, action);
            
            double alphaGrad = rbfValue;
            double meanGrad = rbfValue * alpha * distance / (sigma*sigma);
            double sigmaGrad = rbfValue * alpha * distance * distance / (sigma*sigma*sigma);        
            
            //update the center and shape
            rbfs[i].setAlpha(alpha + td * alphaGrad );
            rbfs[i].setMean(mean + td * meanGrad );
            rbfs[i].setSigma(sigma + td * sigmaGrad );
        }
    }
    
    virtual void getGradient(CStateCollection *state, CAction *action, vector<vector<double> >& gradient)
    {
        CState* currState = state->getState();
        int currIter = currState->getDiscreteState(0);
		double margin = currState->getContinuousState(0);
        
        vector<RBF>& rbfs = _rbfs[action][currIter];
        int numCenters = rbfs.size();//_rbfs[currIter][action].size();
        
        gradient.clear();
        gradient.resize(numCenters);
        
        for (int i = 0; i < numCenters; ++i) {
            
            double alpha = rbfs[i].getAlpha();
            double mean = rbfs[i].getMean();
            double sigma = rbfs[i].getSigma();
            
            double distance = margin - mean;
            double rbfValue = rbfs[i].getActivationFactor(margin);

            double alphaGrad = rbfValue;
            double meanGrad = rbfValue * alpha * distance / (sigma*sigma);
            double sigmaGrad = rbfValue * alpha * distance * distance / (sigma*sigma*sigma);        
            
            gradient[i].resize(3);
            gradient[i][0] = alphaGrad;
            gradient[i][1] = meanGrad;
            gradient[i][2] = sigmaGrad;
        }
    }
    
    //TODO:saveQTable and saveQActionTable
	virtual void saveQTable( const char* fname )
	{
		FILE* outFile = fopen( fname, "w" );
		
		for (CActionSet::iterator it=_actions->begin(); it != _actions->end(); ++it)
		{
			fprintf(outFile,"%d ", dynamic_cast<MultiBoost::CAdaBoostAction*>(*it)->getMode() );
		}
		fprintf(outFile,"\n");
		
		for (CActionSet::iterator it=_actions->begin(); it != _actions->end(); ++it)		
		{
			vector<vector<RBF> >& currentRBFs = _rbfs[*it];
			for(int i=0; i<currentRBFs.size(); ++i)
			{				
				fprintf( outFile, "%d ", i );
				for(int j=0; j<currentRBFs[i].size(); ++j )
				{
					fprintf( outFile, "%g %g %g ", currentRBFs[i][j].getAlpha(),
												   currentRBFs[i][j].getMean(), 
												   currentRBFs[i][j].getSigma() );
				}
				fprintf( outFile, "\n" );
			}
		}
				
		fclose( outFile );
	}
    	
	CAbstractQETraces* getStandardETraces()
	{		
		return new RBFQETraces(this);
	}
    
    //-----------------------------------------------------
    //-----------------------------------------------------	
    void exploreInit()
    {
        CActionSet::iterator it=_actions->begin();
        for(;it!=_actions->end(); ++it )
        {				
			int mode = dynamic_cast<MultiBoost::CAdaBoostAction*>(*it)->getMode();
			
            int iterationNumber = _rbfs[*it].size();
            for( int i=0; i<iterationNumber; ++i)
            {                
                int numFeat = _rbfs[*it][i].size();
				for (int j = 0; j < numFeat; ++j) {
					if (mode == 0) {
						_rbfs[*it][i][j].setMean((j+1) * 1./(numFeat+1));
						_rbfs[*it][i][j].setSigma(1.0 / (2*numFeat));
						_rbfs[*it][i][j].setAlpha(100.0);
					}
                    else {
						_rbfs[*it][i][j].setMean((j+1) * 1./(numFeat+1));
						_rbfs[*it][i][j].setSigma(1.0 / (2*numFeat));
						_rbfs[*it][i][j].setAlpha(1.0);
                    }
                }
            }
        }           
    }
	
	
};

#endif