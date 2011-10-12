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
//#include "RBFQETraces.h"
#include "AdaBoostMDPClassifierAdv.h"
#include "cfeaturefunction.h"

class RBFQETraces;

class RBF {
protected:
	double _mean;
	double _sigma;
	double _alpha;
	string _ID;
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
    
	virtual string& getID() { return _ID;}
	virtual void setID( const string ID ) { _ID = ID; }
};


class RBFBasedQFunctionBinary : public CAbstractQFunction // CAbstractQFunction
{
protected:
//	vector< map< CAction*, vector<RBF> > >	_rbfs;
	map< CAction*, vector<vector<RBF> > > 	_rbfs;
	int _featureNumber;	
	CActionSet* _actions;
	int _numberOfActions;
    int _numberOfIterations;
    
    double _muAlpha;
    double _muMean;
    double _muSigma;
public:
	RBFBasedQFunctionBinary(CActionSet *actions, CStateModifier* statemodifier ) : CAbstractQFunction(actions)
	{
        //CGradientQFunction ancestor init
        //addType(GRADIENTQFUNCTION);        
        //this->localGradientQFunctionFeatures = new CFeatureList();
        
		// the statemodifier must be RBFStateModifier
		RBFStateModifier* smodifier = dynamic_cast<RBFStateModifier*>( statemodifier );
		
		const int iterationNumber = smodifier->getNumOfIterations();
		const int featureNumber = smodifier->getNumOfRBFsPerIteration();
		const int numOfClasses = smodifier->getNumOfClasses();
		
		_featureNumber = featureNumber;
		_numberOfIterations = iterationNumber;
        
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
	}
    
    void uniformInit(double* init=NULL)
    {
        CActionSet::iterator it=_actions->begin();
        for(;it!=_actions->end(); ++it )
        {	
			int index = dynamic_cast<MultiBoost::CAdaBoostAction*>(*it)->getMode();
			
            double initAlpha = 0;
            if (init != NULL) {
                //warning  : no check on the bounds of init                
                initAlpha = init[index];
            }
            
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

                    _rbfs[*it][i][j].setAlpha(initAlpha);
                    _rbfs[*it][i][j].setSigma(1./ (2*numFeat));
					
					stringstream tmpString("");
					tmpString << "[ac_" << index << "|it_" << i << "|fn_" << j << "]";
					_rbfs[*it][i][j].setID( tmpString.str() );
                }
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
				num += _rbfs[*it][i].size();
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
		
		vector<RBF>& currRBFs = _rbfs[action][currIter];		
		for( int i=0; i<_featureNumber; ++i )
		{
			retVal += currRBFs[i].getValue(margin);  // lehet, hogy eggyet hozza kell adni a currIter-hez
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
	virtual void updateValue(CStateCollection *state, CAction *action, double td, vector<vector<double> >& eTraces)
    {
        CState* currState = state->getState();
        int currIter = currState->getDiscreteState(0);
		double margin = currState->getContinuousState(0);

        vector<RBF>& rbfs = _rbfs[action][currIter];
        int numCenters = rbfs.size();//_rbfs[currIter][action].size();
        
//        assert(numCenters = eTraces.size());
        
        for (int i = 0; i < numCenters; ++i) {
            
            double alpha = rbfs[i].getAlpha();
            double mean = rbfs[i].getMean();
            double sigma = rbfs[i].getSigma();

            //update the center and shape
			vector<double>& currentGradient = eTraces[i];
            rbfs[i].setAlpha(alpha + currentGradient[0] * td * _muAlpha );
            rbfs[i].setMean(mean + currentGradient[1] * td * _muMean );
            rbfs[i].setSigma(sigma + currentGradient[2] * td * _muSigma );
#ifdef RBFDEB			
			cout << rbfs[i].getID() << " ";
#endif
        }		
    }
    
//    virtual void getGradient(CStateCollection *state, CAction *action, vector<vector<double> >& eTraces)
//    {
//        CState* currState = state->getState();
//        int currIter = currState->getDiscreteState(0);
//		double margin = currState->getContinuousState(0);
//        
//        vector<RBF>& rbfs = _rbfs[action][currIter];
//        int numCenters = rbfs.size();//_rbfs[currIter][action].size();
//        
//        eTraces.clear();
//        eTraces.resize(numCenters);
//        
//        for (int i = 0; i < numCenters; ++i) {
//            
//            double alpha = rbfs[i].getAlpha();
//            double mean = rbfs[i].getMean();
//            double sigma = rbfs[i].getSigma();
//            
//            double distance = margin - mean;
//            double rbfValue = rbfs[i].getActivationFactor(margin);
//
//            double alphaGrad = rbfValue;
//            double meanGrad = rbfValue * alpha * distance / (sigma*sigma);
//            double sigmaGrad = rbfValue * alpha * distance * distance / (sigma*sigma*sigma);        
//            
//            eTraces[i].resize(3);
//            eTraces[i][0] = alphaGrad;
//            eTraces[i][1] = meanGrad;
//            eTraces[i][2] = sigmaGrad;
//        }
//    }

    virtual void getGradient(CStateCollection *state, CAction *action, vector<vector<double> >& gradient)
    {
        CState* currState = state->getState();
        int currIter = currState->getDiscreteState(0);
		double margin = currState->getContinuousState(0);
		
        getGradient(margin, currIter, action, gradient);
    }

    virtual void getGradient(double margin, int currIter, CAction *action, vector<vector<double> >& gradient)
    {
//        CState* currState = state->getState();
//        int currIter = currState->getDiscreteState(0);
//		double margin = currState->getContinuousState(0);
        
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
		
		for(int i=0; i<_rbfs.size(); ++i)
		{
			for (CActionSet::iterator it=_actions->begin(); it != _actions->end(); ++it)
			{
			}
		}
				
		fclose( outFile );
	}
    	
	CAbstractQETraces* getStandardETraces();
    
    void saveActionValueTable(FILE* stream)
    {
        fprintf(stream, "Q-FeatureActionValue Table\n");
        CActionSet::iterator it;
        
        for (int i = 0; i < _featureNumber; ++i) {
            for (int j = 0; j < _numberOfIterations; ++j) {
                fprintf(stream,"Feature %d: ", i);
                for(it =_actions->begin(); it!=_actions->end(); ++it ) {
                    fprintf(stream,"%f %f %f ", _rbfs[*it][j][i].getAlpha(), _rbfs[*it][j][i].getMean(), _rbfs[*it][j][i].getSigma());
                }
                fprintf(stream, "\n");
            }
        }
    }
    
    void saveActionTable(FILE* stream)
    {
        fprintf(stream, "Q-FeatureAction Table\n");
        double max = 0.0;
        int maxIndex = 0;
        
        CActionSet::iterator it;
        for (int i = 0; i < _featureNumber; ++i) {
            for (int j = 0; j < _numberOfIterations; ++j) {
                fprintf(stream,"Feature %d: ", i);
                
                it =_actions->begin();
                max = _rbfs[*it][j][i].getAlpha();
                maxIndex = 0;
                ++it;
                
                for(; it!=_actions->end(); ++it ) {
                    double v = _rbfs[*it][j][i].getAlpha();
                    if (max < v) {
                        max = v;
                        maxIndex = dynamic_cast<MultiBoost::CAdaBoostAction*>(*it)->getMode();
                    }
                }
                
                fprintf(stream,"%d ", maxIndex);
                fprintf(stream, "\n");
            }
        }
    }
};

#endif