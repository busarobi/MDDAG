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

//---------------------------------------------------------------------
//---------------------------------------------------------------------
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

	virtual void addMean( double m ) { _mean += m; }
	virtual void addSigma( double s ) { _sigma += s; }
	virtual void addAlpha( double a ) { _alpha += a; }	
	
	virtual double getValue( double x ) const
	{ 
		double retVal = _alpha * getActivationFactor(x);
		return retVal;
	}
    
    virtual double getActivationFactor( double x ) const
	{ 
		double retVal = exp( - (( x-_mean )*( x-_mean ))/(2*_sigma*_sigma));
		return retVal;
	}
    
	virtual string& getID() { return _ID;}
	virtual void setID( const string ID ) { _ID = ID; }
	
	virtual void getGradient( double x, vector<double>& gradient )
	{		
		double distance = x - _mean;
		double rbfValue = this->getActivationFactor(x);
		
		double alphaGrad = rbfValue;
		double meanGrad = rbfValue * _alpha * distance / (_sigma*_sigma);
		double sigmaGrad = rbfValue * _alpha * distance * distance / (_sigma*_sigma*_sigma);        
		
		gradient.resize(3);
		gradient[0] = alphaGrad;
		gradient[1] = meanGrad;
		gradient[2] = sigmaGrad;						
	}
};

//---------------------------------------------------------------------
//---------------------------------------------------------------------
template <typename T>
class AbstractArrayFunction
{
protected:
	vector<T> _data;
	int _size;
public:
	AbstractArrayFunction() : _data(0), _size(0) {}
	
	void resize( int size )
	{
		_data.resize( size );
		_size = size;				
	}
	
	virtual void initUniformly( double coeff ) = 0;
	virtual double getValue( double x ) = 0;	
	virtual void getGradient( double x, vector<vector< double > >& gradient ) = 0;
	virtual void updateParameters( vector< vector< double > >& updates ) = 0;	
	virtual void toString( string& str ) = 0;
	
};

//---------------------------------------------------------------------
//---------------------------------------------------------------------
class RBFArray : public AbstractArrayFunction<RBF>
{
protected:	
	
public:
	RBFArray() : AbstractArrayFunction<RBF>() {}
	
	virtual ~RBFArray() {}
	
	virtual void initUniformly( double coeff )
	{
		for (int j = 0; j < _size; ++j) 
		{
			_data[j].setAlpha( coeff );
			_data[j].setMean((j+1) * 1.0/(_size+1));			
			_data[j].setSigma(1./ (2*_size));
			
		}
	}
	
	virtual void getGradient( double x, vector<vector< double > >& gradient )
	{
		gradient.resize(_size);
		for(int i=0; i < _size; ++i )
		{
			_data[i].getGradient( x, gradient[i] );
		}
	}
						
	virtual double getValue( double x )	
	{
		double retVal = 0.0;
		
		for( int i=0; i<_size; ++i )
		{
			retVal += _data[i].getValue(x); 
		}		
		return retVal;		
	}
	
	virtual void updateParameters( vector< vector< double > >& updates )
	{
		assert(_size == updates.size());
		for( int i=0; i<_size; ++i )
		{
			_data[i].addAlpha(updates[i][0]);
			_data[i].addMean(updates[i][1]);
			_data[i].addSigma(updates[i][2]);
		}		
	}

	virtual void toString( string& str )
	{
		stringstream ss("");
		for( int i=0; i<_size; ++i )
		{
			ss << _data[i].getAlpha() << " ";
			ss << _data[i].getMean() << " ";
			ss << _data[i].getSigma() << " ";
		}				
		ss.str( str );
	}	
	
};


//---------------------------------------------------------------------
//---------------------------------------------------------------------
class RBFBasedQFunctionBinary : public CAbstractQFunction // CAbstractQFunction
{
protected:	
	map< CAction*, vector<vector<RBF> > > 	_rbfs;
	int _featureNumber;	
	CActionSet* _actions;
	int _numberOfActions;
    int _numberOfIterations;
    
public:
	// constructor
	RBFBasedQFunctionBinary(CActionSet *actions, CStateModifier* statemodifier );
    
	// initilizes the parameters of RBFs
    virtual void uniformInit(double* init=NULL);
    			
	virtual ~RBFBasedQFunctionBinary() {}
        
	
	/// Interface for getting a Q-Value
	virtual double getValue(CStateCollection *state, CAction *action, CActionData *data = NULL);

	
	// update the RBF-s
	virtual void updateValue(CStateCollection *state, CAction *action, double td, vector<vector<double> >& eTraces);

    
	//clacluates the gradient
    virtual void getGradient(CStateCollection *state, CAction *action, vector<vector<double> >& gradient);
    virtual void getGradient(double margin, int currIter, CAction *action, vector<vector<double> >& gradient );
	
    	
	// creates an Etrace object
	CAbstractQETraces* getStandardETraces();
    
	
	// IO
	virtual void saveQTable( const char* fname );
    void saveActionValueTable(FILE* stream);
    void saveActionTable(FILE* stream);
};



//---------------------------------------------------------------------
//---------------------------------------------------------------------
template< typename T=RBFArray >
class ArrayBasedQFunctionBinary : public CAbstractQFunction // CAbstractQFunction
{
protected:	
	map< CAction*, vector<T> > 	_data;
	
	int _featureNumber;	
	CActionSet* _actions;
	int _numberOfActions;
    int _numberOfIterations;	
public:
	ArrayBasedQFunctionBinary(CActionSet *actions, CStateModifier* statemodifier ) : CAbstractQFunction(actions)
	{
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
			_data[*it].resize( iterationNumber );
			for( int i=0; i<iterationNumber; ++i)
			{
				_data[*it][i].resize(_featureNumber);
			}
		}
		
	}	
	//------------------------------------------------------
	//------------------------------------------------------    
	CAbstractQETraces* getStandardETraces();
//	{
//		return new RBFQETraces(this);
//	}
	
	//------------------------------------------------------
	//------------------------------------------------------    
	double getValue(CStateCollection *state, CAction *action, CActionData *data) 
	{
		CState* currState = state->getState();
		
		int currIter = currState->getDiscreteState(0);
		double margin = currState->getContinuousState(0);
		
		double retVal = _data[action][currIter].getValue(margin);		
		return retVal;
	}

	//------------------------------------------------------
	//------------------------------------------------------    
	void updateValue(CStateCollection *state, CAction *action, double td, vector<vector<double> >& eTraces)
	{
		CState* currState = state->getState();
		int currIter = currState->getDiscreteState(0);
		double margin = currState->getContinuousState(0);
		
		//AbstractArrayFunction<T>& currData = _data[action][currIter];
		int numCenters = eTraces.size();//_rbfs[currIter][action].size();
		
		//        assert(numCenters = eTraces.size());
		
		vector< vector< double > > updateSteps( numCenters );		
		for (int i = 0; i < numCenters; ++i) 
		{
			double th = 0.01;
			
			//update the center and shape
			vector<double>& currentETrace = eTraces[i];
			updateSteps[i].resize( currentETrace.size());
			
			for( int j=0; j<currentETrace.size(); ++j )
			{
				double step = currentETrace[j] * td;

				step = (step>th) ?  th : step;		
				step = (step<-th) ? -th : step;		
								
				updateSteps[i][j]=step;
			}
		}
		_data[action][currIter].updateParameters( updateSteps );
	}
	//------------------------------------------------------
	//------------------------------------------------------    
	void getGradient(CStateCollection *state, CAction *action, vector<vector<double> >& gradient)
	{
		CState* currState = state->getState();
		int currIter = currState->getDiscreteState(0);
		double margin = currState->getContinuousState(0);
		
		getGradient(margin, currIter, action, gradient);
	}
	
	
	//------------------------------------------------------
	//------------------------------------------------------    
	void getGradient(double margin, int currIter, CAction *action, vector<vector<double> >& gradient )
	{		
		_data[action][currIter].getGradient( margin, gradient );		
	}

	//------------------------------------------------------
	//------------------------------------------------------    	
	void saveQTable( const char* fname )
	{
		FILE* outFile = fopen( fname, "w" );		
		for (CActionSet::iterator it=_actions->begin(); it != _actions->end(); ++it)		
		{
			vector<T> currentData = _data[*it];
			//cout << currentRBFs.size() << endl;
			for(int i=0; i<currentData.size(); ++i)
			{				
				fprintf( outFile, "%d ", i );
				string tmpString("");
				currentData[i].toString(tmpString);
				fprintf(outFile, "%s", tmpString.c_str());
				fprintf( outFile, "\n" );
			}
		}
		
		fclose( outFile );
	}
	//------------------------------------------------------
	//------------------------------------------------------    
	void uniformInit(vector<double>& init)
	{
		CActionSet::iterator it=_actions->begin();
		for(;it!=_actions->end(); ++it )
		{	
			int index = dynamic_cast<MultiBoost::CAdaBoostAction*>(*it)->getMode();
			
			double initAlpha = 0;
			if ( init.size() > 0 ) {
				//warning  : no check on the bounds of init                
				initAlpha = init[index];
			}
			
			int iterationNumber = _data[*it].size();
			for( int i=0; i<iterationNumber; ++i)
			{                
				_data[*it][i].initUniformly(initAlpha);
			}
		}           
	}
	
	//------------------------------------------------------
	//------------------------------------------------------    
	void saveActionValueTable(FILE* stream){}
	void saveActionTable(FILE* stream){}		
};


#endif