/*
 *  ArrayBasedQFunctionBinary.h
 *  RLTools
 *
 *  Created by Robert Busa-Fekete on 10/13/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef __ARRAYBASEDQFUNCTIONBINARY_H__
#define __ARRAYBASEDQFUNCTIONBINARY_H__

#include <string>
#include "cqfunction.h"
#include "cqetraces.h"
#include "cgradientfunction.h"
#include "cstatemodifier.h"
#include "RBFStateModifier.h"
//#include "RBFQETraces.h"
#include "AdaBoostMDPClassifierAdv.h"
#include "cfeaturefunction.h"

using namespace std;

//-----------------------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------------
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

	virtual double getValue( vector<double>& x ) const { return getValue(x[0]); }	
	virtual double getValue( double x ) const
	{ 
		double retVal = _alpha * getActivationFactor(x);
		return retVal;
	}
    
	virtual double getActivationFactor( vector<double>&  x ) const { return getActivationFactor(x[0]); }
    virtual double getActivationFactor( double x ) const
	{ 
		double retVal = exp( - (( x-_mean )*( x-_mean ))/(2*_sigma*_sigma));
		return retVal;
	}
    
	virtual string& getID() { return _ID;}
	virtual void setID( const string ID ) { _ID = ID; }
	
	virtual void getGradient( double x, vector<double>& gradient )
	{		
		double diff = x - _mean;
		double rbfValue = this->getActivationFactor(x);
		
		double alphaGrad = rbfValue;
		double meanGrad = rbfValue * _alpha * diff / (_sigma*_sigma);
		double sigmaGrad = rbfValue * _alpha * diff * diff / (_sigma*_sigma*_sigma);        
		
		gradient.resize(3);
		gradient[0] = alphaGrad;
		gradient[1] = meanGrad;
		gradient[2] = sigmaGrad;						
	}
};
//-----------------------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------------
class MultiDimRBFSingleSigma
{
protected:
	vector<double> _mean;
	vector<double> _sigma;
	double _alpha;
	string _ID;	
public:	
	MultiDimRBFSingleSigma() : _mean(0), _sigma(0), _alpha(0) {}
	virtual ~MultiDimRBFSingleSigma() {}
	
	virtual vector<double>& getMean() { return _mean; }
	virtual vector<double>& getSigma() { return _sigma; }
	virtual double getAlpha() { return _alpha; }
	
	virtual void setMean( vector<double>& m ) 
	{
		_mean.resize(m.size());
		copy( _mean.begin(), _mean.end(), m.begin() ); 
	}
	
	virtual void setSigma( vector<double>& s )
	{
		_sigma.resize(s.size());
		copy( _sigma.begin(), _sigma.end(), s.begin() ); 
	}
	
	virtual void setAlpha( double a ) { _alpha = a; }
	
	virtual void addMean( vector<double>& m ) 
	{ 
		for(int i=0; i< _mean.size(); ++i ) _mean[i] += m[i]; 
	}
	virtual void addSigma( vector<double>& s )
	{ 
		for(int i=0; i< _sigma.size(); ++i ) _sigma[i] += s[i]; 
	}
	
	virtual void addAlpha( double a ) { _alpha += a; }	
	
	virtual double getValue( vector<double>& x )
	{ 
		double retVal = _alpha * getActivationFactor(x);
		return retVal;
	}
    virtual double getValue( double x )
	{
		cout << "getValue: Multidimensional RBF is called with a single values" << endl;
		exit(-1);
	}
	
	virtual double getActivationFactor( double x ) const
	{
		cout << "getActivationFactor: Multidimensional RBF is called with a single values" << endl;
		exit(-1);
	}
	
	virtual double getActivationFactor( vector<double>&  x ) const
	{ 
		double retVal = 0.0;
		for(int i=0; i< _mean.size(); ++i ) retVal += ((_mean[i] - x[i])*(_mean[i] - x[i]) / (2*_sigma[0]*_sigma[0]) ); 
		retVal = exp( - retVal);
		return retVal;
	}
    
	virtual string& getID() { return _ID;}
	virtual void setID( const string ID ) { _ID = ID; }
	
	virtual void getGradient( double x, vector<double>& gradient )
	{
		cout << "getGradient: Multidimensional RBF is called with a single values" << endl;
		exit(-1);
	}
	
	virtual void getGradient( vector<double>& x, vector<double>& gradient )
	{		

		double dsquare = 0.0;
		for(int i=0; i< _mean.size(); ++i )
		{
			dsquare += ((_mean[i] - x[i])*(_mean[i] - x[i]));
		}
		
		double rbfValue = this->getActivationFactor(x);
		
		double alphaGrad = rbfValue;
		
		vector<double> meanGrad(_mean.size());
		for (int i=0; i<_mean.size(); ++i )
		{
			meanGrad[i] = rbfValue * _alpha * (x[i]-_mean[i]) / (_sigma[0]*_sigma[0]);
		}
		
		double sigmaGrad = rbfValue * _alpha * dsquare / (_sigma[0]*_sigma[0]*_sigma[0]);        
		
		gradient.resize(_mean.size()+2);
		gradient[0] = alphaGrad;
		for (int i=0; i<_mean.size(); ++i ) gradient[i+1] = meanGrad[i];
		gradient[meanGrad.size()+1] = sigmaGrad;						
	}
	
};


//-----------------------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------------
class RBFLogScaled : public RBF
{   
public:
    virtual double getActivationFactor( double x ) const
	{ 
		double retVal = exp( - (( x-_mean )*( x-_mean ))/(2*exp(_sigma)*exp(_sigma)));
		return retVal;
	}

	virtual void getGradient( double x, vector<double>& gradient )
	{		
		double distance = x - _mean;
		double rbfValue = this->getActivationFactor(x);
		
		double alphaGrad = rbfValue;
		double meanGrad = rbfValue * _alpha * distance / (exp(_sigma)*exp(_sigma));
		double sigmaGrad = rbfValue * _alpha * distance * distance / (2*exp(_sigma*_sigma)*exp(_sigma*_sigma));        
		
		gradient.resize(3);
		gradient[0] = alphaGrad;
		gradient[1] = meanGrad;
		gradient[2] = sigmaGrad;						
	}
};

//-----------------------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------------
template <typename T>
class AbstractArrayFunction
{
protected:
	vector<T> _data;
	int _size;
	int _dimension;
public:
	AbstractArrayFunction() : _data(0), _size(0) {}
	
	void resize( int size )
	{
		_data.resize( size );
		_size = size;				
	}
	
	virtual int getDimension() const { return _dimension; }
	virtual void setDimension( int dim ) { _dimension = dim; }
	
	
	virtual void initUniformly( double coeff ) = 0;
	virtual double getValue( double x ) = 0;	
	virtual double getValue( vector<double>& x ) = 0;		
	virtual void getGradient( double x, vector<vector< double > >& gradient ) = 0;
	virtual void getGradient( vector<double>& x, vector<vector< double > >& gradient ) = 0;
	virtual void updateParameters( vector< vector< double > >& updates ) = 0;	
	virtual void toString( string& str ) = 0;
	
};

template <typename T>
class MultiDimAbstractArrayFunction : virtual public AbstractArrayFunction<T>
{
public:
	virtual double getValue( double x ) { exit(-1); }
	virtual void getGradient( double x, vector<vector< double > >& gradient ) {exit(-1); }
};

//-----------------------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------------
template< typename TF = RBF >
class RBFArray : public AbstractArrayFunction<TF>
{
protected:	
	
public:
	RBFArray() : AbstractArrayFunction<TF>() {}
	
	virtual ~RBFArray() {}
	
	virtual void initUniformly( double coeff )
	{
		for (int j = 0; j < this->_size; ++j) 
		{
			this->_data[j].setAlpha( coeff );
			this->_data[j].setMean((j+1) * 1.0/(this->_size+1));			
			this->_data[j].setSigma(1./ (2*this->_size));
			
		}
	}
	
	virtual void getGradient( vector<double>& x, vector<vector< double > >& gradient )
	{
		getGradient( x[0], gradient );
	}
	
	virtual void getGradient( double x, vector<vector< double > >& gradient )
	{
		gradient.resize(this->_size);
		for(int i=0; i < this->_size; ++i )
		{
			this->_data[i].getGradient( x, gradient[i] );
		}
	}
	
	virtual double getValue( vector<double>& x ) { return getValue(x[0]); }
	
	virtual double getValue( double x )	
	{
		double retVal = 0.0;
		
		for( int i=0; i<this->_size; ++i )
		{
			retVal += this->_data[i].getValue(x); 
		}		
		return retVal;		
	}
	
	virtual void updateParameters( vector< vector< double > >& updates )
	{
		double th=0.001;
		assert(this->_size == updates.size());
		for( int i=0; i<this->_size; ++i )
		{
			this->_data[i].addAlpha(updates[i][0]);

			double step = updates[i][1];
			step = (step>th) ?  th : step;		
			step = (step<-th) ? -th : step;		
			this->_data[i].addMean(step);

			step = updates[i][2];
			step = (step>th) ?  th : step;		
			step = (step<-th) ? -th : step;					
			this->_data[i].addSigma(step);
		}		
	}
	
	virtual void toString( string& str )
	{
		stringstream ss("");
		for( int i=0; i<this->_size; ++i )
		{
			ss << this->_data[i].getAlpha() << " ";
			ss << this->_data[i].getMean() << " ";
			ss << this->_data[i].getSigma() << " ";
		}				
		str=ss.str();
	}	
	
};

//-----------------------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------------
template< typename TF = MultiDimRBFSingleSigma >
class MultiDimRBFArraySingleSigma : virtual public  MultiDimAbstractArrayFunction<TF>
{
protected:	
	
public:
	MultiDimRBFArraySingleSigma() : AbstractArrayFunction<TF>() {}
	
	virtual ~MultiDimRBFArraySingleSigma() {}
	
	virtual void initUniformly( double coeff )
	{
		for (int j = 0; j < this->_size; ++j) 
		{
			this->_data[j].setAlpha( coeff );
			vector<double> mean(this->_dimension, 0.0);
			for(int i=0; i< this->_dimension; ++i ) mean[i] = (double) rand() / (double) RAND_MAX;
			this->_data[j].setMean(mean);	
			vector<double> sigma(1,1./ (2*this->_size));
			this->_data[j].setSigma(sigma);
			
		}
	}
	// i don't know why this function has to be redefined here
	virtual double getValue( double x ) { exit(-1); }	
	virtual void getGradient( double x, vector<vector< double > >& gradient ) {exit(-1); }
	
	
	virtual void getGradient( vector<double>& x, vector<vector< double > >& gradient )
	{
		gradient.resize(this->_size);
		for(int i=0; i < this->_size; ++i )
		{
			this->_data[i].getGradient( x, gradient[i] );
		}
	}
	
	virtual double getValue( vector<double>& x )
	{
		double retVal = 0.0;
		
		for( int i=0; i<this->_size; ++i )
		{
			retVal += this->_data[i].getValue(x); 
		}		
		return retVal;		
	}
	
	virtual void updateParameters( vector< vector< double > >& updates )
	{
		double th=0.001;
		assert(this->_size == updates.size());
		for( int i=0; i<this->_size; ++i )
		{
			this->_data[i].addAlpha(updates[i][0]);
			double step =0.0;
			
			vector<double> meanStep(this->_dimension);
			for(int j=0; j < this->_dimension; ++j )
			{
				step = updates[i][j+1];
				step = (step>th) ?  th : step;		
				step = (step<-th) ? -th : step;		
				meanStep[j] = step;
			}	
			this->_data[i].addMean(meanStep);
			
			step = updates[i][this->_dimension+1];
			step = (step>th) ?  th : step;		
			step = (step<-th) ? -th : step;	
			
			vector<double> sigmaStep(1,step);
			this->_data[i].addSigma(sigmaStep);
		}		
	}
	
	virtual void toString( string& str )
	{
		stringstream ss("");
		
		for( int i=0; i<this->_size; ++i )
		{			
			ss << this->_data[i].getAlpha() << " ";
			//vector<int>& mean = this->_data[i].getMean();
			//for( int i=0; i<this->_size; ++i ) ss << mean[i] << " ";
			//ss << this->_data[i].getSigma() << " ";
		}				
		str=ss.str();
	}	
	
};



//-----------------------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------------
template< typename T=RBFArray<RBF> >
class ArrayBasedQFunctionBinary : public CAbstractQFunction // CAbstractQFunction
{
protected:	
	//map< CAction*, vector<T> > 	_data;
	map<int, vector<T> > 	_data;
	
	int _featureNumber;	
	vector<int> _actionIndices;
	CActionSet* _actions;
	int _numberOfActions;
    int _numberOfIterations;
	int _dimension;
public:
	ArrayBasedQFunctionBinary(CActionSet *actions, CStateModifier* statemodifier ) : CAbstractQFunction(actions)
	{
		// the statemodifier must be RBFStateModifier
		RBFStateModifier* smodifier = dynamic_cast<RBFStateModifier*>( statemodifier );
		_dimension = smodifier->getNumOfClasses();
		
		const int iterationNumber = smodifier->getNumOfIterations();
		const int featureNumber = smodifier->getNumOfRBFsPerIteration();
		const int numOfClasses = smodifier->getNumOfClasses();
		
		_featureNumber = featureNumber;
		_numberOfIterations = iterationNumber;
		
		//assert(numOfClasses==1);
		
		_actions = actions;
		_numberOfActions = actions->size();
		
		CActionSet::iterator it=(*actions).begin();
		for(;it!=(*actions).end(); ++it )
		{			
			int currentActionIndex = dynamic_cast<MultiBoost::CAdaBoostAction* >(*it)->getMode();
			_data[currentActionIndex].resize( iterationNumber );
			for( int i=0; i<iterationNumber; ++i)
			{
				_data[currentActionIndex][i].resize(_featureNumber);
				_data[currentActionIndex][i].setDimension( _dimension );
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
		int currentActionIndex = dynamic_cast<MultiBoost::CAdaBoostAction* >(action)->getMode();

		int currIter = currState->getDiscreteState(0);
		double retVal = 0.0;
		
		if (_dimension==1)
		{				
			double margin = currState->getContinuousState(0);		
			retVal = _data[currentActionIndex][currIter].getValue(margin);		
		} else {
			vector<double> margins(_dimension);
			for(int i=0; i<_dimension; ++i )
				margins[i]= currState->getContinuousState(i);		
			retVal = _data[currentActionIndex][currIter].getValue(margins);			
		}
		return retVal;
	}
	
	//------------------------------------------------------
	//------------------------------------------------------    
	void updateValue(CStateCollection *state, CAction *action, double td, vector<vector<double> >& eTraces)
	{
		CState* currState = state->getState();
		int currentActionIndex = dynamic_cast<MultiBoost::CAdaBoostAction* >(action)->getMode();
		
		int currIter = currState->getDiscreteState(0);
		
		//AbstractArrayFunction<T>& currData = _data[action][currIter];
		int numCenters = eTraces.size();//_rbfs[currIter][action].size();
		
		//        assert(numCenters = eTraces.size());
		
		vector< vector< double > > updateSteps( numCenters );		
		for (int i = 0; i < numCenters; ++i) 
		{
			//update the center and shape
			vector<double>& currentETrace = eTraces[i];
			updateSteps[i].resize( currentETrace.size());
			
			for( int j=0; j<currentETrace.size(); ++j )
			{
				double step = currentETrace[j] * td;								
				updateSteps[i][j]=step;
			}
		}
		_data[currentActionIndex][currIter].updateParameters( updateSteps );
	}
	//------------------------------------------------------
	//------------------------------------------------------    
	void getGradient(CStateCollection *state, CAction *action, vector<vector<double> >& gradient)
	{
		CState* currState = state->getState();
		int currentActionIndex = dynamic_cast<MultiBoost::CAdaBoostAction* >(action)->getMode();
		int currIter = currState->getDiscreteState(0);
		
		if (this->_dimension==1)
		{
			double margin = currState->getContinuousState(0);		
			_data[currentActionIndex][currIter].getGradient( margin, gradient );
		} else {
			vector<double> margins(this->_dimension);
			for( int i=0; i<this->_dimension; ++i ) margins[i]=currState->getContinuousState(i);		
			_data[currentActionIndex][currIter].getGradient( margins, gradient );
		}
	}
			
	//------------------------------------------------------
	//------------------------------------------------------    	
	void saveQTable( const char* fname )
	{
		FILE* outFile = fopen( fname, "w" );		
		for (CActionSet::iterator it=_actions->begin(); it != _actions->end(); ++it)		
		{
			int currentActionIndex = dynamic_cast<MultiBoost::CAdaBoostAction* >(*it)->getMode();
			vector<T> currentData = _data[currentActionIndex];
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
			int currentActionIndex = dynamic_cast<MultiBoost::CAdaBoostAction* >(*it)->getMode();
			int index = dynamic_cast<MultiBoost::CAdaBoostAction*>(*it)->getMode();
			
			double initAlpha = 0;
			if ( init.size() > 0 ) {
				//warning  : no check on the bounds of init                
				initAlpha = init[index];
			}
			
			int iterationNumber = _data[currentActionIndex].size();
			for( int i=0; i<iterationNumber; ++i)
			{                
				_data[currentActionIndex][i].initUniformly(initAlpha);
			}
		}           
	}
	
	//------------------------------------------------------
	//------------------------------------------------------    
	void saveActionValueTable(FILE* stream){}
	void saveActionTable(FILE* stream){}		
};
//-----------------------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------------


#endif