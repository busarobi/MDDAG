/*
 *  ArrayBasedQETraces.h
 *  RLTools
 *
 *  Created by Robert Busa-Fekete on 10/13/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef _ARRAYBASEDQETRACES_H___
#define _ARRAYBASEDQETRACES_H___


//#define RBFDEB

#include "cqetraces.h"
#include "AdaBoostMDPClassifierAdv.h"
#include "RBFBasedQFunction.h"
#include <vector>
#include <list>

//class RBFBasedQFunctionBinary;
using namespace std;

typedef vector<vector<double> > OneIterETrace;
typedef list<OneIterETrace > CustomETrace;
typedef CustomETrace::iterator ETIterator;
typedef CustomETrace::reverse_iterator ETReverseIterator;



template< typename T >
class ArrayBasedQETraces : public CAbstractQETraces 
{
protected:
	list<double> _margins;
	list<int>	 _iters;
	
	list<CAction*> _actions;
    CustomETrace _eTraces; //strange ETrace struture, due to our peculiar state representation
	CStateCollection* _currentState;
public:
	
	// constructor
	ArrayBasedQETraces(CAbstractQFunction *qFunction) : CAbstractQETraces(qFunction) {
        addParameter("ETraceTreshold", 0.001);
    }
	virtual ~ArrayBasedQETraces() {};
	
	
	/// Interface function for reseting the ETraces
	virtual void resetETraces()
	{
		_actions.clear();        
		_eTraces.clear();
		_iters.clear();
		_margins.clear();
		
#ifdef RBFDEB		
		cout << "------------------------------------------------" << endl;
#endif		
	}
	
	
	/// Interface function for updating the ETraces
	/**
	 I.e. all stored ETraces-factors get multiplied by lambda * gamma, gamma is taken from the Q-Function. 
	 If the action is a multistep action, the factor lambda * gamma is exponentiated with the duration.
	 */
	virtual void updateETraces(CAction *action, CActionData *data = NULL)
	{
		double mult = getParameter("Lambda") * getParameter("DiscountFactor");
		ETIterator eIt = _eTraces.begin();
		
		list<CAction*>::iterator actionIt = _actions.begin();
		
		list<double>::iterator itMargin = _margins.begin();
		list<int>::iterator itIters = _iters.begin();		
		
		// later should be done for acceleratiung the E-trace update
		double treshold = getParameter("ETraceTreshold");
		int i = 0;
		
		while (eIt != _eTraces.end())
		{
			OneIterETrace & oneItEtrace = *eIt;
			OneIterETrace gradient;
			
			dynamic_cast<ArrayBasedQFunctionBinary<T>* >(qFunction)->getGradient(*itMargin, *itIters, *actionIt, gradient);
			
			for (int j = 0; j < oneItEtrace.size(); ++j) {
				for (int k = 0; k < oneItEtrace[j].size(); ++k) {
					oneItEtrace[j][k] = oneItEtrace[j][k] * mult ;
					oneItEtrace[j][k] +=  gradient[j][k];
				}
			}
			
			++eIt;
			++actionIt;
			++itMargin;
			++itIters;
		}
	}
	
	
	/// Interface function for adding a State-Action pair with the given factor to the ETraces
	virtual void addETrace(CStateCollection *state, CAction *action, double factor = 1.0, CActionData *data = NULL)
	{
		_currentState = state;
		_actions.push_back( action );
		
		OneIterETrace gradient;
		dynamic_cast<ArrayBasedQFunctionBinary<T>* >(qFunction)->getGradient(state, action, gradient);
		
		_eTraces.push_back(gradient);
		
		CState* currState = state->getState();
		int currIter = currState->getDiscreteState(0);
		_iters.push_back(currIter);
		double margin = currState->getContinuousState(0);
		_margins.push_back(margin);		
	}
	
	/// Interface function for updating the Q-Values of all State-Action Pairs in the ETraces
	virtual void updateQFunction(double td)
	{		
		list<CAction*>::iterator invitAction = _actions.begin();
		ETIterator invitTrace = _eTraces.begin();
		
		list<double>::iterator itMargin = _margins.begin();
		list<int>::iterator itIters = _iters.begin();		
		
#ifdef RBFDEB				
		int index = dynamic_cast<MultiBoost::CAdaBoostAction* >(*(_actions.rbegin()))->getMode();
		cout << "Action: " << index << " TD " << td << endl;
#endif
		
		for (; itMargin != _margins.end(); ++itMargin, ++itIters, ++invitAction, ++invitTrace)
		{
			CAction* currentAction = *invitAction;
			OneIterETrace& currentETrace = *invitTrace;
			
			
			double currMargin = *itMargin;
			int currIter = *itIters;
			
			CState* artificialState = _currentState->getState()->clone();
			artificialState->setContinuousState(0, currMargin);
			artificialState->setDiscreteState(0, currIter);
			
#ifdef RBFDEB					
			cout << "(A:" << dynamic_cast<MultiBoost::CAdaBoostAction* >(*(invitAction))->getMode() << "," << flush; 
			cout << "M: " << currMargin << ",";
			cout << "I: " << currIter << ",";
			cout << "O:" << dynamic_cast<ArrayBasedQFunctionBinary<T>** >(qFunction)->getValue(artificialState, currentAction)<< "," << flush;
#endif			
			dynamic_cast<ArrayBasedQFunctionBinary<RBFArray>* >(qFunction)->updateValue(artificialState, currentAction, td, currentETrace);
#ifdef RBFDEB					
			cout << "N:" << dynamic_cast<ArrayBasedQFunctionBinary<T>** >(qFunction)->getValue(artificialState, currentAction) << ")" << endl << flush;
#endif			
			delete artificialState;
		}
#ifdef RBFDEB				
		cout << endl;
		if (index==2) cout << "End of episode" << endl;
#endif		
	}
	
	
};



#endif