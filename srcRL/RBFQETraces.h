/*
 *  RBFQETraces.h
 *  RLTools
 *
 *  Created by Robert Busa-Fekete on 10/5/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */


#ifndef __RBFQETRACES_H
#define __RBFQETRACES_H

//#define RBFDEB

#include "cqetraces.h"
#include "AdaBoostMDPClassifierAdv.h"
//#include "RBFBasedQFunction.h"
#include <vector>
#include <list>

class RBFBasedQFunctionBinary;
using namespace std;


typedef vector<vector<double> > OneIterETrace;
typedef list<OneIterETrace > CustomETrace;
typedef CustomETrace::iterator ETIterator;
typedef CustomETrace::reverse_iterator ETReverseIterator;
//TODO: this structure supposes that we never come back to a state. It's fine for now.
class RBFQETraces : public CAbstractQETraces 
{
protected:
    //@robi : a list is more appropriate since we always iterate, never direct access
    // and we need to remove items that have a small etrace
	list<double> _margins;
	list<int>	 _iters;
	
	
	//list<CStateCollection*> _states;
	list<CAction*> _actions;
    CustomETrace _eTraces; //strange ETrace struture, due to our peculiar state representation
	CStateCollection* _currentState;
public:
	RBFQETraces(CAbstractQFunction *qFunction) : CAbstractQETraces(qFunction) {
        addParameter("ETraceTreshold", 0.001);
    }
    
	virtual ~RBFQETraces() {};
	
    //TODO: adapt to the new structure of _eTraces
	/// Interface function for reseting the ETraces
	virtual void resetETraces() 
	{
//		for(int i=0; i< _states.size(); ++i )
//		{
//			CState* tmpState = _states[i];
//			if (tmpState) delete tmpState;
//		}
		//_states.clear();

//		for(int i=0; i< _actions.size(); ++i )
//		{
//			CAction* tmpAction = _actions[i];
//			if (tmpAction) delete tmpAction;
//		}
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
        //list<CStateCollection*>::iterator stateIt = _states.begin();
		list<CAction*>::iterator actionIt = _actions.begin();

        list<double>::iterator itMargin = _margins.begin();
        list<int>::iterator itIters = _iters.begin();		
        
        double treshold = getParameter("ETraceTreshold");
        int i = 0;
        while (eIt != _eTraces.end())
        {
            OneIterETrace & oneItEtrace = *eIt;
            OneIterETrace gradient;
									
            dynamic_cast<RBFBasedQFunctionBinary* >(qFunction)->getGradient(*itMargin, *itIters, *actionIt, gradient);
            
            for (int j = 0; j < oneItEtrace.size(); ++j) {
                for (int k = 0; k < oneItEtrace[j].size(); ++k) {
                    oneItEtrace[j][k] = oneItEtrace[j][k] * mult ;
                    oneItEtrace[j][k] +=  gradient[j][k];
                }
            }
            
            ++eIt;
            //++stateIt;
            ++actionIt;
            ++itMargin;
			++itIters;
//            if (fabs(*eIt) < treshold)
//            {
//                _eTraces.erase(eIt, _eTraces.end());
//                _states.erase(stateIt, _states.end());
//                _actions.erase(actionIt, _actions.end());
//                
//                eIt = _eTraces.begin();
//                for (int j = 0; j < i; j++, eIt++);
//            }
//            else
//            {
//                i++;
//                ++eIt;
//                ++stateIt;
//                ++actionIt;
//            }
        }
	}
    
	/// Interface function for adding a State-Action pair with the given factor to the ETraces
	virtual void addETrace(CStateCollection *state, CAction *action, double factor = 1.0, CActionData *data = NULL) 
	{
		//CState* currentState = static_cast<CState*>(state)->clone();		
		//CState* copyState = state->getState()->clone();
		//_states.push_back( copyState );
		_currentState = state;
		//MultiBoost::CAdaBoostAction* currentAction = NULL;
		//int mode = dynamic_cast<MultiBoost::CAdaBoostAction*>(action)->getMode();
		//currentAction = new MultiBoost::CAdaBoostAction( mode );
		_actions.push_back( action );

        
        
        OneIterETrace gradient;
        dynamic_cast<RBFBasedQFunctionBinary* >(qFunction)->getGradient(state, action, gradient);
            
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
		//list<CStateCollection*>::iterator invitState = _states.begin();
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
			cout << "O:" << dynamic_cast<RBFBasedQFunctionBinary* >(qFunction)->getValue(artificialState, currentAction)<< "," << flush;
#endif			
			dynamic_cast<RBFBasedQFunctionBinary* >(qFunction)->updateValue(artificialState, currentAction, td, currentETrace);
#ifdef RBFDEB					
			cout << "N:" << dynamic_cast<RBFBasedQFunctionBinary* >(qFunction)->getValue(artificialState, currentAction) << ")" << endl << flush;
#endif			
		}
#ifdef RBFDEB				
		cout << endl;
		if (index==2) cout << "End of episode" << endl;
#endif		
	}	
};

#endif