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
	list<CStateCollection*> _states;
	list<CAction*> _actions;
    CustomETrace _eTraces; //strange ETrace struture, due to our peculiar state representation

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
		_states.clear();

//		for(int i=0; i< _actions.size(); ++i )
//		{
//			CAction* tmpAction = _actions[i];
//			if (tmpAction) delete tmpAction;
//		}
		_actions.clear();
        
        _eTraces.clear();
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
        list<CStateCollection*>::iterator stateIt = _states.begin();
		list<CAction*>::iterator actionIt = _actions.begin();
        
        double treshold = getParameter("ETraceTreshold");
        int i = 0;
        while (eIt != _eTraces.end())
        {
            OneIterETrace & oneItEtrace = *eIt;
            OneIterETrace gradient;
            dynamic_cast<RBFBasedQFunctionBinary* >(qFunction)->getGradient(*stateIt, *actionIt, gradient);
            
            for (int j = 0; j < oneItEtrace.size(); ++j) {
                for (int k = 0; k < oneItEtrace[j].size(); ++k) {
                    oneItEtrace[j][k] = oneItEtrace[j][k] * mult ;
                    oneItEtrace[j][k] += + gradient[j][k];
                }
            }
            
            ++eIt;
            ++stateIt;
            ++actionIt;
            
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
		_states.push_back( state );
		
		//MultiBoost::CAdaBoostAction* currentAction = NULL;
		//int mode = dynamic_cast<MultiBoost::CAdaBoostAction*>(action)->getMode();
		//currentAction = new MultiBoost::CAdaBoostAction( mode );
		_actions.push_back( action );

        CState* currState = state->getState();
        
        OneIterETrace gradient;
        dynamic_cast<RBFBasedQFunctionBinary* >(qFunction)->getGradient(state, action, gradient);
            
        _eTraces.push_back(gradient);
                        
	}
	
	/// Interface function for updating the Q-Values of all State-Action Pairs in the ETraces
	virtual void updateQFunction(double td) 
	{
		list<CStateCollection*>::reverse_iterator invitState = _states.rbegin();
		list<CAction*>::reverse_iterator invitAction = _actions.rbegin();
        ETReverseIterator invitTrace = _eTraces.rbegin();
        
		for (; invitState != _states.rend(); ++invitState, ++invitAction, ++invitTrace)
		{
			CStateCollection* currentState = *invitState;
			CAction* currentAction = *invitAction;
            OneIterETrace& currentETrace = *invitTrace;
            
			dynamic_cast<RBFBasedQFunctionBinary* >(qFunction)->updateValue(currentState, currentAction, td, currentETrace);
		}
	}	
};

#endif