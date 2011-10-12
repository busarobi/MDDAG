/*
 *  RBFQETraces.cpp
 *  RLTools
 *
 *  Created by Robert Busa-Fekete on 10/12/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#include "RBFQETraces.h"
#include "AdaBoostMDPClassifierAdv.h"
#include "caction.h"
#include "RBFBasedQFunction.h"

//------------------------------------------------------
//------------------------------------------------------    
void RBFQETraces::resetETraces() 
{
	_actions.clear();        
	_eTraces.clear();
	_iters.clear();
	_margins.clear();
	
#ifdef RBFDEB		
	cout << "------------------------------------------------" << endl;
#endif
}

//------------------------------------------------------
//------------------------------------------------------    
void RBFQETraces::updateETraces(CAction *action, CActionData *data) 
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
		
		dynamic_cast<RBFBasedQFunctionBinary* >(qFunction)->getGradient(*itMargin, *itIters, *actionIt, gradient);
		
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
//------------------------------------------------------
//------------------------------------------------------    
void RBFQETraces::addETrace(CStateCollection *state, CAction *action, double factor, CActionData *data) 
{
	_currentState = state;
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

//------------------------------------------------------	
//------------------------------------------------------	
/// Interface function for updating the Q-Values of all State-Action Pairs in the ETraces
void RBFQETraces::updateQFunction(double td) 
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
		cout << "O:" << dynamic_cast<RBFBasedQFunctionBinary* >(qFunction)->getValue(artificialState, currentAction)<< "," << flush;
#endif			
		dynamic_cast<RBFBasedQFunctionBinary* >(qFunction)->updateValue(artificialState, currentAction, td, currentETrace);
#ifdef RBFDEB					
		cout << "N:" << dynamic_cast<RBFBasedQFunctionBinary* >(qFunction)->getValue(artificialState, currentAction) << ")" << endl << flush;
#endif			
		delete artificialState;
	}
#ifdef RBFDEB				
	cout << endl;
	if (index==2) cout << "End of episode" << endl;
#endif		
}
//------------------------------------------------------	
//------------------------------------------------------	


