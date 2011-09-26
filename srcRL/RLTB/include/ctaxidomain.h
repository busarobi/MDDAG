#ifndef C__TAXIDOMAIN_H
#define C__TAXIDOMAIN_H

#include "cgridworldmodel.h"
#include "cagent.h"
#include "caction.h"
#include "cdiscretizer.h"
#include "crewardfunction.h"


#include <vector>

#define PICKUPACTION 32
#define PUTDOWNACTION 64

class CTaxiDomain : public CGridWorldModel
{
protected:
	std::vector<std::pair<int, int> *> *targetXYValues;

	virtual void initTargetVector();
public:

	CTaxiDomain(char* filename);	
	virtual ~CTaxiDomain();

	virtual void load(FILE *stream);

	int getTargetPositionX(int numTarget);

	int getTargetPositionY(int numTarget);

	int getNumTargets() {return targetXYValues->size();};

	double getReward(CStateCollection *, CAction *, CStateCollection *);

	virtual void transitionFunction(CState *oldstate, CAction *action, CState *newState, CActionData *data = NULL);

	virtual bool isResetState(CState *state);

	virtual void getResetState(CState *resetState);


};

class CPickupAction : public  CPrimitiveAction
{
public:
	CPickupAction(){addType(PICKUPACTION);};
};

class CPutdownAction : public  CPrimitiveAction
{
public:
	CPutdownAction(){addType(PUTDOWNACTION);};
};

class CTaxiHierarchicalBehaviour : public CHierarchicalSemiMarkovDecisionProcess, public CRewardFunction
{
protected:
	int target;
	CTaxiDomain *model;

public:
	CTaxiHierarchicalBehaviour(CEpisode *currentEpisode, int target, CTaxiDomain *taximodel);
	~CTaxiHierarchicalBehaviour();

	virtual bool isFinished(CStateCollection *state, CStateCollection *newState);
	virtual double getReward(CStateCollection *oldState, CAction *action, CStateCollection *newState);
};

class CTaxiIsTargetDiscreteState : public CAbstractStateDiscretizer
{
protected:
	CTaxiDomain *model;
public:
	CTaxiIsTargetDiscreteState(CTaxiDomain *model);

	virtual unsigned int getDiscreteStateNumber(CStateCollection *stateCol);

};


#endif
