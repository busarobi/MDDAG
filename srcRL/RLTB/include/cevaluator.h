#ifndef C_EVALUATOR__H 
#define C_EVALUATOR__H 

#include "cagentlistener.h"

class CStateList; 
class CTransitionFunctionEnvironment;
class CAgent;
class CSemiMDPSender;
class CAgentController;
class CActionDataSet;
class CAgentController;
class CDeterministicController;

class CEvaluator 
{
	protected:

	public:	
		virtual ~CEvaluator() {};
		virtual double evaluate() = 0;
};


class CPolicyEvaluator : public CSemiMDPRewardListener, public CEvaluator 
{
	protected:
		CAgent *agent;
		CAgentController *controller;
		CDeterministicController *detController;

		double policyValue;
	
		int nEpisodes;
		int nStepsPerEpisode;

		virtual double getEpisodeValue() = 0;
	public:
		CPolicyEvaluator(CAgent *agent, CRewardFunction *rewardFunction, int nEpisodes, int nStepsPerEpisode);
		virtual ~CPolicyEvaluator() {};

		virtual double evaluatePolicy();
		virtual double evaluate() {return evaluatePolicy();};
	
		virtual void nextStep(CStateCollection *oldState, CAction *action, double reward, CStateCollection *nextState) = 0;

		virtual void setStepsPerEpisode(int steps) {nStepsPerEpisode = steps;};

		virtual void setAgentController(CAgentController *controller);
		virtual void setDeterministicController(CDeterministicController *detController);
};


class CAverageRewardCalculator : public CPolicyEvaluator
{
	protected:
		int nSteps;
		double averageReward;
		double minReward;

		virtual double getEpisodeValue();
	public:
		CAverageRewardCalculator(CAgent *agent, CRewardFunction *rewardFunction, int nEpisodes, int nStepsPerEpisode,double minReward = -2.0);
		virtual ~CAverageRewardCalculator(){};

		virtual void nextStep(CStateCollection *oldState, CAction *action, double reward, CStateCollection *nextState);
		virtual void newEpisode();
};

class CRewardPerEpisodeCalculator : public CPolicyEvaluator
{
	protected:
		
		double reward;
		

		virtual double getEpisodeValue();
	public:
		CRewardPerEpisodeCalculator(CAgent *agent, CRewardFunction *rewardFunction, int nEpisodes, int nStepsPerEpisode);

		virtual void nextStep(CStateCollection *oldState, CAction *action, double reward, CStateCollection *nextState);
		virtual void newEpisode();
};

class CValueCalculator : public CPolicyEvaluator
{
	protected:
		int nSteps;
		double value;

		virtual double getEpisodeValue();
	public:
		CValueCalculator(CAgent *agent, CRewardFunction *rewardFunction, int nEpisodes, int nStepsPerEpisode, double gamma);

		virtual void nextStep(CStateCollection *oldState, CAction *action, double reward, CStateCollection *nextState);
		virtual void newEpisode();
};

class CPolicySameStateEvaluator : public CPolicyEvaluator
{
	protected:
		CStateList *startStates;
		CTransitionFunctionEnvironment *environment;

		CSemiMDPSender *sender;

		virtual double getEpisodeValue() = 0;
	public:
		CPolicySameStateEvaluator(CAgent *agent, CRewardFunction *rewardFunction, CTransitionFunctionEnvironment *environment, CStateList *startStates, int nStepsPerEpisode);
		CPolicySameStateEvaluator(CAgent *agent, CRewardFunction *rewardFunction, CTransitionFunctionEnvironment *environment, int numStartStates, int nStepsPerEpisode);
		virtual ~CPolicySameStateEvaluator();
	
		void setSemiMDPSender(CSemiMDPSender  *l_sender) {sender = l_sender;};

		virtual double evaluatePolicy();

		virtual double getValueForState(CState *state, int nSteps);
		virtual double getActionValueForState(CState *state, CAction *action, int nSteps);


		virtual void nextStep(CStateCollection *oldState, CAction *action, double reward, CStateCollection *nextState) = 0;

		virtual CStateList *getStartStates(){return startStates;};
		virtual void setStartStates(CStateList *newList);

		void getNewStartStates(int numStartStates);
};

class CAverageRewardSameStateCalculator : public CPolicySameStateEvaluator
{
	protected:
		int nSteps;
		double averageReward;
		double minReward;

		virtual double getEpisodeValue();
	public:
		CAverageRewardSameStateCalculator(CAgent *agent, CRewardFunction *rewardFunction, CTransitionFunctionEnvironment *environment, CStateList *startStates, int nStepsPerEpisode, double minReward = -2.0);

		virtual void nextStep(CStateCollection *oldState, CAction *action, double reward, CStateCollection *nextState);
		virtual void newEpisode();
};

class CValueSameStateCalculator : public CPolicySameStateEvaluator
{
	protected:
		int nSteps;
		double value;

		virtual double getEpisodeValue();
	public:
		CValueSameStateCalculator(CAgent *agent, CRewardFunction *rewardFunction, CTransitionFunctionEnvironment *environment, CStateList *startStates, int nStepsPerEpisode, double gamma);

		virtual void nextStep(CStateCollection *oldState, CAction *action, double reward, CStateCollection *nextState);
		virtual void newEpisode();
};

class CPolicyGreedynessEvaluator : public CPolicyEvaluator
{
	protected:
		CAgentController *greedyPolicy;
		CActionDataSet *actionDataSet;

		virtual double getEpisodeValue();

		int nGreedyActions;
	public:
		CPolicyGreedynessEvaluator(CAgent *agent, CRewardFunction *reward, int nEpisodes, int nStepsPerEpsiode, CAgentController *l_greedyPolicy);
		~CPolicyGreedynessEvaluator();

		virtual void nextStep(CStateCollection *oldState, CAction *action, double reward, CStateCollection *nextState);
		virtual void newEpisode();

};



#endif
