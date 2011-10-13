#include "RBFBasedQFunction.h"
#include "RBFQETraces.h"
	

//------------------------------------------------------
//------------------------------------------------------    
CAbstractQETraces* RBFBasedQFunctionBinary::getStandardETraces()
{
    return new RBFQETraces(this);
}

//------------------------------------------------------
//------------------------------------------------------    
RBFBasedQFunctionBinary::RBFBasedQFunctionBinary(CActionSet *actions, CStateModifier* statemodifier ) : CAbstractQFunction(actions)
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
		_rbfs[*it].resize( iterationNumber );
		for( int i=0; i<iterationNumber; ++i)
		{
			_rbfs[*it][i].resize(_featureNumber);
		}
	}
}


//------------------------------------------------------
//------------------------------------------------------    
double RBFBasedQFunctionBinary::getValue(CStateCollection *state, CAction *action, CActionData *data) 
{
	CState* currState = state->getState();
	
	int currIter = currState->getDiscreteState(0);
	double margin = currState->getContinuousState(0);
	
	double retVal = 0.0;
	
	vector<RBF>& currRBFs = _rbfs[action][currIter];		
	for( int i=0; i<_featureNumber; ++i )
	{
		retVal += currRBFs[i].getValue(margin); 
	}		
	return retVal;
}

//------------------------------------------------------
//------------------------------------------------------    
void RBFBasedQFunctionBinary::updateValue(CStateCollection *state, CAction *action, double td, vector<vector<double> >& eTraces)
{
	CState* currState = state->getState();
	int currIter = currState->getDiscreteState(0);
	double margin = currState->getContinuousState(0);
	
	vector<RBF>& rbfs = _rbfs[action][currIter];
	int numCenters = rbfs.size();//_rbfs[currIter][action].size();
	
	//        assert(numCenters = eTraces.size());
	
	for (int i = 0; i < numCenters; ++i) {
		double th = 0.01;
		double alpha = rbfs[i].getAlpha();
		double mean = rbfs[i].getMean();
		double sigma = rbfs[i].getSigma();
		
		//update the center and shape
		vector<double>& currentGradient = eTraces[i];
		double alphaStep = currentGradient[0] * td * _muAlpha;
		
		rbfs[i].setAlpha(alpha + alphaStep );
		
		double meanStep = currentGradient[1] * td * _muMean;
		meanStep = (meanStep>th) ?  th : meanStep;		
		meanStep = (meanStep<-th) ? -th : meanStep;				
		rbfs[i].setMean(mean + meanStep );
		
		double sigmaStep = currentGradient[2] * td * _muSigma;
		sigmaStep = (sigmaStep>th) ?  th : sigmaStep;
		sigmaStep = (sigmaStep<-th) ? -th : sigmaStep;
		rbfs[i].setSigma(sigma + sigmaStep );
#ifdef RBFDEB			
		cout << rbfs[i].getID() << " ";
#endif
	}		
}


//------------------------------------------------------
//------------------------------------------------------    
void RBFBasedQFunctionBinary::getGradient(CStateCollection *state, CAction *action, vector<vector<double> >& gradient)
{
	CState* currState = state->getState();
	int currIter = currState->getDiscreteState(0);
	double margin = currState->getContinuousState(0);
	
	getGradient(margin, currIter, action, gradient);
}

//------------------------------------------------------
//------------------------------------------------------    
void RBFBasedQFunctionBinary::getGradient(double margin, int currIter, CAction *action, vector<vector<double> >& gradient, bool isNorm )
{
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
	
	if (isNorm)
	{
		double lengthScale = 0.0;
		for (int i = 0; i < numCenters; ++i) {
			lengthScale += (gradient[i][0]*gradient[i][0]);
			lengthScale += (gradient[i][1]*gradient[i][1]);
			lengthScale += (gradient[i][2]*gradient[i][2]);			
		}
		lengthScale = sqrt(lengthScale);
		if ( ! nor_utils::is_zero(lengthScale))
		for (int i = 0; i < numCenters; ++i) {
			gradient[i][0] /= lengthScale;
			gradient[i][1] /= lengthScale;
			gradient[i][2] /= lengthScale;			
		}
		
	}	
}

//------------------------------------------------------
//------------------------------------------------------    
void RBFBasedQFunctionBinary::saveQTable( const char* fname )
{
	FILE* outFile = fopen( fname, "w" );
	
//	for (CActionSet::iterator it=_actions->begin(); it != _actions->end(); ++it)
//	{
//		fprintf(outFile,"%d ", dynamic_cast<MultiBoost::CAdaBoostAction*>(*it)->getMode() );
//	}
//	fprintf(outFile,"\n");
	
	for (CActionSet::iterator it=_actions->begin(); it != _actions->end(); ++it)		
	{
		vector<vector<RBF> >& currentRBFs = _rbfs[*it];
		//cout << currentRBFs.size() << endl;
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
//------------------------------------------------------
//------------------------------------------------------    
void RBFBasedQFunctionBinary::saveActionValueTable(FILE* stream)
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

//------------------------------------------------------
//------------------------------------------------------    
void RBFBasedQFunctionBinary::saveActionTable(FILE* stream)
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
//------------------------------------------------------
//------------------------------------------------------    
void RBFBasedQFunctionBinary::uniformInit(double* init)
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
//------------------------------------------------------
//------------------------------------------------------    
