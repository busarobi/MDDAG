#include "RBFBasedQFunction.h"
#include "RBFQETraces.h"
	
RBFBasedQFunctionBinary::RBFBasedQFunctionBinary(CActionSet *actions, CStateModifier* statemodifier ) : CAbstractQFunction(actions)
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
    
//    assert(numOfClasses==1);
    
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
    
    addParameter("InitRBFSigma", 0.01);
    addParameter("AddCenterOnError", 1.);
    addParameter("MaxTDErrorDivFactor", 10);
    addParameter("MinActivation", 0.3);
    addParameter("QLearningRate", 0.2);
    
}

CAbstractQETraces* RBFBasedQFunctionBinary::getStandardETraces()
{
    return new RBFQETraces(this);
}


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
                _rbfs[*it][i][j].setSigma(1./ (2*numFeat + 0.1));
                
                stringstream tmpString("");
                tmpString << "[ac_" << index << "|it_" << i << "|fn_" << j << "]";
                _rbfs[*it][i][j].setID( tmpString.str() );
            }
        }
    }           
}

/// Interface for getting a Q-Value
double RBFBasedQFunctionBinary::getValue(CStateCollection *state, CAction *action, CActionData *data) 
{
    CState* currState = state->getState();
    
    int currIter = currState->getDiscreteState(0);
    double margin = currState->getContinuousState(0);
    
    double retVal = 0.0;
    
    vector<RBF>& currRBFs = _rbfs[action][currIter];		
    for( int i=0; i<currRBFs.size(); ++i )
    {
        retVal += currRBFs[i].getValue(margin);  // lehet, hogy eggyet hozza kell adni a currIter-hez
    }		
    return retVal;
}


double RBFBasedQFunctionBinary::getActivation(CStateCollection *state, CAction *action, CActionData *data ) 
{
    CState* currState = state->getState();
    
    int currIter = currState->getDiscreteState(0);
    double margin = currState->getContinuousState(0);
    
    double retVal = 0.0;
    
    vector<RBF>& currRBFs = _rbfs[action][currIter];		
    for( int i=0; i<currRBFs.size(); ++i )
    {
        retVal += currRBFs[i].getActivationFactor(margin);  // lehet, hogy eggyet hozza kell adni a currIter-hez
    }		
    return retVal;
}

double RBFBasedQFunctionBinary::addCenter(double tderror, double newCenter, int iter, CAction* action, double& maxError) 
{
    vector<RBF>& rbfs = _rbfs[action][iter];
    
    int numCenters = rbfs.size();
    double defaultSigma = getParameter("InitRBFSigma");
    double newSigma = defaultSigma;
    int index = 0;        
    if (numCenters > 10) {        
        for (int i = 0; i < rbfs.size(); ++i) {
            if (rbfs[i].getMean() >= newCenter) {
                break; 
            }
            ++index;
        }
        
        double leftPosition = 0.;
        double leftSigma = 0.;
        double rightPosition = 1.;
        double rightSigma = 0.;
        
        if (index != numCenters) {
            rightPosition = rbfs[index].getMean();
            rightSigma = rbfs[index].getSigma();
        }

        if (index != 0) {
            leftPosition = rbfs[index - 1].getMean();
            leftSigma = rbfs[index - 1].getSigma();
            --index;
        }
        
        newSigma = rightPosition - leftPosition - 4*(rightSigma + leftSigma) - 0.1;
        
        //assess this
        if (newSigma > _maxSigma) {
            newSigma = defaultSigma;
        }
    }
    
    //        cout << "[+] State : [ " << iter << " ]" << endl ;
    //        cout << "[+] New center is added : [ " << tderror << ", " << newCenter << ", " << newSigma << " ]" << endl ;
    
    RBF newRBF;
    newRBF.setMean(newCenter);
    newRBF.setAlpha(tderror);
    newRBF.setSigma(newSigma);

    int actionIndex = dynamic_cast<MultiBoost::CAdaBoostAction*>(action)->getMode();
    stringstream tmpString("");
    tmpString << "[ac_" << actionIndex << "|it_" << iter << "|fn_" << index << "]";
    newRBF.setID( tmpString.str() );
    
#ifdef RBFDEB
    cout << "New center : " << newRBF.getID() << endl;
#endif
    
    rbfs.insert(rbfs.begin()+index, newRBF );
    
    for (int i=0; i < rbfs.size(); ++i) {
        if (rbfs[i].getAlpha() > maxError) {
            maxError = rbfs[i].getAlpha();
        }
    }
    
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
void RBFBasedQFunctionBinary::updateValue(CStateCollection *state, CAction *action, double td, vector<vector<double> >& eTraces)
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

void RBFBasedQFunctionBinary::getGradient(CStateCollection *state, CAction *action, vector<vector<double> >& gradient)
{
    CState* currState = state->getState();
    int currIter = currState->getDiscreteState(0);
    double margin = currState->getContinuousState(0);
    
    getGradient(margin, currIter, action, gradient);
}

void RBFBasedQFunctionBinary::getGradient(double margin, int currIter, CAction *action, vector<vector<double> >& gradient)
{
    //        CState* currState = state->getState();
    //        int currIter = currState->getDiscreteState(0);
    //		double margin = currState->getContinuousState(0);
    
    assert (_rbfs.find(action) != _rbfs.end());
    
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

void RBFBasedQFunctionBinary::saveQTable( const char* fname )
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


//    void saveActionValueTable(FILE* stream)
//    {
//        fprintf(stream, "Q-FeatureActionValue Table\n");
//        CActionSet::iterator it;
//        
//        for (int i = 0; i < _featureNumber; ++i) {
//            for (int j = 0; j < _numberOfIterations; ++j) {
//                fprintf(stream,"Feature %d: ", i);
//                for(it =_actions->begin(); it!=_actions->end(); ++it ) {
//                    fprintf(stream,"%f %f %f ", _rbfs[*it][j][i].getAlpha(), _rbfs[*it][j][i].getMean(), _rbfs[*it][j][i].getSigma());
//                }
//                fprintf(stream, "\n");
//            }
//        }
//    }

void RBFBasedQFunctionBinary::saveActionValueTable(FILE* stream)
{
    fprintf(stream, "Q-FeatureActionValue Table\n");
    CActionSet::iterator it;
    
    for (int j = 0; j < _numberOfIterations; ++j) {
        int k=0;
        for(it =_actions->begin(); it!=_actions->end();  ++it, ++k ) {
            fprintf(stream,"classifier %d action %d: ", j,k);
            for (int i = 0; i < _rbfs[*it][j].size(); ++i) {
                fprintf(stream,"%f %f %f ", _rbfs[*it][j][i].getAlpha(), _rbfs[*it][j][i].getMean(), _rbfs[*it][j][i].getSigma());
            }
            fprintf(stream, "\n");
        }
        
        //            vector<vector<double> > tmp;
        //            tmp.resize(3);
        //            int k=0;
        //            for(it =_actions->begin(); it!=_actions->end(); ++k, ++it ) {
        //                tmp[k].resize(_rbfs[*it][j].size());
        //                for (int i = 0; i < _rbfs[*it][j].size(); ++i) {
        //                    tmp[k][i] = _rbfs[*it][j][i].getAlpha();
        //                }
        //            }
        //            
        //            for (int i = 0; i < tmp[0].size()  ; ++i) {
        //                fprintf(stream,"Feature %d-%d: ", j,i);
        //                for (int j = 0; j < 3; ++j) {
        //                    fprintf(stream,"%f ", tmp[j][i]);
        //                }
        //                fprintf(stream, "\n");
        //            }
    }
}

void RBFBasedQFunctionBinary::saveActionTable(FILE* stream)
{
    fprintf(stream, "Q-FeatureAction Table\n");
    double max = 0.0;
    int maxIndex = 0;
    
    CActionSet::iterator it;
    for (int i = 0; i < _featureNumber; ++i) {
        for (int j = 0; j < _numberOfIterations; ++j) {
            fprintf(stream,"Feature %d-%d: ", j,i);
            
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

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

GSBNFBasedQFunction::GSBNFBasedQFunction(CActionSet *actions, CStateModifier* statemodifier )
: CAbstractQFunction(actions)
{
    RBFStateModifier* smodifier = dynamic_cast<RBFStateModifier*>( statemodifier );

    
    const int iterationNumber = smodifier->getNumOfIterations();
    const int featureNumber = smodifier->getNumOfRBFsPerIteration();
    const int numOfClasses = smodifier->getNumOfClasses();
    
    _featureNumber = featureNumber;
    _numberOfIterations = iterationNumber;
    _numDimensions = numOfClasses;
    
    //    assert(numOfClasses==1);
    
    _actions = actions;
    _numberOfActions = actions->size();
    
    _rbfs.clear();
    _rbfs.resize(_numberOfActions);
    
//    CActionSet::iterator it=_actions->begin();
    for (int ac=0; ac < _numberOfActions; ++ac) 
    {
//        int actionIndex = dynamic_cast<MultiBoost::CAdaBoostAction*>(*it)->getMode();
        int actionIndex = ac;
        _rbfs[actionIndex].resize( iterationNumber );
        for( int i=0; i<iterationNumber; ++i)
        {
            _rbfs[actionIndex][i].reserve(_featureNumber);
            for (int j = 0; j < _featureNumber; ++j) {
                MultiRBF rbf(_numDimensions);
                _rbfs[actionIndex][i].push_back(rbf);
            }
        }
    }
    
    addParameter("InitRBFSigma", 0.01);
    addParameter("AddCenterOnError", 1.);
    addParameter("NormalizedRBFs", 1);
    addParameter("MaxTDErrorDivFactor", 10);
    addParameter("MinActivation", 0.3);
    addParameter("QLearningRate", 0.2);
}


double GSBNFBasedQFunction::getValue(CStateCollection *state, CAction *action, CActionData *data)
{
    CState* currState = state->getState();
    int currIter = currState->getDiscreteState(0);
    
    RBFParams margin(_numDimensions);
    for (int i = 0; i < _numDimensions; ++i) {
        margin[i] = currState->getContinuousState(i);
    }
    
    double retVal = 0.0;
    double rbfSum = 0.0;
    
    int actionIndex = dynamic_cast<MultiBoost::CAdaBoostAction*>(action)->getMode();
    
    vector<MultiRBF>& currRBFs = _rbfs[actionIndex][currIter];		

    if (currRBFs.size() == 0) {
        return  0;
    }
    
    for( int i=0; i<currRBFs.size(); ++i )
    {
        if (rbfSum != rbfSum) {
            assert(false);
        }
        
        rbfSum += currRBFs[i].getActivationFactor(margin);
        retVal += currRBFs[i].getValue(margin);
    }		

    bool norm  = getParameter("NormalizedRBFs") > 0.5;
    if (norm) {
        assert(false);
        retVal /= rbfSum;
    }
    
    assert( retVal == retVal);
    return retVal;
}

double GSBNFBasedQFunction::getMaxActivation(CStateCollection *state, int action, CActionData *data )
{
    CState* currState = state->getState();
    int currIter = currState->getDiscreteState(0);

    RBFParams margin(_numDimensions);
    for (int i = 0; i < _numDimensions; ++i) {
        margin[i] = currState->getContinuousState(i);
    }
    
    double maxVal = 0.0;
    
    vector<MultiRBF>& currRBFs = _rbfs[action][currIter];		
    for( int i=0; i<currRBFs.size(); ++i )
    {
        double act = currRBFs[i].getActivationFactor(margin); 
        if (act > maxVal) {
            maxVal = act;
        }
    }		
    return maxVal;
}


void GSBNFBasedQFunction::getActivationFactors(RBFParams& margin, int currIter, int action, vector<double>& factors)
{
    double sum = 0.0;
    
    vector<MultiRBF>& currRBFs = _rbfs[action][currIter];		
    factors.clear();
    
    if (currRBFs.size() == 0) {
        return ;
    }

    factors.resize(currRBFs.size());
                   
    for( int i=0; i<currRBFs.size(); ++i )
    {
        factors[i] = currRBFs[i].getActivationFactor(margin); 
        sum += currRBFs[i].getActivationFactor(margin);
    }
                   
    for( int i=0; i<currRBFs.size(); ++i )
    {
        factors[i] /= sum;
    }
}

double GSBNFBasedQFunction::addCenter(double tderror, RBFParams& newCenter, int iter, int action, double& maxError)
{
    vector<MultiRBF>& rbfs = _rbfs[action][iter];
    
    double newSigma = getParameter("InitRBFSigma");

    MultiRBF newRBF(_numDimensions);
    newRBF.setMean(newCenter);
    newRBF.setAlpha(tderror);
    newRBF.setSigma(newSigma);
    
    stringstream tmpString("");
    tmpString << "[ac_" << action << "|it_" << iter << "|fn_" << index << "]";
    newRBF.setId( tmpString.str() );
    
//#ifdef RBFDEB
    cout << "New center : " << newRBF.getId() << " at " << newCenter[0] << endl;
    cout << tderror << "\t" << newCenter[0] << "\t" << newSigma << endl ;
//#endif
    
    rbfs.push_back( newRBF );
    
    for (int i=0; i < rbfs.size(); ++i) {
        if (rbfs[i].getAlpha()[0] > maxError) {
            maxError = rbfs[i].getAlpha()[0];
        }
    }
    
    //normalizeNetwork();
    
}

void GSBNFBasedQFunction::updateValue(int currIter, RBFParams& margin, int action, double td, vector<vector<RBFParams> >& eTraces)
{
    
    if (td != td) {
        assert(false);
    }
    
//    CState* currState = state->getState();
//    int currIter = currState->getDiscreteState(0);
//
//    RBFParams margin(_numDimensions);
//    for (int i = 0; i < _numDimensions; ++i) {
//        margin[i] = currState->getContinuousState(i);
//    }
//    assert (_rbfs.find(action) != _rbfs.end());
    
    vector<MultiRBF>& rbfs = _rbfs[action][currIter];
    int numCenters = rbfs.size();//_rbfs[currIter][action].size();
    
    for (int i = 0; i < numCenters; ++i) {
        
        RBFParams& alpha = rbfs[i].getAlpha();
        RBFParams& mean = rbfs[i].getMean();
        RBFParams& sigma = rbfs[i].getSigma();
        
//        cout << "RBF : " << rbfs[i].getId() << endl;
//        cout << alpha  << mean  << sigma << endl;

        //update the center and shape
        vector<RBFParams>& currentGradient = eTraces[i];
        
        RBFParams newAlpha(_numDimensions);
        RBFParams newMean(_numDimensions);
        RBFParams newSigma(_numDimensions);
        
        for (int j = 0; j < _numDimensions; ++j) {
            newAlpha[j] = alpha[j] + _muAlpha * currentGradient[0][j] * td  ;
            newMean[j] = mean[j] + _muMean * currentGradient[1][j] * td  ;
            newSigma[j] = sigma[j] + _muSigma * currentGradient[2][j] * td  ;
            
            if (newMean[j] != newMean[j]) {
                assert(false);
            }
            
            if (newSigma[j] != newSigma[j]) {
                cout << currentGradient[2][j] << endl;
                assert(false);
            }
            
        }
        
        rbfs[i].setAlpha(newAlpha);
        rbfs[i].setMean(newMean);
        rbfs[i].setSigma(newSigma);
#ifdef RBFDEB			
        cout << rbfs[i].getID() << " ";
#endif
    }		
}

void GSBNFBasedQFunction::getGradient(CStateCollection *state, int action, vector<vector<RBFParams> >& gradient)
{
    CState* currState = state->getState();
    int currIter = currState->getDiscreteState(0);

    RBFParams margin(_numDimensions);
    for (int i = 0; i < _numDimensions; ++i) {
        margin[i] = currState->getContinuousState(i);
    }
    
    getGradient(margin, currIter, action, gradient);
}

void GSBNFBasedQFunction::getGradient(RBFParams& margin, int currIter, int action, vector<vector<RBFParams> >& gradient)
{
    vector<MultiRBF>& rbfs = _rbfs[action][currIter];
    int numCenters = rbfs.size();
    gradient.clear();
    gradient.resize(numCenters);

    vector<double> activationFactors;
    getActivationFactors(margin, currIter, action, activationFactors);
    
    for (int i = 0; i < numCenters; ++i) {
        RBFParams& alpha = rbfs[i].getAlpha();
        RBFParams& mean = rbfs[i].getMean();
        RBFParams& sigma = rbfs[i].getSigma();
        
        RBFParams distance(_numDimensions);
        for (int j = 0; j < _numDimensions; ++j) {
            distance[j] = margin[j] - mean[j];
        }
        
        double rbfValue = activationFactors[i];
        
        RBFParams alphaGrad = RBFParams(_numDimensions);
        for (int j = 0; j < _numDimensions; ++j) {
            alphaGrad[j] = rbfValue;
        }
        
        
        RBFParams meanGrad(_numDimensions);
        RBFParams sigmaGrad(_numDimensions);
        for (int j = 0; j < _numDimensions; ++j) {
            meanGrad[j] = rbfValue * alpha[0] * distance[j] / (sigma[j]*sigma[j]);
            sigmaGrad[j] = rbfValue * alpha[0] * distance[j] * distance[j]/ (sigma[j] * sigma[j] * sigma[j]);
            
            if (meanGrad[j] != meanGrad[j]) {
                assert(false);
            }
            
            if (sigmaGrad[j] != sigmaGrad[j]) {
                assert(false);
            }
            
            
            bool norm  = getParameter("NormalizedRBFs") > 0.5;
            if (norm) {
                meanGrad[j] *= (1 - rbfValue);
                sigmaGrad[j] *= (1 - rbfValue);
            }
        }
        
                

//        RBFParams meanGrad = SP( SP((1 - rbfValue) , SP(rbfValue, SP(alpha,distance))) , 1/SP(sigma,sigma));
//        RBFParams sigmaGrad = SP( SP((1 - rbfValue) , SP(rbfValue, SP(alpha,SP(distance,distance)))) , 1/SP(sigma, SP(sigmasigma)));
        
        gradient[i].resize(3);
        gradient[i][0] = alphaGrad;
        gradient[i][1] = meanGrad;
        gradient[i][2] = sigmaGrad;
    }
}

void GSBNFBasedQFunction::saveActionValueTable(FILE* stream, int dim)
{
    fprintf(stream, "Q-FeatureActionValue Table\n");
//    CActionSet::iterator it;
    
    for (int j = 0; j < _numberOfIterations; ++j) {
        for (int k = 0; k < _numberOfActions; ++k) {
            fprintf(stream,"classifier %d action %d: ", j,k);
            for (int i = 0; i < _rbfs[k][j].size(); ++i) {
                fprintf(stream,"%f %f %f ", _rbfs[k][j][i].getAlpha()[dim], _rbfs[k][j][i].getMean()[dim], _rbfs[k][j][i].getSigma()[dim]);
            }
            fprintf(stream, "\n");
        }
        
    }
}

CAbstractQETraces* GSBNFBasedQFunction::getStandardETraces()
{
    return new GSBNFQETraces(this);
}


void GSBNFBasedQFunction::uniformInit(double* init)
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
        
        int iterationNumber = _rbfs[index].size();
        for( int i=0; i<iterationNumber; ++i)
        {                
            int numFeat = _rbfs[index][i].size();
            for (int j = 0; j < numFeat; ++j) {
                //                    if (numFeat % 2 == 0) {
                _rbfs[index][i][j].setMean((j+1) * 1./(numFeat+1));
                //                    }
                //                    else {
                //                        _rbfs[*it][i][j].setMean(j * 1./numFeat);   
                //                    }
                
                _rbfs[index][i][j].setAlpha(initAlpha);
                _rbfs[index][i][j].setSigma(1./ (2.2*numFeat));
                
                stringstream tmpString("");
                tmpString << "[ac_" << index << "|it_" << i << "|fn_" << j << "]";
                _rbfs[index][i][j].setId( tmpString.str() );
            }
        }
    }           
}

