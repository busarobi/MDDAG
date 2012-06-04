//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
// general includes

#include <time.h>

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
// for RL toolbox
#include "ril_debug.h"
#include "ctdlearner.h"
#include "cpolicies.h"
#include "cagent.h"
#include "cagentlogger.h"
#include "crewardmodel.h"
#include "canalyzer.h"
#include "cgridworldmodel.h"
#include "cvetraces.h"
#include "cvfunctionlearner.h"

#include "cadaptivesoftmaxnetwork.h"
#include "crbftrees.h"
#include "ctorchvfunction.h"
#include "ccontinuousactions.h"
#include "MLP.h"
#include "GradientMachine.h"
#include "LogRBF.h"
#include "Linear.h"
#include "Tanh.h"
#include "RBFBasedQFunction.h"

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////


//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
// for multiboost
#include "Defaults.h"
#include "Utils/Args.h"

#include "StrongLearners/GenericStrongLearner.h"
#include "WeakLearners/BaseLearner.h" // To get the list of the registered weak learners

#include "IO/Serialization.h" // for unserialization
#include "Bandits/GenericBanditAlgorithm.h" 
#include "AdaBoostMDPClassifier.h"
#include "AdaBoostMDPClassifierAdv.h"
#include "AdaBoostMDPClassifierContinous.h"
#include "AdaBoostMDPClassifierDiscrete.h"
#include "AdaBoostMDPClassifierContinousBinary.h"
#include "AdaBoostMDPClassifierContinousMultiClass.h"
#include "AdaBoostMDPClassifierSubsetSelectorBinary.h"

using namespace std;
using namespace MultiBoost;
using namespace Torch;


//#define LOGQTABLE				
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////

static const char CURRENT_VERSION[] = "1.0.00";


//---------------------------------------------------------------------------

/**
 * Check if a given base learner has been registered. If not it will give an error
 * and exit.
 * \param baseLearnerName The name of the base learner to be checked.
 * \date 21/3/2006
 */
void checkBaseLearner(const string& baseLearnerName)
{
	if ( !BaseLearner::RegisteredLearners().hasLearner(baseLearnerName) )
	{
		// Not found in the registered!
		cerr << "ERROR: learner <" << baseLearnerName << "> not found in the registered learners!" << endl;
		exit(1);
	}
}

//---------------------------------------------------------------------------

/**
 * Show the basic output. Called when no argument is provided.
 * \date 11/11/2005
 */
void showBase()
{
	cout << "MultiBoost (v" << CURRENT_VERSION << "). An obvious name for a multi-class AdaBoost learner." << endl;
	cout << "---------------------------------------------------------------------------" << endl;
	cout << "Build: " << __DATE__ << " (" << __TIME__ << ") (C) Robert Busa-Fekete, Balazs Kegl, Norman Casagrande 2005-2010" << endl << endl;
	cout << "===> Type --help for help or --static to show the static options" << endl;
	
	exit(0);
}

//---------------------------------------------------------------------------

/**
 * Show the help. Called when -h argument is provided.
 * \date 11/11/2005
 */
void showHelp(nor_utils::Args& args, const vector<string>& learnersList)
{
	cout << "MultiBoost (v" << CURRENT_VERSION << "). An obvious name for a multi-class AdaBoost learner." << endl;
	cout << "------------------------ HELP SECTION --------------------------" << endl;
	
	args.printGroup("Parameters");
	
	cout << endl;
	cout << "For specific help options type:" << endl;
	cout << "   --h general: General options" << endl;
	cout << "   --h io: I/O options" << endl;
	cout << "   --h algo: Basic algorithm options" << endl;
	cout << "   --h bandits: Bandit algorithm options" << endl;
	
	cout << endl;
	cout << "For weak learners specific options type:" << endl;
	
	vector<string>::const_iterator it;
	for (it = learnersList.begin(); it != learnersList.end(); ++it)
		cout << "   --h " << *it << endl;
	
	exit(0);
}

//---------------------------------------------------------------------------

/**
 * Show the help for the options.
 * \param args The arguments structure.
 * \date 28/11/2005
 */
void showOptionalHelp(nor_utils::Args& args)
{
	string helpType = args.getValue<string>("h", 0);
	
	cout << "MultiBoost (v" << CURRENT_VERSION << "). An obvious name for a multi-class AdaBoost learner." << endl;
	cout << "---------------------------------------------------------------------------" << endl;
	
	if (helpType == "general")
		args.printGroup("General Options");
	else if (helpType == "io")
		args.printGroup("I/O Options");
	else if (helpType == "algo")
		args.printGroup("Basic Algorithm Options");
	else if (helpType == "bandits")
		args.printGroup("Bandit Algorithm Options");
	else if ( BaseLearner::RegisteredLearners().hasLearner(helpType) )
		args.printGroup(helpType + " Options");
	else
		cerr << "ERROR: Unknown help section <" << helpType << ">" << endl;
}

//---------------------------------------------------------------------------

/**
 * Show the default values.
 * \date 11/11/2005
 */
void showStaticConfig()
{
	cout << "MultiBoost (v" << CURRENT_VERSION << "). An obvious name for a multi-class AdaBoost learner." << endl;
	cout << "------------------------ STATIC CONFIG -------------------------" << endl;
	
	cout << "- Sort type = ";
#if CONSERVATIVE_SORT
	cout << "CONSERVATIVE (slow)" << endl;
#else
	cout << "NON CONSERVATIVE (fast)" << endl;
#endif
	
	cout << "Comment: " << COMMENT << endl;
#ifndef NDEBUG
	cout << "Important: NDEBUG not active!!" << endl;
#endif
	
#if MB_DEBUG
	cout << "MultiBoost debug active (MB_DEBUG=1)!!" << endl;
#endif
	
	exit(0);  
}

//---------------------------------------------------------------------------

void setBasicOptions(nor_utils::Args& args)
{	
	args.setArgumentDiscriminator("--");
	
	args.declareArgument("help");
	args.declareArgument("static");
	
	args.declareArgument("h", "Help", 1, "<optiongroup>");
	
	//////////////////////////////////////////////////////////////////////////
	// Basic Arguments
	
	args.setGroup("Parameters");
	
	args.declareArgument("train", "Performs training.", 2, "<dataFile> <nInterations>");
	args.declareArgument("traintestmdp", "Performs training and test at the same time.", 5, "<trainingDataFile> <testDataFile> <nInterations> <shypfile> <outfile>");
    args.declareArgument("testmdp", "Performs test of a previously leant model.", 3, "<qtable> <train log file> <test log file>");
	args.declareArgument("test", "Test the model.", 3, "<dataFile> <numIters> <shypFile>");
	args.declareArgument("test", "Test the model and output the results", 4, "<datafile> <shypFile> <numIters> <outFile>");
	args.declareArgument("cmatrix", "Print the confusion matrix for the given model.", 2, "<dataFile> <shypFile>");
	args.declareArgument("cmatrixfile", "Print the confusion matrix with the class names to a file.", 3, "<dataFile> <shypFile> <outFile>");
	args.declareArgument("posteriors", "Output the posteriors for each class, that is the vector-valued discriminant function for the given dataset and model.", 4, "<dataFile> <shypFile> <outFile> <numIters>");
	args.declareArgument("posteriors", "Output the posteriors for each class, that is the vector-valued discriminant function for the given dataset and model periodically.", 5, "<dataFile> <shypFile> <outFile> <numIters> <period>");	
	args.declareArgument("cposteriors", "Output the calibrated posteriors for each class, that is the vector-valued discriminant function for the given dataset and model.", 4, "<dataFile> <shypFile> <outFile> <numIters>");
	
	args.declareArgument("likelihood", "Output the likelihoof of data for each iteration, that is the vector-valued discriminant function for the given dataset and model.", 4, "<dataFile> <shypFile> <outFile> <numIters>");
	
	args.declareArgument("encode", "Save the coefficient vector of boosting individually on each point using ParasiteLearner", 6, "<inputDataFile> <autoassociativeDataFile> <outputDataFile> <nIterations> <poolFile> <nBaseLearners>");
	args.declareArgument("roc", "Print out the ROC curve (it calculate the ROC curve for the first class)", 4, "<dataFile> <shypFile> <outFile> <numIters>" );
	
	args.declareArgument("ssfeatures", "Print matrix data for SingleStump-Based weak learners (if numIters=0 it means all of them).", 4, "<dataFile> <shypFile> <outFile> <numIters>");
	
	args.declareArgument( "fileformat", "Defines the type of intput file. Available types are:\n" 
						 "* simple: each line has attributes separated by whitespace and class at the end (DEFAULT!)\n"
						 "* arff: arff filetype. The header file can be specified using --arffheader option\n"
						 "* arffbzip: bziped arff filetype. The header file can be specified using --arffheader option\n"
						 "* svmlight: \n"
						 "(Example: --fileformat simple)",
                         1, "<fileFormat>" );
	
	args.declareArgument("headerfile", "The filename of the header file (SVMLight).", 1, "header.txt");
	
	args.declareArgument("constant", "Check constant learner in each iteration.", 0, "");
	args.declareArgument("timelimit", "Time limit in minutes", 1, "<minutes>" );
	args.declareArgument("stronglearner", "Strong learner. Available strong learners:\n"
						 "AdaBoost (default)\n"
						 "BrownBoost\n", 1, "<stronglearner>" );
	
	args.declareArgument("slowresumeprocess", "Compute the results in each iteration (slow resume)\n"
						 "Compute only the data of the last iteration (fast resume, default)\n", 0, "" );
	args.declareArgument("weights", "Outputs the weights of instances at the end of the learning process", 1, "<filename>" );
	args.declareArgument("Cn", "Resampling size for FilterBoost (default=300)", 1, "<val>" );
	//// ignored for the moment!
	//args.declareArgument("arffheader", "Specify the arff header.", 1, "<arffHeaderFile>");
	
	//////////////////////////////////////////////////////////////////////////
	// Options
	
	args.setGroup("I/O Options");
	
	/////////////////////////////////////////////
	// these are valid only for .txt input!
	// they might be removed!
	args.declareArgument("d", "The separation characters between the fields (default: whitespaces).\nExample: -d \"\\t,.-\"\nNote: new-line is always included!", 1, "<separators>");
	args.declareArgument("classend", "The class is the last column instead of the first (or second if -examplelabel is active).");
	args.declareArgument("examplename", "The data file has an additional column (the very first) which contains the 'name' of the example.");
	
	/////////////////////////////////////////////
	
	args.setGroup("Basic Algorithm Options");
	args.declareArgument("weightpolicy", "Specify the type of weight initialization. The user specified weights (if available) are used inside the policy which can be:\n"
						 "* sharepoints Share the weight equally among data points and between positiv and negative labels (DEFAULT)\n"
						 "* sharelabels Share the weight equally among data points\n"
						 "* proportional Share the weights freely", 1, "<weightType>");
	
	
	args.setGroup("General Options");
	
	args.declareArgument("verbose", "Set the verbose level 0, 1 or 2 (0=no messages, 1=default, 2=all messages).", 1, "<val>");
	args.declareArgument("outputinfo", "Output informations on the algorithm performances during training, on file <filename>.", 1, "<filename>");
	args.declareArgument("seed", "Defines the seed for the random operations.", 1, "<seedval>");
	
	//////////////////////////////////////////////////////////////////////////
	// Options for TL tool
	args.setGroup("RL options");
	
	args.declareArgument("gridworldfilename", "The naem of gridwold description filename", 1, "<val>");
	args.declareArgument("episodes", "The number of episodes", 2, "<episod> <testiter>");
	args.declareArgument("rewards", "success, class, skip", 3, "<succ> <class> <skip>");
	args.declareArgument("logdir", "Dir of log", 1, "<dir>");
	args.declareArgument("succrewartdtype", "The mode of the reward calculation", 1, "<mode>" );
	args.declareArgument("statespace", "The statespace representation", 1, "<mode>" );
	args.declareArgument("numoffeat", "The number of feature in statespace representation", 1, "<featnum>" );
    args.declareArgument("optimistic", "Set the initial values of the Q function", 3, "<real> <real> <real>" );
    args.declareArgument("etrace", "Lambda parameter", 1, "<real>" );
    args.declareArgument("rbfbias", "Set the bias of the RBF network in the QTable representation", 3, "<real> <real> <real>" );
    args.declareArgument("noaddcenter", "Disactivate adding centers on TD error", 0, "" );
    args.declareArgument("normrbf", "Normalized RBFs", 0, "" );
    args.declareArgument("rbfsigma", "Initialize RBF with a given sigma", 1, "<val>" );
    args.declareArgument("maxtderr", "Max error on the TD value for adding a center. It is given by the inverse of the ratio of the max Q", 1, "<val>" );
    args.declareArgument("minrbfact", "Min activation factor for adding a center", 1, "<val>" );
    args.declareArgument("positivelabel", "The name of positive label", 1, "<labelname>" );
    args.declareArgument("failpenalties", "Negative rewards for misclassifying resp. positives and negatives", 2, "<pospenalty> <negpenalty>" );
    args.declareArgument("learningrate", "The learning rate", 3, "<numerator> <denominator> <denominator increment>" );
    args.declareArgument("explorationrate", "The exploration rate", 3, "<numerator> <denominator> <denominator increment>" );
    args.declareArgument("paramupdate", "The number of episodes required before updating the learning rate and the exploration rate.", 1, "<num>" );
    args.declareArgument("withoutquitQ", "Take the quit action off.", 0, "" );
    args.declareArgument("maxrbfnumber", "The maximum number of RBF per whyp per action.", 1, "<num>" );
    args.declareArgument("incrementalrewardQ", "Give a reward after each evalation.", 0, "" );
    args.declareArgument("qtable", "Load the GSBNF from a file.", 1, "<file>" );
}


//---------------------------------------------------------------------------




// This is the entry point for this application


int main(int argc, const char *argv[])
{
	int steps = 0;
    int ges_failed = 0, ges_succeeded = 0, last_succeeded = 0;	
    int totalSteps = 0;
    
	// Initialize the random generator 
	srand((unsigned int) time(NULL));
	
	// no need to synchronize with C style stream
	std::ios_base::sync_with_stdio(false);
	
#if STABLE_SORT
	cerr << "WARNING: Stable sort active! It might be slower!!" << endl;
#endif
	
	
	//////////////////////////////////////////////////////////////////////////
	// Standard arguments
	nor_utils::Args args;
	
	//////////////////////////////////////////////////////////////////////////
	// Define basic options	
	setBasicOptions(args);
	
	
	//////////////////////////////////////////////////////////////////////////
	// Shows the list of available learners
	string learnersComment = "Available learners are:";
	
	vector<string> learnersList;
	BaseLearner::RegisteredLearners().getList(learnersList);
	vector<string>::const_iterator it;
	for (it = learnersList.begin(); it != learnersList.end(); ++it)
	{
		learnersComment += "\n ** " + *it;
		// defaultLearner is defined in Defaults.h
		if ( *it == defaultLearner )
			learnersComment += " (DEFAULT)";
	}
	
	args.declareArgument("learnertype", "Change the type of weak learner. " + learnersComment, 1, "<learner>");
	
	//////////////////////////////////////////////////////////////////////////
	//// Declare arguments that belongs to all weak learners
	BaseLearner::declareBaseArguments(args);
	
	////////////////////////////////////////////////////////////////////////////
	//// Weak learners (and input data) arguments
	for (it = learnersList.begin(); it != learnersList.end(); ++it)
	{
		args.setGroup(*it + " Options");
		// add weaklearner-specific options
		BaseLearner::RegisteredLearners().getLearner(*it)->declareArguments(args);
	}
	
	//////////////////////////////////////////////////////////////////////////
	//// Declare arguments that belongs to all bandit learner
	GenericBanditAlgorithm::declareBaseArguments(args);
	
	
	//////////////////////////////////////////////////////////////////////////////////////////  
	//////////////////////////////////////////////////////////////////////////////////////////
	
	switch ( args.readArguments(argc, argv) )
	{
		case nor_utils::AOT_NO_ARGUMENTS:
			showBase();
			break;
			
		case nor_utils::AOT_UNKOWN_ARGUMENT:
			exit(1);
			break;
			
		case nor_utils::AOT_INCORRECT_VALUES_NUMBER:
			exit(1);
			break;
			
		case nor_utils::AOT_OK:
			break;
	}
	
	//////////////////////////////////////////////////////////////////////////////////////////  
	//////////////////////////////////////////////////////////////////////////////////////////
	
	if ( args.hasArgument("help") )
		showHelp(args, learnersList);
	if ( args.hasArgument("static") )
		showStaticConfig();
	
	//////////////////////////////////////////////////////////////////////////////////////////  
	//////////////////////////////////////////////////////////////////////////////////////////
	
	if ( args.hasArgument("h") )
		showOptionalHelp(args);
	
	//////////////////////////////////////////////////////////////////////////////////////////  
	//////////////////////////////////////////////////////////////////////////////////////////
	
	int verbose = 1;
	
	if ( args.hasArgument("verbose") )
		args.getValue("verbose", 0, verbose);
	
	//////////////////////////////////////////////////////////////////////////////////////////  
	//////////////////////////////////////////////////////////////////////////////////////////
	
	// defines the seed
	if (args.hasArgument("seed"))
	{
		unsigned int seed = args.getValue<unsigned int>("seed", 0);
		srand(seed);
	}
	
	//////////////////////////////////////////////////////////////////////////////////////////  
	//////////////////////////////////////////////////////////////////////////////////////////
	string gridworldFileName = "Gridworld_10x10.txt";
	
	// Console Input Processing
	if (verbose>5)
	{
		char *debugFile = "debug.txt";
		DebugInit("debug.txt", "+", false);
	}
	
	if (args.hasArgument("gridworldfilename"))
	{
		args.getValue("gridworldfilename", 0, gridworldFileName);
	}
	
	
	int evalTestIteration=0;
	int episodeNumber = 0;
	
	if (args.hasArgument("episodes"))
	{
		episodeNumber = args.getValue<int>("episodes", 0);				
		evalTestIteration = args.getValue<int>("episodes", 1);				
	}
	else {
		cout << "No episode argiment!!!!" << endl;
		exit(-1);
	}
	
	string logDirContinous="";
	if (args.hasArgument("logdir"))
	{
		logDirContinous = args.getValue<string>("logdir", 0);
	}
	
	DataReader* datahandler = new DataReader( args, verbose );
	datahandler->setCurrentDataToTrain();
	
    double epsNumerator = 1.;    
	double epsDivisor = 4.0;
    double epsIncrement = 0.1;

    double qRateNumerator = 0.2 ;
	double qRateDivisor = 1.;
    double qRateIncrement = 1;

    
    if (args.hasArgument("learningrate"))
	{
		qRateNumerator = args.getValue<double>("learningrate", 0);
		qRateDivisor = args.getValue<double>("learningrate", 1);
        qRateIncrement = args.getValue<double>("learningrate", 2);
	}

    if (args.hasArgument("explorationrate"))
	{
		epsNumerator = args.getValue<double>("explorationrate", 0);
		epsDivisor = args.getValue<double>("explorationrate", 1);
		epsIncrement = args.getValue<double>("explorationrate", 2);
	}
    

    double currentEpsilon = epsNumerator / epsDivisor;
    double currentAlpha = qRateNumerator / qRateDivisor;
    
    double lambdaParam = 0.95;
    
    int paramUpdate = 10000;
    if (args.hasArgument("paramupdate"))
	{
		paramUpdate = args.getValue<double>("paramupdate", 0);
    }
    
	if ( datahandler->getClassNumber() <= 2 )
	{
		cout << "*****Binary mode******" << endl << flush;
		//// new version for binary classification
		//AdaBoostMDPClassifierContinous* classifierContinous = new AdaBoostMDPClassifierContinous(args, verbose, datahandler, datahandler->getClassNumber(), 0 );
		AdaBoostMDPClassifierContinousBinary* classifierContinous = new AdaBoostMDPClassifierContinousBinary(args, verbose, datahandler );
		//AdaBoostMDPClassifierDiscrete* classifierContinous = new AdaBoostMDPClassifierDiscrete(args, verbose, datahandler );
		
		//CAbstractStateDiscretizer* adaBooststate = new CGlobalGridWorldDiscreteState(classifierContinous->getIterNum(), classifierContinous->getNumClasses());
		
		
		// the gridworld model implements the reward function too, so we can use this
		CRewardFunction *rewardFunctionContinous = classifierContinous;
		
		// Create the agent in our environmentModel.
		CAgent *agentContinous = new CAgent(classifierContinous);
		
		// Add all possible Actions to the agent
		// skip
		agentContinous->addAction(new CAdaBoostAction(0)); 
		// classify
		agentContinous->addAction(new CAdaBoostAction(1));
        
        if (!args.hasArgument("withoutquitQ"))
        {
            // jump to the end	
    		agentContinous->addAction(new CAdaBoostAction(2));
        }
		
		CStateModifier* discState = NULL;
		// simple discretized state space
		//CStateModifier* discState = classifierContinous->getStateSpace();
		int featnum = 11;
		if ( args.hasArgument("numoffeat") ) 
            featnum = args.getValue<int>("numoffeat", 0);
        
		if ( args.hasArgument("etrace") ) 
            lambdaParam = args.getValue<double>("etrace", 0);
        
        
        
		CRBFCenterFeatureCalculator* rbfFC;
        CRBFCenterNetwork* rbfNW;
        CAbstractQFunction * qData;
        
        int sptype = -1;
		if ( args.hasArgument("statespace") )
		{
			sptype = args.getValue<int>("statespace", 0);
			
            if (sptype == 2 ) {
//                CStateModifier * nnStateModifier = classifierContinous->getStateSpaceNN();
//                
//                agentContinous->addStateModifier(nnStateModifier);
//                
//                qData = new CQFunction(agentContinous->getActions());
//                
//                CActionSet * actionSet = agentContinous->getActions();
//                CActionSet::iterator acIt = actionSet->begin();
//                
//                for (; acIt != actionSet->end(); ++acIt) {
//                    
//                    //                    MLP * gm = new MLP(3, 2, "linear", featnum, "sigmoid", featnum, "linear" , 1);
//                    Linear layer1(2, featnum);
//                    LogRBF layer2(featnum, featnum);
//                    Linear layer3(featnum, 1);
//                    
//                    ConnectedMachine gm;
//                    gm.addFCL(&layer1);
//                    gm.addFCL(&layer2);
//                    gm.addFCL(&layer3);
//                    
//                    gm.build();
//                    
//                    CTorchGradientFunction * torchGradientFunction = new CTorchGradientFunction(&gm);
//                    CVFunctionFromGradientFunction* vFunction = new CVFunctionFromGradientFunction(torchGradientFunction,  agentContinous->getStateProperties());
//                    static_cast<CQFunction*>(qData)->setVFunction(*acIt, vFunction);
//                    
//                }
                
            }
            else {
                if (sptype==0) {
                    discState = classifierContinous->getStateSpace(featnum);
					agentContinous->addStateModifier(discState);
					qData = new CFeatureQFunction(agentContinous->getActions(), discState);					
                }
                else if (sptype==1) {
                    discState = classifierContinous->getStateSpaceRBF(featnum);
					agentContinous->addStateModifier(discState);
					qData = new CFeatureQFunction(agentContinous->getActions(), discState);					
                }
                else if (sptype==3) {
                    discState = classifierContinous->getStateSpaceRBFAdaptiveCenters(featnum, &rbfFC, &rbfNW);
					agentContinous->addStateModifier(discState);
					qData = new CFeatureQFunction(agentContinous->getActions(), discState);					
                }
				else if (sptype==4) {
					discState = classifierContinous->getStateSpaceForRBFQFunction(featnum);
					agentContinous->addStateModifier(discState);
					qData = new RBFBasedQFunctionBinary(agentContinous->getActions(), discState);
                    
                    double initRBFs[] = {1.0,1.0,1.0};
                    if ( args.hasArgument("optimistic") )
                    {
                        assert(args.getNumValues("optimistic") == 3);
                        initRBFs[0] = args.getValue<double>("optimistic", 0);	
                        initRBFs[1] = args.getValue<double>("optimistic", 1);	
                        initRBFs[2] = args.getValue<double>("optimistic", 2);	
                    }
                    
                    dynamic_cast<RBFBasedQFunctionBinary*>( qData )->uniformInit(initRBFs);
                    
                    dynamic_cast<RBFBasedQFunctionBinary*>( qData )->setMuAlpha(1) ;
                    dynamic_cast<RBFBasedQFunctionBinary*>( qData )->setMuMean(0) ;
                    dynamic_cast<RBFBasedQFunctionBinary*>( qData )->setMuSigma(0) ;
                    
				}
                else if (sptype>=5 && sptype <= 6) {
					discState = classifierContinous->getStateSpaceForRBFQFunction(featnum);
					agentContinous->addStateModifier(discState);
					qData = new GSBNFBasedQFunction(agentContinous->getActions(), discState);
                    
                    double initRBFs[] = {1.0,1.0,1.0};
                    if ( args.hasArgument("optimistic") )
                    {
                        assert(args.getNumValues("optimistic") == 3);
                        initRBFs[0] = args.getValue<double>("optimistic", 0);	
                        initRBFs[1] = args.getValue<double>("optimistic", 1);	
                        initRBFs[2] = args.getValue<double>("optimistic", 2);	
                    }

                    vector<double> bias(3);
                    if ( args.hasArgument("rbfbias") )
                    {
                        assert(args.getNumValues("rbfbias") == 3);
                        bias[0] = args.getValue<double>("rbfbias", 0);	
                        bias[1] = args.getValue<double>("rbfbias", 1);	
                        bias[2] = args.getValue<double>("rbfbias", 2);	
                    }
                    
                    if (args.hasArgument("qtable")) {
                        cout << "Loading Q-Table..." << endl;
                        dynamic_cast<GSBNFBasedQFunction*>( qData )->loadQFunction(args.getValue<string>("qtable", 0));
                    }
                    
                    dynamic_cast<GSBNFBasedQFunction*>( qData )->setBias(bias);
                    
                    int addCenter = 1;
                    if ( args.hasArgument("noaddcenter") )
                        addCenter = 0;

                    int normalizeRbf = 0;
                    if ( args.hasArgument("normrbf") )
                        normalizeRbf = 1;
                    
                    double initSigma = 0.01;
                    if ( args.hasArgument("rbfsigma") )
                        initSigma = args.getValue<double>("rbfsigma", 0);
                    
                    int maxtderr = 10;
                    if ( args.hasArgument("maxtderr") )
                        maxtderr = args.getValue<int>("maxtderr", 0);
                    
                    double minact = 0.4;
                    if ( args.hasArgument("minrbfact") )
                        minact = args.getValue<double>("minrbfact", 0);

                    int maxrbfnumber = 1000;
                    if ( args.hasArgument("maxrbfnumber") )
                        maxrbfnumber = args.getValue<int>("MaxRBFNumber", 0);
                    
                    qData->setParameter("AddCenterOnError", addCenter);
                    qData->setParameter("NormalizedRBFs", normalizeRbf);
                    qData->setParameter("InitRBFSigma", initSigma); 
                    qData->setParameter("MaxTDErrorDivFactor", maxtderr);
                    qData->setParameter("MinActivation", minact);
                    qData->setParameter("QLearningRate", currentAlpha);
                    qData->setParameter("MaxRBFNumber", maxrbfnumber);
                    
                    dynamic_cast<GSBNFBasedQFunction*>( qData )->uniformInit(initRBFs);
                    
                    dynamic_cast<GSBNFBasedQFunction*>( qData )->setMuAlpha(1) ;
                    dynamic_cast<GSBNFBasedQFunction*>( qData )->setMuMean(0.000) ;
                    dynamic_cast<GSBNFBasedQFunction*>( qData )->setMuSigma(0.000) ;
                    
				
                    if (sptype==6) {
                        

//                        CFeatureVFunction *vFunctionCritic = new CFeatureVFunction(agentContinous);
//                        CFeatureVFunction *vFunctionActor = new CFeatureVFunction(agentContinous);
//                        
//                        CVFunctionLearner *vFunctionLearner = new CVFunctionLearner(classifierContinous, vFunctionCritic);
//                        
//                        CActorFromQFunction *actor = new CActorFromQFunction(qData);
//                        
//                        CAgentController *vLearnerPolicyAB = new CVMStochasticPolicy(agentAB->getActions(), new CEpsilonGreedyDistribution(1.0), vFunctionAB, classifier, rewardFunctionAB, agentAB->getStateModifiers());
//                        
//                        // Set some options of the Etraces which are not default
//                        vFunctionLearnerAB->setParameter("ReplacingETraces", 1.0);
//                        vFunctionLearnerAB->setParameter("Lambda", 0.95);
//                        
//                        // Add the learner to the agent listener list, so he can learn from the agent's steps.
//                        agentAB->addSemiMDPListener(vFunctionLearnerAB);
//                        
//                        // Set the controller of the agent
//                        agentAB->setController(vLearnerPolicyAB);
                    }
                }
                else {
                    cout << "unkown statespcae" << endl;
                }
                
                
                
                // optimistic intial values
                //                const int numWeights = qData->getNumWeights();
                //                double weights[numWeights];
                //                qData->getWeights(weights);
                //                
                //                for (int i = numWeights/3; i < numWeights*2/3; ++i) {
                //                    weights[i] = 0;//2000;
                //                }
                //                qData->setWeights(weights);
                
            }
		} else {
			cout << "No state space resresantion is given, the default is used" << endl;
			discState = classifierContinous->getStateSpace(featnum);
		}
        
        
		//CStateModifier* discState = classifierContinous->getStateSpace(100,0.01);
		// RBF
		//CFeatureCalculator* discState = classifierContinous->getStateSpaceRBF(5);
		// add the discrete state to the agent's state modifier
		// discState must not be modified (e.g. with a State-Substitution) by now
        //		agentContinous->addStateModifier(discState);
		
		
		//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		
		// Monte Carlo planning starts here	
		// ADDED TODO LIST
		
		
		//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		
		// SARSA Q-Learning starts here	
		// Create our Q-Function, we will use a Feature Q-Function, which is table-like representation of the Q-Function.
		// The Q-Function needs to know which actions and which state it has to use
        
        //		CFeatureQFunction *qData = new CFeatureQFunction(agentContinous->getActions(), discState);
        
        //obsolete
        //        CQFunctionFromGradientFunction *qData = new CQFunctionFromGradientFunction(new CContinuousAction(new CContinuousActionProperties(0)), torchGradientFunction, agentContinous->getActions(), classifierContinous->getStateProperties());
        
        CTDLearner *qFunctionLearner = new CQLearner(classifierContinous, qData);
//		CSarsaLearner *qFunctionLearner = new CSarsaLearner(rewardFunctionContinous, qData, agentContinous);
        
        //gradient stuff !!!
        CDiscreteResidual* residualFunction = new CDiscreteResidual(0.95);
        CConstantBetaCalculator* betaCalculator = new CConstantBetaCalculator(1);
        //        CVariableBetaCalculator * betaCalculator = new CVariableBetaCalculator(0.1, 0.99) ; //mu and maxBeta
        CResidualBetaFunction* residualGradient = new CResidualBetaFunction(betaCalculator, residualFunction);
        
        
        //        CTDGradientLearner *qFunctionLearner = new CTDGradientLearner(rewardFunctionContinous, qData, agentContinous, residualFunction, residualGradient);
        //        CTDResidualLearner *qFunctionLearner = new CTDResidualLearner(rewardFunctionContinous, dynamic_cast<CGradientQFunction*>(qData), agentContinous, residualFunction, residualGradient, betaCalculator);
        
        // Create the Controller for the agent from the QFunction. We will use a EpsilonGreedy-Policy for exploration.

        CAgentController *policy = new CQStochasticPolicy(agentContinous->getActions(), new CEpsilonGreedyDistribution(currentEpsilon), qData);
//        CAgentController *policy = new CQStochasticPolicy(agentContinous->getActions(), new CSoftMaxDistribution(currentEpsilon), qData);
        
        
		// Set some options of the Etraces which are not default
        qFunctionLearner->setParameter("ReplacingETraces", 1.0);
		qFunctionLearner->setParameter("Lambda", lambdaParam);
		qFunctionLearner->setParameter("DiscountFactor", 1.0);
        
        qFunctionLearner->setParameter("QLearningRate", currentAlpha);
        qData->setParameter("QLearningRate", currentAlpha);
        
        //        CVFunctionFromGradientFunction* vFunctionAB = new CVFunctionFromGradientFunction(torchGradientFunction,  nnStateModifier); //classifierContinous->getStateProperties()
        //        CVFunctionLearner *vFunctionLearnerAB = new CVFunctionLearner(rewardFunctionContinous, vFunctionAB);
        //        CAgentController *vLearnerPolicyAB = new CVMStochasticPolicy(agentContinous->getActions(), new CEpsilonGreedyDistribution(1.0), vFunctionAB, classifierContinous, rewardFunctionContinous, agentContinous->getStateModifiers() );
        //        vFunctionLearnerAB->setParameter("ReplacingETraces", 1.0);
        //        vFunctionLearnerAB->setParameter("Lambda", 0.95);
        //        vFunctionLearnerAB->setParameter("DiscountFactor", 1.0);
        
		
		// Add the learner to the agent listener list, so he can learn from the agent's steps.
		agentContinous->addSemiMDPListener(qFunctionLearner);
		agentContinous->setController(policy);        
        
        //        agentContinous->addSemiMDPListener(vFunctionLearnerAB);
        //        agentContinous->setController(vLearnerPolicyAB);
        
        
		// disable automatic logging of the current episode from the agent
		agentContinous->setLogEpisode(false);
        
        int steps2 = 0;
		int usedClassifierNumber=0;
		int max_Steps = 100000;		
		double ovaccTrain, ovaccValid, ovaccTest;
		
		classifierContinous->setCurrentDataToTrain();
		ovaccTrain = classifierContinous->getAccuracyOnCurrentDataSet();
		classifierContinous->setCurrentDataToTest();
		ovaccValid = classifierContinous->getAccuracyOnCurrentDataSet();
        
        if (classifierContinous->setCurrentDataToTest2()) 
            ovaccTest = classifierContinous->getAccuracyOnCurrentDataSet();    
            
		classifierContinous->setCurrentDataToTrain();
        
        if (args.hasArgument("testmdp"))  
        {            
            agentContinous->removeSemiMDPListener(qFunctionLearner);
            
            CAgentController* greedypolicy = new CQGreedyPolicy(agentContinous->getActions(), qData);
            agentContinous->setController(greedypolicy);
            
            dynamic_cast<GSBNFBasedQFunction*>( qData )->loadQFunction(args.getValue<string>("testmdp", 0));
            
//            FILE *f = fopen("tmp.txt", "w");
//            dynamic_cast<GSBNFBasedQFunction*>(qData)->saveActionValueTable(f);
            
            classifierContinous->setCurrentDataToTrain();
            AdaBoostMDPBinaryDiscreteEvaluator<AdaBoostMDPClassifierContinousBinary> evalTrain( agentContinous, rewardFunctionContinous );
            BinaryResultStruct bres;
            bres.iterNumber=0;
            bres.origAcc = ovaccTrain;
            
            string logFileName = args.getValue<string>("testmdp", 1);
            evalTrain.classficationAccruacy(bres,logFileName);
            
            classifierContinous->setCurrentDataToTest();
            AdaBoostMDPBinaryDiscreteEvaluator<AdaBoostMDPClassifierContinousBinary> evalTest( agentContinous, rewardFunctionContinous );
            
            bres.origAcc = ovaccTest;
//            bres.iterNumber=0;
            string logFileName2 = args.getValue<string>("testmdp", 2);
            
            evalTest.classficationAccruacy(bres,logFileName2);			
            
            cout << "******** Overall Test accuracy by MDP: " << bres.acc << "(" << ovaccTest << ")" << endl;
            cout << "******** Average Test classifier used: " << bres.usedClassifierAvg << endl;
            cout << "******** Sum of rewards on Test: " << bres.avgReward << endl;
            
            exit(0);
        }
        
        
		//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		
		/*
		 // Create the learner and the Q-Function
		 CFeatureQFunction *qData = new CFeatureQFunction(agentContinous->getActions(), discState);
		 
		 CTDLearner *qFunctionLearner = new CQLearner(classifierContinous, qData);
		 // initialise the learning algorithm parameters
		 qFunctionLearner->setParameter("QLearningRate", 0.1);
		 qFunctionLearner->setParameter("DiscountFactor", 1.0);
		 qFunctionLearner->setParameter("ReplacingETraces", 0.05);
		 qFunctionLearner->setParameter("Lambda", 0.5);
		 
		 // Set the minimum value of a etrace, we need very small values
		 qFunctionLearner->setParameter("ETraceTreshold", 0.00001);
		 // Set the maximum size of the etrace list, standard is 100
		 qFunctionLearner->setParameter("ETraceMaxListSize", 1000);
		 
		 // add the Q-Learner to the listener list
		 agentContinous->addSemiMDPListener(qFunctionLearner);
		 
		 // Create the learners controller from the Q-Function, we use a SoftMaxPolicy
		 //CAgentController* policy = new CQStochasticPolicy(agentContinous->getActions(), new CEpsilonGreedyDistribution(0.01), qData);
		 CAgentController* policy = new CQStochasticPolicy(agentContinous->getActions(), new CRandomDistribution(0.1), qData);
		 
		 // set the policy as controller of the agent
		 agentContinous->setController(policy);
		 */
		
		//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		
		
		
		cout << "Valid: " << ovaccTrain << " Test: " << ovaccTest << endl << flush;
		
        double bestAcc=0., bestWhypNumber=0.;
        int bestEpNumber = 0;
        
        classifierContinous->outHeader();
                
		// Learn for 500 Episodes
		for (int i = 0; i < episodeNumber; i++)
		{
			//cout << qFunctionLearner->getParameter("QLearningRate") << endl;			
			// Do one training trial, with max max_Steps steps
			agentContinous->startNewEpisode();
			classifierContinous->setRandomizedInstance();
			steps2 = agentContinous->doControllerEpisode(1, max_Steps);
			
			//printf("Number fo classifier: %d\n", classifierContinous->getUsedClassifierNumber() );
			//printf("Episode %d %s with %d steps\n", i, classifierContinous->classifyCorrectly() ? "*succed*" : "*failed*", steps2);
			
			
			
			usedClassifierNumber += classifierContinous->getUsedClassifierNumber();
			
			bool clRes = classifierContinous->classifyCorrectly();				
			if ( clRes ) {
				//cout << "Classification result: CORRECT, instance: " << classifier->getCurrentInstance() << endl;
				ges_succeeded++;
			}
			else {
				ges_failed++;
				//cout << "Classification result: FAILED, instance: " << classifier->getCurrentInstance() << endl;			
			}
			
			
			
			if ((i>2)&&((i%1000)==0))
			{
				
				cout << "----------------------------------------------------------" << endl;
				cout << "Episode number: " << '\t' << i << endl;		
				cout << "Current Accuracy :" << '\t' << (((float)ges_succeeded / ((float)(ges_succeeded+ges_failed))) * 100.0) << endl;;
				cout << "Used Classifier  :" << '\t' << ((float)usedClassifierNumber / 1000.0) << endl;						
				cout << "Current alpha: " << currentAlpha << endl;
				cout << "Current Epsilon: " << currentEpsilon << endl;
				usedClassifierNumber = 0;
			}
			
			
			if ((i>2)&&((i%paramUpdate)==0))
			{	
				epsDivisor += epsIncrement;
				currentEpsilon =  epsNumerator / epsDivisor;
				policy->setParameter("EpsilonGreedy", currentEpsilon);
				//policy->setParameter("SoftMaxBeta", currentEpsilon);
                
			}
			if ((i>2)&&((i%paramUpdate)==0)) 
			{
				qRateDivisor += qRateIncrement;
				currentAlpha = qRateNumerator / qRateDivisor;
				qFunctionLearner->setParameter("QLearningRate", currentAlpha);
                qData->setParameter("QLearningRate", currentAlpha);
			}
			
			if ((i>2)&&((i%evalTestIteration)==0))
				//if ((i>2)&&((i%100)==0))
			{
				agentContinous->removeSemiMDPListener(qFunctionLearner);
				
				// set the policy to be greedy
				// Create the learners controller from the Q-Function, we use a SoftMaxPolicy
				CAgentController* greedypolicy = new CQGreedyPolicy(agentContinous->getActions(), qData);
				
				// set the policy as controller of the agent
				agentContinous->setController(greedypolicy);
				
                //#ifdef LOGQTABLE				
                //				cout << "####### BEGIN QTABLE" << endl;
                //				qData->printValues();
                //				cout << "####### END QTABLE" << endl;
                //#endif
                
				
				/*
                 sprintf( logfname, "./%s/qfunction_%d.txt", logDirContinous.c_str(), i );
                 FILE *vFuncFileAB = fopen(logfname,"w");
                 qData->saveData(vFuncFileAB);				
                 fclose(vFuncFileAB);
                 */
                
							
                // TRAIN stats
                
				classifierContinous->setCurrentDataToTrain();
				//AdaBoostMDPClassifierContinousBinaryEvaluator evalTrain( agentContinous, rewardFunctionContinous );
				AdaBoostMDPBinaryDiscreteEvaluator<AdaBoostMDPClassifierContinousBinary> evalTrain( agentContinous, rewardFunctionContinous );
				
				BinaryResultStruct bres;
				bres.origAcc = ovaccTrain;
				bres.iterNumber=i;
                
				evalTrain.classficationAccruacy(bres, "");

                cout << "[+] Training set results: " << endl;
				cout << "******** Overall accuracy by MDP: " << bres.acc << "(" << ovaccTrain << ")" << endl;
				cout << "******** Average classifier used: " << bres.usedClassifierAvg << endl;
				cout << "******** Sum of rewards: " << bres.avgReward << endl << endl;
				
                //                cout << "----> Best accuracy so far : " << bestAcc << endl << "----> Num of whyp used : " << bestWhypNumber << endl << endl;
                
				classifierContinous->outPutStatistic( bres );
				

                // VALID stats
                
                classifierContinous->setCurrentDataToTest();
				//AdaBoostMDPClassifierContinousBinaryEvaluator evalTrain( agentContinous, rewardFunctionContinous );
				AdaBoostMDPBinaryDiscreteEvaluator<AdaBoostMDPClassifierContinousBinary> evalValid( agentContinous, rewardFunctionContinous );
				
				bres.origAcc = ovaccValid;
				bres.iterNumber=i;
                
                string logFileName;
                if (!logDirContinous.empty()) {
                    char logfname[4096];
                    sprintf( logfname, "./%s/classValid_%d.txt", logDirContinous.c_str(), i );
                    logFileName = string(logfname);
                }
                
				evalValid.classficationAccruacy(bres, logFileName);
                
                cout << "[+] Validation set results: " << endl;
				cout << "******** Overall accuracy by MDP: " << bres.acc << "(" << ovaccValid << ")" << endl;
				cout << "******** Average classifier used: " << bres.usedClassifierAvg << endl;
				cout << "******** Sum of rewards: " << bres.avgReward << endl << endl;
				
                //                cout << "----> Best accuracy so far : " << bestAcc << endl << "----> Num of whyp used : " << bestWhypNumber << endl << endl;
                
				classifierContinous->outPutStatistic( bres );
                
                
                // TEST stats
						
				if (classifierContinous->setCurrentDataToTest2() )
                {
                    //AdaBoostMDPClassifierContinousBinaryEvaluator evalTest( agentContinous, rewardFunctionContinous );
                    AdaBoostMDPBinaryDiscreteEvaluator<AdaBoostMDPClassifierContinousBinary> evalTest( agentContinous, rewardFunctionContinous );
                    
                    bres.origAcc = ovaccTest;
                    
                    if (!logDirContinous.empty()) {
                        char logfname[4096];
                        sprintf( logfname, "./%s/classTest_%d.txt", logDirContinous.c_str(), i );
                        logFileName = string(logfname);
                    }
                    
                    evalTest.classficationAccruacy(bres,logFileName);			
                    
                    
                    //                ss.clear();
                    //                ss << "qtables/ActionTable_" << i << ".dta";
                    //                FILE *actionTableFile2 = fopen(ss.str().c_str(), "w");
                    //                dynamic_cast<RBFBasedQFunctionBinary*>(qData)->saveActionTable(actionTableFile2);
                    //                fclose(actionTableFile2);
                    
                    cout << "******** Overall Test accuracy by MDP: " << bres.acc << "(" << ovaccTest << ")" << endl;
                    cout << "******** Average Test classifier used: " << bres.usedClassifierAvg << endl;
                    cout << "******** Sum of rewards on Test: " << bres.avgReward << endl;
                    
                    cout << endl << "----> Best accuracy so far ( " << bestEpNumber << " ) : " << bestAcc << endl << "----> Num of whyp used : " << bestWhypNumber << endl << endl;
                    
                    classifierContinous->outPutStatistic( bres );
                }
                    
                if (sptype == 5) {
                    std::stringstream ss;
                    ss << "qtables/QTable_" << i << ".dta";
                    FILE *qTableFile2 = fopen(ss.str().c_str(), "w");
                    dynamic_cast<GSBNFBasedQFunction*>(qData)->saveActionValueTable(qTableFile2);
                    fclose(qTableFile2);                    
                }
                
				
                
                
                if (sptype==3)
                {
                    cout << "CENTERS : " << endl;                    
                    rbfFC->saveData(stdout);
                    //                for (int k = 0 ; k < rbfFC->getNumCenters(); ++k) {
                    //                    cout << dynamic_cast<CRBFCenterNetworkSimpleSearch*> ( rbfFC )->getCenter(k)->getCenter()[0] << "   ";
                    //                }
                    //                cout << endl;                    
                }
                
                if (bres.acc > bestAcc && sptype==0) {
                    bestEpNumber = i;
                    bestAcc = bres.acc;
                    bestWhypNumber = bres.usedClassifierAvg;
                }
                
                if ((bres.acc > bestAcc) && sptype==1) {
                    bestEpNumber = i;
                    bestAcc = bres.acc;
                    bestWhypNumber = bres.usedClassifierAvg;

                    //                    std::stringstream ss;
//                    ss << "qtables/QTable_" << i << ".dta";
//                    FILE* qTableFile = fopen(ss.str().c_str(), "w");
                    FILE* qTableFile = fopen("QTable.dta", "w");
                    dynamic_cast<CFeatureQFunction*>(qData)->saveFeatureActionValueTable(qTableFile);
                    fclose(qTableFile);
                    
//                    ss.clear();
//                    ss << "qtables/ActionTable_" << i << ".dta";
//                    FILE* actionTableFile = fopen(ss.str().c_str(), "w");
                    FILE* actionTableFile = fopen("ActionTable.dta", "w");
                    dynamic_cast<CFeatureQFunction*>(qData)->saveFeatureActionTable(actionTableFile);
                    fclose(actionTableFile);
                    
                    FILE *improvementLogFile = fopen("ImprovementLog.dta", "a");
                    fprintf(improvementLogFile, "%i\n", i);
                    fclose(improvementLogFile);
                    //TMP                    FILE* rbfDataFile = fopen("RBF.dta", "w");
                    //TMP                    dynamic_cast<CFeatureQFunction*>(qData)->saveParameters(rbfDataFile);
                    //TMP                    fclose(rbfDataFile);
                    
                    //                    const int numWeights = qData->getNumWeights();
                    //                    double weights[numWeights];
                    //                    qData->getWeights(weights);
                    //                    ofstream rbfWeights;
                    //                    rbfWeights.open("RBF.dta");
                    //                    for (int w = 0; w < qData->getNumWeights(); ++w) {
                    //                        rbfWeights << weights[w] << " ";
                    //                    }
                    //                    rbfWeights << endl << endl;

                }
                if ((bres.acc > bestAcc)&&(sptype==4)) {
                    bestEpNumber = i;
                    bestAcc = bres.acc;
                    bestWhypNumber = bres.usedClassifierAvg;
                    
                    FILE* qTableFile = fopen("QTable.dta", "w");
                    dynamic_cast<RBFBasedQFunctionBinary*>(qData)->saveActionValueTable(qTableFile);
                    fclose(qTableFile);
                    
//                    FILE* actionTableFile = fopen("ActionTable.dta", "w");
//                    dynamic_cast<RBFBasedQFunctionBinary*>(qData)->saveActionTable(actionTableFile);
//                    fclose(actionTableFile);
                    
                    FILE *improvementLogFile = fopen("ImprovementLog.dta", "a");
                    fprintf(improvementLogFile, "%i\n", i);
                    fclose(improvementLogFile);
                    

                    //					dynamic_cast<RBFBasedQFunctionBinary*>(qData)->saveQTable("QTable.dta");
				}

                if ((bres.acc > bestAcc)&&(sptype==5)) {
                    bestEpNumber = i;
                    bestAcc = bres.acc;
                    bestWhypNumber = bres.usedClassifierAvg;
                    
                    FILE* qTableFile = fopen("QTable.dta", "w");
                    dynamic_cast<GSBNFBasedQFunction*>(qData)->saveActionValueTable(qTableFile);
                    fclose(qTableFile);
                    
                    FILE *improvementLogFile = fopen("ImprovementLog.dta", "a");
                    fprintf(improvementLogFile, "%i\n", i);
                    fclose(improvementLogFile);
                    
				}
                
                //stringstream ss;
                //ss << "rbf/centers_" << i << ".txt";
                //FILE* pFile = fopen(ss.str().c_str(),"w");
                //rbfFC->saveData(pFile);
                
				
				//sprintf( logfname, "./%s/qfunction_%d_2.txt", logDirContinous.c_str(), i );
				//FILE *vFuncFileAB2 = fopen(logfname,"w");
				//qData->saveData(vFuncFileAB2);
				//fclose(vFuncFileAB2);
				
				
				
				agentContinous->setController(policy);
				//qData->printValues();
				agentContinous->addSemiMDPListener(qFunctionLearner);
				
				classifierContinous->setCurrentDataToTrain();
			}
			
			
		}
		////////////////////////////////////////////////////
		// multi-class
		////////////////////////////////////////////////////		
	} else { 
		cout << "*****Multi-class mode******" << endl << flush;
		
		AdaBoostMDPClassifierContinousMH* classifierContinous = new AdaBoostMDPClassifierContinousMH(args, verbose, datahandler, datahandler->getClassNumber() );		
		
		
		
		//CAbstractStateDiscretizer* adaBooststate = new CGlobalGridWorldDiscreteState(classifierContinous->getIterNum(), classifierContinous->getNumClasses());
		
		
		// the gridworld model implements the reward function too, so we can use this
		CRewardFunction *rewardFunctionContinous = classifierContinous;
		
		// Create the agent in our environmentModel.
		CAgent *agentContinous = new CAgent(classifierContinous);
		
		// Add all possible Actions to the agent
		// skip
		agentContinous->addAction(new CAdaBoostAction(0)); 
		// classify
		agentContinous->addAction(new CAdaBoostAction(1));
		
        if (!args.hasArgument("withoutquitQ"))
        {
            // jump to the end	
    		agentContinous->addAction(new CAdaBoostAction(2));
        }
        		
		
		CStateModifier* discState = NULL;
		// simple discretized state space
		//CStateModifier* discState = classifierContinous->getStateSpace();
		int featnum = 1;
		if ( args.hasArgument("numoffeat") ) featnum = args.getValue<int>("numoffeat", 0);			
		
		CRBFCenterFeatureCalculator* rbfFC;
//        CQFunction * qData;
        
        if ( args.hasArgument("etrace") ) 
            lambdaParam = args.getValue<double>("etrace", 0);

        CAbstractQFunction * qData;
        
		if ( args.hasArgument("statespace") )
		{
			int sptype = args.getValue<int>("statespace", 0);
			
            if (sptype == 2 ) {
                CStateModifier * nnStateModifier = classifierContinous->getStateSpaceNN();
                
                agentContinous->addStateModifier(nnStateModifier);
                
                qData = new CQFunction(agentContinous->getActions());
                
                CActionSet * actionSet = agentContinous->getActions();
                CActionSet::iterator acIt = actionSet->begin();
                
                for (; acIt != actionSet->end(); ++acIt) {
                    
//                    Torch::MLP * gm = new Torch::MLP(3, 2, "linear", featnum, "sigmoid", featnum, "linear" , 1);
//                    CTorchGradientFunction * torchGradientFunction = new CTorchGradientFunction(gm);
//                    CVFunctionFromGradientFunction* vFunction = new CVFunctionFromGradientFunction(torchGradientFunction,  agentContinous->getStateProperties());
//TMP                    qData->setVFunction(*acIt, vFunction);
                    
                }
                
            }
            else {
                if (sptype==0){
                    agentContinous->addStateModifier(discState);
                    qData = new CFeatureQFunction(agentContinous->getActions(), discState);
                    discState = classifierContinous->getStateSpaceExp(featnum,2.0);
                } 
                else if (sptype==1) {
                    discState = classifierContinous->getStateSpaceRBF(featnum);
                    agentContinous->addStateModifier(discState);
                    qData = new CFeatureQFunction(agentContinous->getActions(), discState);
                }
                
                else if (sptype==5) {
					discState = classifierContinous->getStateSpaceForGSBNFQFunction(featnum);
					agentContinous->addStateModifier(discState);
					qData = new GSBNFBasedQFunction(agentContinous->getActions(), discState);
                    
                    double initRBFs[] = {1.0,1.0,1.0};
                    if ( args.hasArgument("optimistic") )
                    {
                        assert(args.getNumValues("optimistic") == 3);
                        initRBFs[0] = args.getValue<double>("optimistic", 0);	
                        initRBFs[1] = args.getValue<double>("optimistic", 1);	
                        initRBFs[2] = args.getValue<double>("optimistic", 2);	
                    }
                    
                    vector<double> bias(3);
                    if ( args.hasArgument("rbfbias") )
                    {
                        assert(args.getNumValues("rbfbias") == 3);
                        bias[0] = args.getValue<double>("rbfbias", 0);	
                        bias[1] = args.getValue<double>("rbfbias", 1);	
                        bias[2] = args.getValue<double>("rbfbias", 2);	
                    }
                    
                    dynamic_cast<GSBNFBasedQFunction*>( qData )->setBias(bias);
                    
                    int addCenter = 1;
                    if ( args.hasArgument("noaddcenter") )
                        addCenter = 0;
                    
                    int normalizeRbf = 0;
                    if ( args.hasArgument("normrbf") )
                        normalizeRbf = 1;
                    
                    double initSigma = 0.01;
                    if ( args.hasArgument("rbfsigma") )
                        initSigma = args.getValue<double>("rbfsigma", 0);
                    
                    int maxtderr = 10;
                    if ( args.hasArgument("maxtderr") )
                        maxtderr = args.getValue<int>("maxtderr", 0);
                    
                    double minact = 0.4;
                    if ( args.hasArgument("minrbfact") )
                        minact = args.getValue<double>("minrbfact", 0);
                    
                    int maxrbfnumber = 100;
                    if ( args.hasArgument("maxrbfnumber") )
                        maxrbfnumber = args.getValue<int>("maxrbfnumber", 0);
                    
                    qData->setParameter("AddCenterOnError", addCenter);
                    qData->setParameter("NormalizedRBFs", normalizeRbf);
                    qData->setParameter("InitRBFSigma", initSigma); 
                    qData->setParameter("MaxTDErrorDivFactor", maxtderr);
                    qData->setParameter("MinActivation", minact);
                    qData->setParameter("QLearningRate", currentAlpha);
                    qData->setParameter("MaxRBFNumber", maxrbfnumber);
                    
                    dynamic_cast<GSBNFBasedQFunction*>( qData )->uniformInit(initRBFs);
                    
                    dynamic_cast<GSBNFBasedQFunction*>( qData )->setMuAlpha(1) ;
                    dynamic_cast<GSBNFBasedQFunction*>( qData )->setMuMean(0.000) ;
                    dynamic_cast<GSBNFBasedQFunction*>( qData )->setMuSigma(0.0000) ;
                    
				}
                
                else {
                    cout << "unkown statespcae" << endl;
                }
                
            }
		} else {
			cout << "No state space resresantion is given, the default is used" << endl;
			discState = classifierContinous->getStateSpace(featnum);
            agentContinous->addStateModifier(discState);
            qData = new CFeatureQFunction(agentContinous->getActions(), discState);

		}
		//CStateModifier* discState = classifierContinous->getStateSpace(100,0.01);
		// RBF
		//CFeatureCalculator* discState = classifierContinous->getStateSpaceRBF(5);
		// add the discrete state to the agent's state modifier
		// discState must not be modified (e.g. with a State-Substitution) by now
		//		agentContinous->addStateModifier(discState);
		
		
		//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		
		// Monte Carlo planning starts here	
		// ADDED TODO LIST
		
		
		//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		
		// SARSA Q-Learning starts here	
		// Create our Q-Function, we will use a Feature Q-Function, which is table-like representation of the Q-Function.
		// The Q-Function needs to know which actions and which state it has to use
		//CFeatureQFunction *qData = new CFeatureQFunction(agentContinous->getActions(), discState);
		
		// Create the Q-Function learner, we will use a SarsaLearner
		// The Sarsa Learner needs the reward function, the Q-Function and the agent.
		// The agent is used to get the estimation policy, because Sarsa Learning is On-Policy learning.
		//CSarsaLearner *qFunctionLearner = new CSarsaLearner(rewardFunctionContinous, qData, agentContinous);
        CTDLearner *qFunctionLearner = new CQLearner(classifierContinous, qData);
		
		// Create the Controller for the agent from the QFunction. We will use a EpsilonGreedy-Policy for exploration.
		CAgentController *policy = new CQStochasticPolicy(agentContinous->getActions(), new CEpsilonGreedyDistribution(currentEpsilon), qData);
		
		// Set some options of the Etraces which are not default
		qFunctionLearner->setParameter("ReplacingETraces", 1.0);
		qFunctionLearner->setParameter("Lambda", lambdaParam);
		qFunctionLearner->setParameter("DiscountFactor", 1.0);
        
        qFunctionLearner->setParameter("QLearningRate", currentAlpha);
		
		// Add the learner to the agent listener list, so he can learn from the agent's steps.
		agentContinous->addSemiMDPListener(qFunctionLearner);
		
		// Set the controller of the agent
		agentContinous->setController(policy);
		
		//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		
		/*
		 // Create the learner and the Q-Function
		 CFeatureQFunction *qData = new CFeatureQFunction(agentContinous->getActions(), discState);
		 
		 CTDLearner *learner = new CQLearner(classifierContinous, qData);
		 // initialise the learning algorithm parameters
		 learner->setParameter("QLearningRate", 0.1);
		 learner->setParameter("DiscountFactor", 1.0);
		 learner->setParameter("ReplacingETraces", 0.05);
		 learner->setParameter("Lambda", 1.0);
		 
		 // Set the minimum value of a etrace, we need very small values
		 learner->setParameter("ETraceTreshold", 0.00001);
		 // Set the maximum size of the etrace list, standard is 100
		 learner->setParameter("ETraceMaxListSize", 1000);
		 
		 // add the Q-Learner to the listener list
		 agentContinous->addSemiMDPListener(learner);
		 
		 // Create the learners controller from the Q-Function, we use a SoftMaxPolicy
		 CAgentController* policy = new CQStochasticPolicy(agentContinous->getActions(), new CEpsilonGreedyDistribution(0.9), qData);
		 
		 // set the policy as controller of the agent
		 agentContinous->setController(policy);
		 */
		
		//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		
		
		// disable automatic logging of the current episode from the agent
		agentContinous->setLogEpisode(false);
		
		int steps2 = 0;
		int usedClassifierNumber=0;
		int max_Steps = 100000;
		double ovacc = 0.0;
		double ovaccTrain, ovaccTest;
		
		classifierContinous->setCurrentDataToTrain();
		ovaccTrain = classifierContinous->getAccuracyOnCurrentDataSet();
		classifierContinous->setCurrentDataToTest();
		ovaccTest = classifierContinous->getAccuracyOnCurrentDataSet();
		classifierContinous->setCurrentDataToTrain();
		
		cout << "Valid: " << ovaccTrain << " Test: " << ovaccTest << endl << flush;
		
//		int epsDivisor = 1;
//		int qRateDivisor = 1;
        
        double bestAcc=0., bestWhypNumber=0.;
        int bestEpNumber = 0;
		// Learn for 500 Episodes
		for (int i = 0; i < episodeNumber; i++)
		{
			// set adaptive Epsilon
			//policy->setParameter("EpsilonGreedy", 0.1 / (i + 1));
			//policy->setParameter("EpsilonGreedy", 0.5 / epsDivisor);
			//learner->setParameter("QLearningRate", 0.1 / qRateDivisor);
			
			// Do one training trial, with max max_Steps steps
			agentContinous->startNewEpisode();
			classifierContinous->setRandomizedInstance();
			steps2 = agentContinous->doControllerEpisode(1, max_Steps);
			
			//printf("Number fo classifier: %d\n", classifierContinous->getUsedClassifierNumber() );
			//printf("Episode %d %s with %d steps\n", i, classifierContinous->classifyCorrectly() ? "*succed*" : "*failed*", steps2);
			
			
			
			usedClassifierNumber += classifierContinous->getUsedClassifierNumber();
			
			bool clRes = classifierContinous->classifyCorrectly();				
			if ( clRes ) {
				//cout << "Classification result: CORRECT, instance: " << classifier->getCurrentInstance() << endl;
				ges_succeeded++;
			}
			else {
				ges_failed++;
				//cout << "Classification result: FAILED, instance: " << classifier->getCurrentInstance() << endl;			
			}
			
			
			if ((i>2)&&((i%1000)==0))
			{
				
				cout << "----------------------------------------------------------" << endl;
				cout << "Episode number: " << '\t' << i << endl;		
				cout << "Current Accuracy :" << '\t' << (((float)ges_succeeded / ((float)(ges_succeeded+ges_failed))) * 100.0) << endl;;
				cout << "Used Classifier  :" << '\t' << ((float)usedClassifierNumber / 1000.0) << endl;						
				cout << "Current alpha: " << currentAlpha << endl;
				cout << "Current Epsilon: " << currentEpsilon << endl;
				usedClassifierNumber = 0;
                ges_failed = ges_succeeded = 0;
			}
			
			
			if ((i>2)&&((i%paramUpdate)==0))
			{	
				epsDivisor += epsIncrement;
				currentEpsilon =  epsNumerator / epsDivisor;
				policy->setParameter("EpsilonGreedy", currentEpsilon);				
			}
			if ((i>2)&&((i%paramUpdate)==0)) 
			{
				qRateDivisor += qRateIncrement;
				currentAlpha = qRateNumerator / qRateDivisor;
				qFunctionLearner->setParameter("QLearningRate", currentAlpha);
                qData->setParameter("QLearningRate", currentAlpha);                
			}
			
			
			if ((i>2)&&((i%evalTestIteration)==0))
				//if ((i>2)&&((i%100)==0))
			{				
				char logfname[4096];
				/*
                 sprintf( logfname, "./%s/qfunction_%d.txt", logDirContinous.c_str(), i );
                 FILE *vFuncFileAB = fopen(logfname,"w");
                 qData->saveData(vFuncFileAB);
                 fclose(vFuncFileAB);
                 */
				agentContinous->removeSemiMDPListener(qFunctionLearner);
				
				// Create the learners controller from the Q-Function, we use a SoftMaxPolicy
				CAgentController* greedypolicy = new CQGreedyPolicy(agentContinous->getActions(), qData);
				
				// set the policy as controller of the agent
				agentContinous->setController(greedypolicy);
				
				
				// TRAIN			
				classifierContinous->setCurrentDataToTrain();				
				AdaBoostMDPClassifierContinousEvaluator evalTrain( agentContinous, rewardFunctionContinous );
				
				double acc, usedclassifierNumber;			
				sprintf( logfname, "./%s/classValid_%d.txt", logDirContinous.c_str(), i );
				double sumRew = evalTrain.classficationAccruacy(acc,usedclassifierNumber,logfname);
                
                
                //save the number of centers per wc per action
                std::stringstream ss;
                ss << logDirContinous << "/rbfCenters_" << i << ".dta";
                FILE* rbfCentersFile = fopen(ss.str().c_str(), "w");
                dynamic_cast<GSBNFBasedQFunction*>(qData)->saveCentersNumber(rbfCentersFile);
                fclose(rbfCentersFile);
                
				cout << "******** Overall Train accuracy by MDP: " << acc << "(" << ovaccTrain << ")" << endl;
				cout << "******** Average Train classifier used: " << usedclassifierNumber << endl;
				cout << "******** Sum of rewards on Train: " << sumRew << endl << endl;
//				cout << "----> Best accuracy so far ( " << bestEpNumber << " ) : " << bestAcc << endl << "----> Num of whyp used : " << bestWhypNumber << endl;
                
				classifierContinous->outPutStatistic( ovaccTrain, acc, usedclassifierNumber, sumRew );
				
				
				// TEST
				classifierContinous->setCurrentDataToTest();
				AdaBoostMDPClassifierContinousEvaluator evalTest( agentContinous, rewardFunctionContinous);
				
				
				sprintf( logfname, "./%s/classTest_%d.txt", logDirContinous.c_str(), i );				
				sumRew = evalTest.classficationAccruacy(acc,usedclassifierNumber,logfname);		
                
                if (acc > bestAcc) {
                    bestAcc = acc;
                    bestWhypNumber = usedclassifierNumber;
                    bestEpNumber = i;
                }
                
				cout << "******** Overall Test accuracy by MDP: " << acc << "(" << ovaccTest << ")" << endl;
				cout << "******** Average Test classifier used: " << usedclassifierNumber << endl;
				cout << "******** Sum of rewards on Test: " << sumRew << endl;
                cout << "----> Best accuracy so far ( " << bestEpNumber << " ) : " << bestAcc << endl 
                     << "----> Num of whyp used : " << bestWhypNumber << endl;
				classifierContinous->outPutStatistic( ovaccTest, acc, usedclassifierNumber, sumRew );
				
				classifierContinous->setCurrentDataToTrain();
				
				/*
                 sprintf( logfname, "./%s/qfunction_%d_2.txt", logDirContinous.c_str(), i );
                 FILE *vFuncFileAB2 = fopen(logfname,"w");
                 qData->saveData(vFuncFileAB2);
                 fclose(vFuncFileAB2);
                 */
				
				agentContinous->addSemiMDPListener(qFunctionLearner);
                agentContinous->setController(policy);
			}
			
			
		}
		
		
		
	}
	
	//printf("\n\n<< Press Enter >>\n");
	//getchar();
	
	
	
	//// end of new version
	
	
	/*
	 AdaBoostMDPClassifier* classifier = new AdaBoostMDPClassifier(args,verbose);
	 classifier->init();
	 
	 if (args.hasArgument("rewards"))
	 {
	 double rew = args.getValue<double>("rewards", 0);
	 classifier->setRewardSuccess(rew); // classified correctly
	 
	 // the reward we incur if we use a classifier		
	 rew = args.getValue<double>("rewards", 1);
	 classifier->setClassificationReward( rew );
	 
	 rew = args.getValue<double>("rewards", 2);
	 classifier->setSkipReward( rew );		
	 
	 classifier->setJumpReward(0.0);
	 } else {
	 cout << "No rewards are given!" << endl;
	 exit(-1);
	 }
	 
	 string logDir="";
	 if (args.hasArgument("logdir"))
	 {
	 logDir = args.getValue<string>("logdir", 0);
	 }
	 
	 // Create the environment for the agent, the environment saves the current state of the agent.
	 CEnvironmentModel *environmentModelAB = new CTransitionFunctionEnvironment(classifier);
	 
	 // the gridworld model implements the reward function too, so we can use this
	 CRewardFunction *rewardFunctionAB = classifier;
	 
	 // Create the agent in our environmentModel.
	 CAgent *agentAB = new CAgent(environmentModelAB);
	 
	 // Add all possible Actions to the agent
	 // skip
	 agentAB->addAction(new CGridWorldActionAdaBoost(0)); 
	 // classify
	 agentAB->addAction(new CGridWorldActionAdaBoost(1));
	 // jump to the end	
	 //agentAB->addAction(new CGridWorldActionAdaBoost(2));
	 
	 
	 // For the shortest path problem, we need a global state, i.e. each possible position in the grid is an own state
	 CAbstractStateDiscretizer *globalGridworldstateAB = new CGlobalGridWorldDiscreteState(classifier->getSizeX(), classifier->getSizeY());
	 
	 // In order to use the discretizer we have to add it to the agent's state modifier list. 
	 // Always add your state modifiers to that list !!
	 agentAB->addStateModifier(globalGridworldstateAB);
	 
	 // Create an Agent Logger for logging the episodes
	 // Our agent logger logs the gridworld model state and the actions of the agent. It saves the episodes automatically in the given file. 
	 // Only one episode is hold in memory because we don't need the logged episodes now.
	 //CAgentLogger *loggerAB = new CAgentLogger(classifier->getStateProperties(), agentAB->getActions(), "AdaBoost.episodes", 1);
	 // add the logger to the agent's listener list
	 //agentAB->addSemiMDPListener(loggerAB);
	 
	 
	 //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	 //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	 //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	 
	 
	 // V-Learning starts here
	 
	 // Create our V-Function, we will use a Feature V-Function, which is table-like representation of the V-Function.
	 // The Q-Function needs to know which state it has to use
	 //CFeatureVFunction *vFunctionAB = new CFeatureVFunction(globalGridworldstateAB);
	 
	 // Create the V-Function learner, we will use a standard TD-Learner
	 // The V-Function Learner needs the reward function and of course the V-Function.
	 //CVFunctionLearner *vFunctionLearnerAB = new CVFunctionLearner(rewardFunctionAB, vFunctionAB);
	 
	 // Create the Controller for the agent from the VFunction and the gridworld model as state predictor. 
	 // Additionally to the V-Function the policy needs the gridworld model to calculate the next states for all actions 
	 // and the reward function to calculate the rewards for that states. It also needs the state modifiers list from the agent to get 
	 // the value from the V-Function.
	 // We will use a EpsilonGreedy-Policy for exploration.
	 //CAgentController *vLearnerPolicyAB = new CVMStochasticPolicy(agentAB->getActions(), new CEpsilonGreedyDistribution(1.0), vFunctionAB, classifier, rewardFunctionAB, agentAB->getStateModifiers());
	 
	 // Set some options of the Etraces which are not default
	 //vFunctionLearnerAB->setParameter("ReplacingETraces", 1.0);
	 //vFunctionLearnerAB->setParameter("Lambda", 0.95);
	 
	 // Add the learner to the agent listener list, so he can learn from the agent's steps.
	 //agentAB->addSemiMDPListener(vFunctionLearnerAB);
	 
	 // Set the controller of the agent
	 //agentAB->setController(vLearnerPolicyAB);
	 
	 
	 
	 
	 
	 
	 //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	 //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	 //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	 
	 // Q-Learning starts here
	 
	 // Create our Q-Function, we will use a Feature Q-Function, which is table-like representation of the Q-Function.
	 // The Q-Function needs to know which actions and which state it has to use
	 CFeatureQFunction *qFunction = new CFeatureQFunction(agentAB->getActions(), globalGridworldstateAB);
	 
	 // Create the Q-Function learner, we will use a SarsaLearner
	 // The Sarsa Learner needs the reward function, the Q-Function and the agent.
	 // The agent is used to get the estimation policy, because Sarsa Learning is On-Policy learning.
	 CSarsaLearner *qFunctionLearner = new CSarsaLearner(rewardFunctionAB, qFunction, agentAB);
	 
	 // Create the Controller for the agent from the QFunction. We will use a EpsilonGreedy-Policy for exploration.
	 CAgentController *qLearnerPolicy = new CQStochasticPolicy(agentAB->getActions(), new CEpsilonGreedyDistribution(0.05), qFunction);
	 
	 // Set some options of the Etraces which are not default
	 qFunctionLearner->setParameter("ReplacingETraces", 1.0);
	 qFunctionLearner->setParameter("Lambda", 0.95);
	 qFunctionLearner->setParameter("DiscountFactor", 1.0);
	 
	 // Add the learner to the agent listener list, so he can learn from the agent's steps.
	 agentAB->addSemiMDPListener(qFunctionLearner);
	 
	 // Set the controller of the agent
	 agentAB->setController(qLearnerPolicy);
	 //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	 //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	 //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	 
	 
	 // Disable logging of the current Episode
	 agentAB->setLogEpisode(false);
	 
	 ovacc = classifier->classifyTest();
	 // Start Learning, Learn 50 Episodes
	 for (int i = 0; i < episodeNumber; i++)
	 {
	 // Start a new Episode, the agent gets reseted in one of the start states
	 agentAB->startNewEpisode();
	 // Learn 1 Episode with maximal 2000 steps 
	 steps = agentAB->doControllerEpisode(1, classifier->getSizeX());
	 
	 bool clRes = classifier->currentClassifyingResult();
	 
	 
	 if ( clRes ) {
	 //cout << "Classification result: CORRECT, instance: " << classifier->getCurrentInstance() << endl;
	 ges_succeeded++;
	 }
	 else {
	 ges_failed++;
	 //cout << "Classification result: FAILED, instance: " << classifier->getCurrentInstance() << endl;			
	 }
	 
	 if ((i>2)&&((i%100)==0))
	 {
	 cout << "----------------------------------------------------------" << endl;
	 cout << "Episode number: " << '\t' << i << endl;		
	 cout << "Current Accuracy :" << '\t' << (((float)ges_succeeded / ((float)(ges_succeeded+ges_failed))) * 100.0) << endl;;
	 cout << "Used Classifier  :" << '\t' << classifier->getUsedClassifierNumber() << endl;						
	 }
	 
	 if ((i>2)&&((i%evalTestIteration)==0))
	 {
	 AdaBoostMDPClassifierEvaluator eval( agentAB, classifier );
	 //eval.setAgentController(qFunctionLearner->getEstimationPolicy());			
	 
	 char logfname[4096];
	 
	 // Save the QFunction
	 sprintf( logfname, "./%s/qfunction_%d.txt", logDir.c_str(), i );
	 FILE *vFuncFileAB = fopen(logfname,"w");
	 qFunction->saveData(vFuncFileAB);
	 fclose(vFuncFileAB);
	 
	 
	 // Save classfication result
	 sprintf( logfname, "./%s/classificationlog_%d.txt", logDir.c_str(), i );			
	 double acc, usedclassifierNumber;			
	 double sumRew = eval.classficationAccruacy(acc,usedclassifierNumber, logfname);
	 //double acc = classifier->classifyTestMDP(); not correct, it follows the training policy!!!
	 cout << "******** Overall accuracy by MDP: " << acc << "(" << ovacc << ")" << endl;
	 cout << "******** Average classifier used: " << usedclassifierNumber << endl;
	 cout << "******** Sum of rewards: " << sumRew << endl;
	 
	 classifier->outPutStatistic( ovacc, acc, usedclassifierNumber, sumRew );
	 
	 ges_succeeded=ges_failed=0;
	 
	 }
	 
	 // Check if the Episode failed
	 // The episode has failed if max_bounces has been reached (indicated through environmentModel->isFailed()), 
	 // or max_steps has been reached
	 
	 }
	 
	 
	 //printf("\n\n<< Press Enter >>\n");
	 //getchar();
	 
	 // Cleaning Up
	 //delete vFunctionAB;
	 //delete vLearnerPolicyAB;
	 //delete vFunctionLearnerAB;
	 delete agentAB;
	 delete environmentModelAB;
	 delete classifier;
	 */
}

