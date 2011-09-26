//
// C++ Interface: cforest
//
// Description: 
//
//
// Author: Neumann Gerhard <gerhard@tu-graz.ac.at>, (C) 2006
//
// Copyright: See COPYING file that comes with this distribution
//
//

#ifndef C_FOREST__H
#define C_FOREST__H

#include "ctrees.h"
#include "cinputdata.h"

///*****************  CForest ***********************

template <typename TreeData> class CForest 
{
	protected:
		
		CTree<TreeData> **forest;
		int numTrees;	
	public:
		CForest(int numTrees);
		virtual ~CForest();
		
		virtual void getTreeDatas(ColumnVector *vector, TreeData *outputs);
		
		virtual void addTree(int index, CTree<TreeData> *tree);
		virtual void removeTree(int index);

		 CTree<TreeData> *getTree(int index);		

		int getNumTrees();

		double getAverageDepth();
		double getAverageNumLeaves();

		virtual void getActiveLeafNumbers(ColumnVector *vector, int *leafNumbers);
		virtual void getActiveLeaves(ColumnVector *vector, CLeaf<TreeData> **leafNumbers);

		int getNumLeaves();
};

template<typename TreeData> CForest<TreeData>::CForest(int l_numTrees)
{
	numTrees = l_numTrees;
	forest = new CTree<TreeData> *[numTrees];
	
	for (int i = 0; i < numTrees; i++)
	{
		forest[i] = NULL;
	}
}

template<typename TreeData> CForest<TreeData>::~CForest()
{
	delete [] forest;
}
		
template<typename TreeData> void  CForest<TreeData>::getActiveLeafNumbers(ColumnVector *vector, int *leafNumbers)
{
	int leaveSum = 0;

	for (int i = 0; i < numTrees; i ++)
	{
		leafNumbers[i] = forest[i]->getLeaf(vector)->getLeafNumber() + leaveSum;
		leaveSum += forest[i]->getNumLeaves();
	}
}


template<typename TreeData> void  CForest<TreeData>::getActiveLeaves(ColumnVector *vector, CLeaf<TreeData> **leaves)
{
	for (int i = 0; i < numTrees; i ++)
	{
		leaves[i] = forest[i]->getLeaf(vector);
	}
}

template<typename TreeData> int CForest<TreeData>::getNumLeaves()
{
	int numLeaves = 0;	
	for (int i = 0; i < numTrees; i ++)
	{
		numLeaves += forest[i]->getNumLeaves();
	}
	return numLeaves;
}


template<typename TreeData> double CForest<TreeData>::getAverageDepth()
{
	double depth = 0;
	for (int i = 0; i < numTrees; i ++)
	{
		depth += forest[i]->getDepth();
	}
	return depth / numTrees;
}

template<typename TreeData> double CForest<TreeData>::getAverageNumLeaves()
{

	double leaves = 0;
	for (int i = 0; i < numTrees; i ++)
	{
		leaves += forest[i]->getNumLeaves();
	}
	return leaves / numTrees;
}



template<typename TreeData> CTree<TreeData> *CForest<TreeData>::getTree(int index)
{
	return forest[index];
}

template<typename TreeData>  void CForest<TreeData>::getTreeDatas(ColumnVector *vector, TreeData *outputs)
{
	for (int i = 0; i < numTrees; i++)
	{
		if (forest[i] != NULL)
		{
			TreeData element = forest[i]->getOutputValue(vector);
			outputs[i] = element; 
		}
	}
}
		
template<typename TreeData> void CForest<TreeData>::addTree(int index, CTree<TreeData> *tree)
{
	forest[index] = tree;
}


template<typename TreeData> void CForest<TreeData>::removeTree(int index)
{
	forest[index] = NULL;;
}

template<typename TreeData> int CForest<TreeData>::getNumTrees()
{
	return numTrees;
}

///*****************  CRegressionForest ***********************

class CRegressionForest : public CForest<double>, public CMapping<double>
{
	protected:
		virtual double doGetOutputValue(ColumnVector *vector);
	public:
		CRegressionForest(int numTrees, int numDim);
		virtual ~CRegressionForest();
		
		

		virtual void saveASCII(FILE *stream);
};

class CExtraTreeRegressionForest : public CRegressionForest
{
	protected:

	public:
		CExtraTreeRegressionForest(int numTrees, CDataSet *inputData, CDataSet1D *outputData, unsigned int K,unsigned  int n_min, double treshold, CDataSet1D *weightData = NULL);
		virtual ~CExtraTreeRegressionForest();

};

class CRegressionMultiMapping : public CMapping<double>
{
protected:
	CMapping<double> **mappings;
	int numMappings;



	double doGetOutputValue(ColumnVector *inputVector);
public:
	bool deleteMappings;

	CRegressionMultiMapping(int numMappings, int numDimensions);
	virtual ~CRegressionMultiMapping();

	void addMapping(int index, CMapping<double> *mapping);
};

#endif
