#ifndef CTREES__H
#define CTREES__H

#include "cinputdata.h"

#include <newmat/newmat.h>
#include <list>
#include <vector>
#include <stdio.h>




class CSplittingCondition
{
	public:
		virtual ~CSplittingCondition() {};
		virtual bool isLeftNode(ColumnVector *input) = 0;
};

class C1DSplittingCondition : public CSplittingCondition
{
	protected:
		int dimension;
		double treshold;
	public:
		C1DSplittingCondition(int dimension, double treshold);
		virtual ~C1DSplittingCondition();
		
		virtual bool isLeftNode(ColumnVector *input);

		int getDimension();
		double getTreshold();
};

class CSplittingConditionFactory 
{
	protected:

	public:
		virtual ~CSplittingConditionFactory() {};
		virtual CSplittingCondition *createSplittingCondition(DataSubset *dataSubset) = 0;

		virtual bool isLeaf(DataSubset *dataSubset) = 0;
};



template<typename TreeData> class CTreeDataFactory
{
	public:
		virtual ~CTreeDataFactory() {};
		
		virtual TreeData createTreeData(DataSubset *dataSubset, int numLeaves) = 0;
		virtual void deleteData(TreeData ){}; 
};
		

template<typename TreeData> class CLeaf;

template<typename TreeData> class CTreeElement
{
	protected:
		CTreeElement *parent;

	public:
		CTreeElement(CTreeElement<TreeData> *l_parent)
		{
			parent = l_parent;
		};
		virtual ~CTreeElement() {};

		virtual CLeaf<TreeData> *getLeaf(ColumnVector *input) = 0;
		
		CTreeElement<TreeData> *getParent() {return parent;};

		virtual int getDepth() {return 0;};

		virtual bool isLeaf() {return false;};
};

template<typename TreeData> class CNode : public CTreeElement<TreeData>
{
	protected:
		CSplittingCondition *split;
		CTreeElement<TreeData> *leftElement;
		CTreeElement<TreeData> *rightElement;
	public:
		CNode(CTreeElement<TreeData> *parent, CSplittingCondition *l_condition, CTreeElement<TreeData> *l_leftElement, CTreeElement<TreeData> *l_rightElement);
		virtual ~CNode();
	
		virtual CLeaf<TreeData> *getLeaf(ColumnVector *input);
		virtual int getDepth();

		CTreeElement<TreeData> *getLeftElement() {return leftElement;};
		CTreeElement<TreeData> *getRightElement() {return rightElement;};

		void setLeftElement(CTreeElement<TreeData> *l_leftElement) {leftElement = l_leftElement;};
		void setRightElement(CTreeElement<TreeData> *l_rightElement) {rightElement = l_rightElement;};


		CSplittingCondition *getSplittingCondition();
};

template<typename TreeData> CNode<TreeData>::CNode(CTreeElement<TreeData> *parent, CSplittingCondition *l_condition, CTreeElement<TreeData> *l_leftElement, CTreeElement<TreeData> *l_rightElement) : CTreeElement<TreeData>(parent)
{
	split = l_condition;
	leftElement = l_leftElement;
	rightElement = l_rightElement;
}
	
template<typename TreeData> CNode<TreeData>::~CNode()
{
	delete split;
	delete rightElement;
	delete leftElement;
}

template<typename TreeData> CSplittingCondition *CNode<TreeData>::getSplittingCondition()
{
	return split;
}
	
template<typename TreeData> CLeaf<TreeData> *CNode<TreeData>::getLeaf(ColumnVector *input)
{
	bool isLeft = split->isLeftNode(input);
	if (isLeft)
	{
		return leftElement->getLeaf(input);
	}
	else
	{
		return rightElement->getLeaf(input);
	}
}

template<typename TreeData> int CNode<TreeData>::getDepth()
{
	return std::max(leftElement->getDepth(), rightElement->getDepth()) + 1;
}

template <typename TreeData> class CLeaf : public CTreeElement<TreeData>
{
	protected:
		TreeData data;
		CTreeDataFactory<TreeData> *dataFactory;
		
		int numLeaf;
		DataSubset *subset;
	public:
		CLeaf(CTreeElement<TreeData> *parent, TreeData l_data, DataSubset *subset, int numLeaf, CTreeDataFactory<TreeData> *l_dataFactory);

		virtual ~CLeaf();
				
		virtual CLeaf<TreeData> *getLeaf(ColumnVector *);
		virtual TreeData getTreeData();

		virtual bool isLeaf() {return true;};

		int getLeafNumber() {return numLeaf;};
		int getNumSamples() {return subset->size();};

		DataSubset *getDataSet() {return subset;};
};

template <typename TreeData> CLeaf<TreeData>::CLeaf(CTreeElement<TreeData> *parent, TreeData l_data, DataSubset *l_subset, int l_numLeaf, CTreeDataFactory<TreeData> *l_dataFactory) : CTreeElement<TreeData>(parent)
{
	data = l_data;
	dataFactory = l_dataFactory;
	numLeaf = l_numLeaf;
	subset = new DataSubset(*l_subset);
}
		
template <typename TreeData> CLeaf<TreeData>::~CLeaf()
{
	dataFactory->deleteData(data);
	delete subset;
}
				
template <typename TreeData> TreeData CLeaf<TreeData>::getTreeData()
{
	//printf("Returning Data from Leave %d\n", numLeaf);
	return data;
}

template <typename TreeData> CLeaf<TreeData> *CLeaf<TreeData>::getLeaf(ColumnVector *)
{
	//printf("Returning Data from Leave %d\n", numLeaf);
	return this;
}


template<typename TreeData> class CTree : public CMapping<TreeData>
{
	protected:
		CTreeElement<TreeData> *root;
		
		CTreeDataFactory<TreeData> *dataFactory;
		
		int numLeaves;
		
		virtual void createTree(CDataSet *inputData, CSplittingConditionFactory *splittingFactory, CTreeDataFactory<TreeData> *l_dataFactory, bool createLeaves = true);
		
		
		virtual CTreeElement<TreeData> *createNode(CTreeElement<TreeData> *parent, CDataSet *inputData, DataSubset *inputDataSubset, CSplittingConditionFactory *splittingFactory, CTreeDataFactory<TreeData> *dataFactory);
	
		CTree(int numDim);
		CLeaf<TreeData> **leaves;

		virtual int setLeaves(CTreeElement<TreeData> *element, int numLeaf);

		CDataSet *inputData;

		virtual TreeData doGetOutputValue(ColumnVector *input);
	public:	
		
		
		CTree(CDataSet *inputData, CSplittingConditionFactory *splittingFactory, CTreeDataFactory<TreeData> *l_dataFactory);
		
		
		virtual ~CTree();
		
		CLeaf<TreeData> *getLeaf(int index);		

		
		int getNumLeaves();
		int getDepth();
		int getNumSamples();

		CTreeElement<TreeData> *getRoot() {return root;};

		CDataSet *getInputData() {return inputData;};

		virtual CLeaf<TreeData> *getLeaf(ColumnVector *input);	
		CTreeDataFactory<TreeData> *getDataFactory() {return dataFactory;};	
	
		virtual void addNewInput(int index, CSplittingConditionFactory *splitting);

		void createLeavesArray();
};


template<typename TreeData> CTree<TreeData>::CTree(int numDim) : CMapping<TreeData>(numDim)
{
	root = NULL;
	dataFactory = NULL;

	leaves = NULL;
}


template<typename TreeData> CTree<TreeData>::~CTree()
{
	if (root != NULL)
	{
		delete root;
	}
	if (leaves != NULL)
	{
 		delete [] leaves;
	}
}

template<typename TreeData> CLeaf<TreeData> * CTree<TreeData>::getLeaf(int index)
{
	return leaves[index];
}

template<typename TreeData> void CTree<TreeData>::addNewInput(int index, CSplittingConditionFactory *splittingFactory)
{
	ColumnVector *newInput = (*inputData)[index];
	CLeaf<TreeData> *leaf = getLeaf(newInput);

	if (leaf != NULL && leaf->getParent() != NULL)
	{
		CNode<TreeData> *parent = dynamic_cast<CNode<TreeData> *>(leaf->getParent());
		
		bool isLeft = parent->getLeftElement() == leaf;
		DataSubset *subset = leaf->getDataSet();
		subset->insert(index);		

		CTreeElement<TreeData> *newNode = createNode(parent, getInputData(), subset, splittingFactory, getDataFactory()); 
		
		if (isLeft)
		{
			parent->setLeftElement(newNode);
		}	
		else
		{
			parent->setRightElement(newNode);
		}
		delete leaf;
		numLeaves --;  

		
	}
	else
	{
		if (root != NULL)
		{
			delete root;
		}

		createTree(getInputData(), splittingFactory, getDataFactory(), false);
	}
	//printf("DataSetSize %d, Leaves: %d\n", inputData->size(), numLeaves);
}

template<typename TreeData> int CTree<TreeData>::getNumLeaves()
{
	return numLeaves;
}

template<typename TreeData> int CTree<TreeData>::getDepth()
{
	return root->getDepth();
}
		
template<typename TreeData> CTree<TreeData>::CTree(CDataSet *inputData, CSplittingConditionFactory *splittingFactory, CTreeDataFactory<TreeData> *l_dataFactory) : CMapping<TreeData>(inputData->getNumDimensions())
{
	createTree(inputData, splittingFactory, l_dataFactory);
}
		
		
template<typename TreeData>  TreeData CTree<TreeData>::doGetOutputValue(ColumnVector *input)
{
	return root->getLeaf(input)->getTreeData();
}

template<typename TreeData>  CLeaf<TreeData> * CTree<TreeData>::getLeaf(ColumnVector *input)
{
	if (root)
	{
		ColumnVector *l_input = CMapping<TreeData>::getPreprocessedInput(input);
	
		return root->getLeaf(l_input);
	}
	else
	{
		return NULL;
	}
}


template<typename TreeData> void CTree<TreeData>::createTree(CDataSet *l_inputData, CSplittingConditionFactory *splittingFactory, CTreeDataFactory<TreeData> *l_dataFactory, bool setLeaves)
{
	inputData = l_inputData;
	dataFactory = l_dataFactory;

	DataSubset subset;
	numLeaves = 0;
	
	if (l_inputData->size() > 0)
	{ 
		for (unsigned int i = 0; i < l_inputData->size(); i ++)
		{
			subset.insert(i);
		}
		dataFactory = l_dataFactory;
		root = createNode(NULL, l_inputData, &subset, splittingFactory, l_dataFactory);
		
		if (setLeaves)
		{
			createLeavesArray();
		}
	}
}

template<typename TreeData> void CTree<TreeData>::createLeavesArray()
{
	if (leaves)
	{
		delete leaves;
	}

	leaves = new CLeaf<TreeData>*[numLeaves];
	setLeaves(root, 0);	
}

template<typename TreeData> int CTree<TreeData>::getNumSamples()
{
	int samples = 0;
	for (int i = 0; i < numLeaves; i ++)
	{
		samples += leaves[i]->getDataSet()->size();
	}
	return samples;
}



template<typename TreeData> int CTree<TreeData>::setLeaves(CTreeElement<TreeData> *element, int numLeaf)
{
	if (element->isLeaf())
	{
		CLeaf<TreeData> *leaf = dynamic_cast<CLeaf<TreeData> *>(element);
		leaves[numLeaf] = leaf;
		return 1;
	}
	else
	{
		int newLeaves = 0;
		CNode<TreeData> *node = dynamic_cast<CNode<TreeData> *>(element);
		
		newLeaves += setLeaves(node->getLeftElement(), numLeaf);
		
		newLeaves += setLeaves(node->getRightElement(), numLeaf + newLeaves);
		
		return newLeaves;
	}
	
}

template<typename TreeData>  CTreeElement<TreeData> *CTree<TreeData>::createNode(CTreeElement<TreeData> *parent,CDataSet *inputData, DataSubset *inputDataSubset, CSplittingConditionFactory *splittingFactory, CTreeDataFactory<TreeData> *dataFactory)
{
	//printf("Creating Node with %d Inputs\n", inputDataSubset->size());
	
	bool isLeaf = splittingFactory->isLeaf(inputDataSubset);
	CTreeElement<TreeData> *newNode = NULL;
			
	if (isLeaf)
	{
		TreeData data = dataFactory->createTreeData(inputDataSubset, numLeaves);
		newNode = new CLeaf<TreeData>(parent, data,inputDataSubset, numLeaves, dataFactory);
		numLeaves ++;
	}
	else
	{
		CSplittingCondition *split = splittingFactory->createSplittingCondition(inputDataSubset);
				
		DataSubset leftSet;
		DataSubset rightSet;
				
		DataSubset::iterator it = inputDataSubset->begin();
				
		for (; it != inputDataSubset->end(); it++)
		{
			ColumnVector *input = (*inputData)[*it];
					
			if (split->isLeftNode(input))
			{
				leftSet.insert(*it);
			}
			else
			{
				rightSet.insert(*it);
			}
		}
		//printf("LeftSet : %d, Right Set : %d\n", leftSet.size(), rightSet.size());
		
		CNode<TreeData> *newNode1 = new CNode<TreeData>(parent, split, NULL, NULL);

		CTreeElement<TreeData> *leftElement = createNode(newNode1, inputData, &leftSet, splittingFactory, dataFactory);
				
		CTreeElement<TreeData> *rightElement = createNode(newNode1, inputData, &rightSet, splittingFactory, dataFactory);
				
		newNode1->setLeftElement(leftElement);
		newNode1->setRightElement(rightElement);

		newNode = newNode1;
		
	}
	return newNode;
}


#endif
 
