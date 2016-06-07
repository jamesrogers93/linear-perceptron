package linear.perceptron;

import java.util.Arrays;
import weka.classifiers.Classifier;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;
import java.util.Random;

/** 
 * Random Linear Perceptron Classifier
 * @author James Rogers
 * @version 1.0 2015.
 */
public class RandomLinearPerceptron implements Classifier  {
    
    /****************************************************************
    *********Random Linear Perceptron Classifier Variables***********
    ****************************************************************/
    /**
     * The array of Linear Perceptrons (Ensembles).
     */
    private LinearPerceptron[] perceptrons;
    /**
     * A flag to select which Linear Perceptron method to use.
     */
    private boolean perceptronClassifier;
    /**
     * The number of classifiers in ensemble.
     */
    private int numEnsembles;
    /**
     * An attribute iterator for each ensemble.
     */
    private int[][] attributeIterator;
    /**
     * The number of Attributes to be classified.
     */
    private int numAttributes; 
    /**
     * Standardise the data Flag.
     */
    private boolean standardiseFlag;
    /**
     * The Mean to standardise the data .
     */
    private double[] mean;
    /**
     * The Standard Deviation to standardise the data .
     */
    private double[] standardDeviation;
    
    /****************************************************************
    ************Linear Perceptron Classifier Variables***************
    ****************************************************************/
    /**
     * A flag to randomise the weights.
     */
    private boolean randomiseWeights;
    /**
     * Specifies the number that the weights will be filled with.
     */
    private int weightFilled;
    /**
     * The learning rate that alters the weight.
     */
    private int learningRate;
    /**
     * The limit of cycles when classifying the instances.
     */
    private int depth;

    /****************************************************************
     ********Enhanced Linear Perceptron Classifier Variables**********
     ****************************************************************/
    /*
     * Specify update method.
     */
    private boolean updateMethodFlag;
    /**
     * Specify whether or not to use cross validation.
     */
    private boolean crossValidationFlag;
    /**
     * The number of folds for the cross validation.
     */
    private int folds;
    
    
    RandomLinearPerceptron() {
        
        //Random Linear Perceptron Variables
        perceptrons = null;
        perceptronClassifier = true; //Default to Linear Perceptrons
        numEnsembles = 500;          //Default to 500
        attributeIterator = null;
        numAttributes = 0;
        standardiseFlag = true;
        mean = null;
        standardDeviation = null;
        
        //Linear Perceptron variables
        randomiseWeights = true;
        weightFilled = 1;
        learningRate = 1;
        depth = 10;
        
        //Enhanced Linear Perceptron variables
        updateMethodFlag = true;
        crossValidationFlag = true;
        folds = 4;
    }

    /**
     * Builds the classifer based on the methods set
     * @param ins    The instances t build the classifier on
     * @throws Exception 
     */
    @Override
    public void buildClassifier(Instances ins) throws Exception {
        
        //Check data using code from the Linear Perceptron
        linear.perceptron.LinearPerceptron.checkAttributes(ins);
        
        //Standardise Data
        if(standardiseFlag) {
            standardiseInstancesAttributes(ins);  
        }
        
        //Choose which perceptron to use
        if(perceptronClassifier) {
            initLinearPerceptrons();
        }
        else {
            initEnhancedLinearPerceptrons();
        }
            
        //If attributes have not been set, default to the square root of the 
        //num of attributes
        if(numAttributes == 0)
            numAttributes = (int)(Math.sqrt(ins.numAttributes()-1) + 0.5);
        //If attributes have been set and greater than the attribs in ins
        else if(numAttributes > ins.numAttributes()-1) {
            throw new Exception("numattributes cannot be greater than the"
                    + "number of attributes in the instances object");
        }
            
        //Generate the attribute iterator using the number of attributes in 
        //the instances object
        generateAttributeIterator(ins.numAttributes()-1);
        
        //Build the perceptron classiffiers 
        buildPerceptronClassifiers(ins);
        
    }
    
    /** 
    * Standardise the Attributes in an Instances Object
    * @param ins    The Instances that are to be standardised
    * @return       The new standardised Instances
    */
    protected Instances standardiseInstancesAttributes(Instances ins) {
        
        double sum;
        double standardised;
        
        //Initalise mean and standardDevation to the number of attributes in ins
        mean = new double[ins.numAttributes()-1];
        standardDeviation = new double[ins.numAttributes()-1];
        
        for(int i = 0; i < ins.numAttributes()-1; i++) {
            
            //Find the mean of each attribute
            sum = 0;
            for(int j = 0; j < ins.numInstances(); j++) {
                sum += ins.get(j).value(i);
            }
            mean[i] = sum / ins.numInstances();
            
            //Find the standard deviation
            sum = 0;
            for(int j = 0; j < ins.numInstances(); j++) {
                sum += (ins.get(j).value(i) - mean[i]) 
                        * (ins.get(j).value(i) - mean[i]);
            }
            standardDeviation[i] = sum / ins.numInstances();
            
            //Calculate the standardised data and insert it in to instances
            for(int j = 0; j < ins.numInstances(); j++) {
                standardised = (ins.get(j).value(i) - mean[i]);
                standardised /= standardDeviation[i];
                ins.get(j).setValue(i, standardised);
            } 
        }
        //Return the standardised Instances
        return ins;
    }
    
    /**
     * Initalises the Ensembles to linear perceptrons and sets all of their
     * varaibles.
     */
    public void initLinearPerceptrons() {
        perceptrons = new LinearPerceptron[numEnsembles];
        LinearPerceptron p;
        for(int i = 0; i < numEnsembles; i++) {
            p = new LinearPerceptron();
            p.setCheckAttributesFlag(false);
            p.setRandomiseWeightsFlag(this.randomiseWeights);
            p.setWeightFilled(this.weightFilled);
            p.setLearningRate(this.learningRate);
            p.setDepth(this.depth);
            perceptrons[i] = p;
        }
    }
    
    /**
     * Initalises the Ensembles to Enhanced linear perceptrons and sets all of their
     * varaibles.
     */
    public void initEnhancedLinearPerceptrons() throws Exception {
        perceptrons = new EnhancedLinearPerceptron[numEnsembles];
        EnhancedLinearPerceptron p;
        for(int i = 0; i < numEnsembles; i++) {
            p = new EnhancedLinearPerceptron();
            p.setCheckAttributesFlag(false);
            p.setRandomiseWeightsFlag(this.randomiseWeights);
            p.setWeightFilled(this.weightFilled);
            p.setLearningRate(this.learningRate);
            p.setDepth(this.depth);
            p.setStandardiseFlag(false);
            p.setCrossValidationFlag(this.crossValidationFlag);
            p.setFolds(this.folds);
            p.setUpdateMethodFlag(this.updateMethodFlag);
            perceptrons[i] = p;
        }
    }
    
   /**
    * Generates the attribute iterator to random indexes.
    * @param size The number of attributes in the iterator
    */
    private void generateAttributeIterator(int size) {
        
        //Init instancesIndex array
        attributeIterator = new int[numEnsembles][numAttributes];
        
        //Randomize instanceIndex attributes
        Random rand = new Random();
        
        for (int i = 0; i < attributeIterator.length; i++) {
            
            for(int j = 0; j < attributeIterator[i].length; j++) {
                
               attributeIterator[i][j] = rand.nextInt(size);
                for (int k = 0; k < j; k++) {
                    if (attributeIterator[i][j] == attributeIterator[i][k]) {
                        j--;
                        break;
                    }
                }  
            }
            Arrays.sort(attributeIterator[i]);
        }
    }
    
    /**
     * Builds all of the classifiers in the ensembles
     * @param ins
     * @throws Exception 
     */
    private void buildPerceptronClassifiers(Instances ins) throws Exception {
        for(int i = 0; i < this.numEnsembles; i++) {
            perceptrons[i].setAttributeIterator(attributeIterator[i]);
            perceptrons[i].buildClassifier(ins);
        }
    } 

    /**
     * Classfies the instances using the vote generated by the ensembles
     * @param in
     * @return
     * @throws Exception 
     */
    @Override
    public double classifyInstance(Instance in) throws Exception {
        
        double[] vote = distributionForInstance(in);
        
        if(vote[0] > vote[1])
            return 0.0;
        return 1.0;
    }
    
    /**
     * Returns the majority vote from each ensemble
     * @param in
     * @return
     * @throws Exception 
     */
    @Override
    public double[] distributionForInstance(Instance in) throws Exception {
        double[] vote = new double[2];
        for(LinearPerceptron p : perceptrons)
            vote[(int)p.classifyInstance(in)]++;
        
        return vote;
    }
    
    /** Prints the weight of the classifier */
    public void printWeight() {
        System.out.println("\nWeight: ");
        for(int i = 0; i < this.numEnsembles; i++)
            this.perceptrons[i].printWeight();
    }
    
    /****************************************************************
    *********Random Linear Perceptron Classifier Variables***********
    ****************************************************************/    
    public void setNumEnsembles(int e) throws Exception {
        if ( e <= 1)
            throw new Exception("The number of Ensembles must be greater than 1");
        numEnsembles = e;
    }
    
    public void setAttributeIterator(int[][] i) {
        attributeIterator = i;
    }
    
    public void setNumAttributes(int a) throws Exception {
        if ( a <= 1)
            throw new Exception("The numer of Attributes must be greater than 1");
        
        numAttributes = a;
    }
    
    public void setPerceptronClassifier(boolean perceptronClassifier) {
        this.perceptronClassifier = perceptronClassifier;
    }

    public void setStandardiseFlag(boolean standardiseFlag) {
        this.standardiseFlag = standardiseFlag;
    }
    
    
    /****************************************************************
    ************Linear Perceptron Classifier Variables***************
    ****************************************************************/

    public void setRandomiseWeights(boolean randomiseWeights) {
        this.randomiseWeights = randomiseWeights;
    }

    public void setWeightFilled(int weightFilled) {
        this.weightFilled = weightFilled;
    }

    public void setLearningRate(int learningRate) {
        this.learningRate = learningRate;
    }

    public void setDepth(int depth) {
        this.depth = depth;
    }
    
    /****************************************************************
     ********Enhanced Linear Perceptron Classifier Variables**********
     ****************************************************************/
    
    public void setUpdateMethodFlag(boolean updateMethodFlag) {
        this.updateMethodFlag = updateMethodFlag;
    }

    public void setCrossValidationFlag(boolean crossValidationFlag) {
        this.crossValidationFlag = crossValidationFlag;
    }

    public void setFolds(int folds) {
        this.folds = folds;
    }

    @Override
    public Capabilities getCapabilities() {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }
}
