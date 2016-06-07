package linear.perceptron;

import java.util.Arrays;
import weka.core.Instance;
import weka.core.Instances;

/** 
 * Enhanced Linear Perceptron Classifier child of LinearPerceptron
 * @author James Rogers
 * @version 1.0 2015.
 */
public class EnhancedLinearPerceptron extends LinearPerceptron {
    
    /**
     * Standardise the data.
     */
    private boolean standardiseFlag;
    /**
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
    /**
     * The mean of the data in Instances.
     */
    private double[] mean;
    /**
     * The standard Deviation of the data in Instances.
     */
    private double[] standardDeviation;
    
    /** 
    * Initialises. 
    * Calls parent constructor.
    * Default standardise to true.
    * Default update method to offline.
    * Default to use crossValidation.
    * Default to 4 folds.
    * Mean default to null.
    * standardDeviation default to null.
    */
    EnhancedLinearPerceptron() {
        super();
        
        standardiseFlag = true;
        updateMethodFlag = true;
        crossValidationFlag = true;
        
        //Folds default to 4
        folds = 4;
        mean = null;
        standardDeviation = null;
    }
    
    /** 
    * Builds the classifier
    * @param ins    Instances
    * @throws       Exception
    */
    @Override
    public void buildClassifier(Instances ins) throws Exception {
        
        //Check if values are continuous
        if(getCheckAttributesFlag()) {
            checkAttributes(ins);
        }
        
        //Standardise all Attributes
        if(standardiseFlag) {
            ins = standardiseInstancesAttributes(ins);
        }
        
        //Iniatlise Attribute Indexes to default if not specified
        if(getAttributeIterator() == null)
            initAttributeIndexs(ins);
        
        //Initalise the weights
        setWeight(initaliseWeights(getAttributeIterator().length));
        
        //Choose best update method cross validation
        if(crossValidationFlag) {
            crossValidation(ins);
        }
        
        //Classify Instances using an update method
        if(updateMethodFlag) {
            setWeight(offlineClassification(ins, getWeight()));
        }
        else {
            setWeight(onlineClassification(ins, getWeight()));
        }
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
     * Cross validates the Instances using a given number of folds. 
     * Calculates the error rate of both Offline and Online and sets the 
     * update method depending on the best performing.
     * @param ins   The Instances to be cross validated.
     */
    protected void crossValidation(Instances ins) {
        
        //Weight to be used
        double[] onlineWeight;
        double[] offlineWeight;
        double[] tempWeight;
        
        //The classification result
        double result;
        
        //The final average Error Rate
        float averageOnlineErrorRate = 0;
        float averageOfflineErrorRate = 0;
        
        int onlineErrorRate;
        int offlineErrorRate;
        
        //Iterate over number of folds
        for (int n = 0; n < folds; n++) {
            
            //Create training and test folds
            Instances train = ins.trainCV(folds, n);
            Instances test = ins.testCV(folds, n);
            
            //Initalise temp weight to random numbers 
            //Initalise the weights
            if(getRandomiseWeights())
                tempWeight = EnhancedLinearPerceptron.initWeightRandom(getAttributeIterator().length);
            else
                tempWeight = EnhancedLinearPerceptron.initWeightNumber(getAttributeIterator().length, 
                    getWeightFilled());
            
            //Classify using Online update
            onlineWeight = onlineClassification(train, tempWeight.clone());
            //Classify using Offline update
            offlineWeight = offlineClassification(train, tempWeight.clone());
            
            //Online and Offline error rate set to 0
            onlineErrorRate = 0;
            offlineErrorRate = 0;
            
            //Classify test instance using Online and Offline
            for(Instance in : test) {
                
                //Classify instance using Online
                result = classify(in, onlineWeight);
                //Increment Online error rate if incorrect
                if(result != in.classValue())
                    onlineErrorRate++;
                
                //Classify instance using Offline
                result = classify(in, offlineWeight);
                //Increment Offline error rate if incorrect
                if(result != in.classValue())
                    offlineErrorRate++;
            }
            
            //Update average error rate percentages
            averageOnlineErrorRate += (onlineErrorRate * 100) / test.numInstances();
            averageOfflineErrorRate += (offlineErrorRate * 100) / test.numInstances();
        }
        
        //Calculate average error rate percentages
        averageOnlineErrorRate /= folds;
        averageOfflineErrorRate /= folds;
        
        //System.out.println("Online update error rate: " + averageOnlineErrorRate + "%");
        //System.out.println("Offline update error rate: " + averageOfflineErrorRate + "%");
        
        //Decide best update method to use
        updateMethodFlag = averageOnlineErrorRate > averageOfflineErrorRate;
        
        /*if(updateMethodFlag)
            System.out.println("Offline update is the most optimal");
        else
            System.out.println("Online update is the most optimal");*/
    }
    
    /** 
    * Classifies the Instances by generating a new weight using the 
    * Offline Update method.
    * The Offline Update method updates a delta weight that is calculated 
    * and summed after each classification. The delta weight is then applied
    * to the final weight after each full iteration of the Instances. The loop 
    * will not stop altering the weight until a full loop has passed with 
    * correct classifications, or until the maximum number of loops have 
    * passed defined by the depth attribute.
    * @param ins    The Instances object that will used for training.
    * @param weight The weight that will be updated
    * @return       double[] The new weight 
    */
    protected double[] offlineClassification(Instances ins, double[] weight) {
        double[] weightDelta;
        double[] tempWeight;
        
        double result;
        int depthCounter = 0;
        
        do {
            //Initalise weightDelta to size of number of attributes with
            //values of 0
            weightDelta = initWeightNumber(getAttributeIterator().length, 0);
            tempWeight = weight.clone();
            
            for(Instance in : ins) {
                
                //Classify instance
                result = getBias(); //?
                for(int i = 0; i < getAttributeIterator().length; i++)
                    result += weight[i] * in.value(getAttributeIterator()[i]);
                
                //Update weightDelta based on previous classification
                int c = classification(in.classValue()) - classification(result);
                for(int i = 0; i < getAttributeIterator().length; i++)
                    weightDelta[i] += 0.5 * getLearningRate() * c 
                            * in.value(getAttributeIterator()[i]);
            }
            
            //Update the weights using the delta weight
            for(int i = 0; i < getAttributeIterator().length; i++)
                weight[i] += weightDelta[i];
            
            //Increment the depth counter after each full loop
            depthCounter++;
            
        //Repeat until weight has not altered or maximm depth has been reached
        } while(!Arrays.equals(tempWeight, weight) && depthCounter < getDepth());
        
        //Return the weight
        return weight;
    }
    
    /** 
    * Classify the instance
    * @param in     Instance that is to be classified
    * @return       The classification result
    * @throws       Exception
    */
    @Override
    public double classifyInstance(Instance in) throws Exception {
        
        //Standardise attributes
        if(standardiseFlag)
            in = standardiseInstanceAttributes(in, mean, standardDeviation);
                
        //Return the classification result
        return classify(in, getWeight());
    }
    
    /** 
    * Standardise the Attributes in an Instance Object using the given
    * mean and standardDeviation created in 
    * standardiseInstancesAttributes(Instances) method
    * @param in    The Instance that is to be standardised
     * @param mean
     * @param standardDeviation
    * @return       The new standardised Instance
    */
    public static Instance standardiseInstanceAttributes(Instance in, double[] mean, double[] standardDeviation) {
        
        double standardised;
    
        for(int i = 0; i < in.numAttributes()-1; i++) {
            
            //Calculate the standardised data and insert it in to instances
            standardised = (in.value(i) - mean[i]);
            standardised /= standardDeviation[i];
            in.setValue(i, standardised);
        }
        return in;
    }
    
    public boolean getStandardiseFlag() {
        return standardiseFlag;
    }
    
    public boolean getUpdateMethodFlag() {
        return updateMethodFlag;
    }
    
    public boolean getCrossValidationFlag() {
        return crossValidationFlag;
    }
    
    public int getFolds() {
        return folds;
    }
    
    public double[] getMean() {
        return mean;
    }
    
    public double[] getstandardDeviation() {
        return standardDeviation;
    }
    
    /** 
    * Set the classifier to standardise the data 
    * @param b  True to standardise. False to not.
    */
    public void setStandardiseFlag(boolean b) {
        standardiseFlag = b;
    }
    
    /** 
    * Sets the update method. Online of Offline.
    * @param b  True for Offline, False for Online.
    */
    public void setUpdateMethodFlag(boolean b) {
        updateMethodFlag = b;
    }
    
    /** 
    * Sets classifier to use Cross Validation.
    * @param b  True to use cross validation, False or not.
    */
    public void setCrossValidationFlag(boolean b) {
        crossValidationFlag = b;
    }
    
    /** 
    * Sets the number of folds.
    * @param f  The number of folds
    * @throws Exception if folds is smaller than 2.
    */
    public void setFolds(int f) throws Exception {
        
        if(f < 2)
            throw new Exception("Exception: The number of folds must be >= 2");
        
        folds = f;
    }
}