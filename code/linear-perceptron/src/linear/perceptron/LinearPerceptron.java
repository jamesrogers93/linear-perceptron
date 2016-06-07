package linear.perceptron;

//Java
import java.util.Arrays;

//Weka
import weka.classifiers.Classifier;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;

/** 
 * Linear Perceptron Classifier
 * @author James Rogers
 * @version 1.0 2015.
 */
public class LinearPerceptron implements Classifier {

    /**
     * The weight that is used to classify an Instance.
     */
    private double[] weight;
    /**
     * A flag to randomise the weights
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
    /**
     * The iterator that for the attributes, defaults to iterate through all
     * the attributes
     */
    private int[] attributeIterator;
    /**
     * Specify whether or not to use cross validation.
     */
    private boolean checkAttributesFlag;
    /**
     * The bias t0 apply to the weight updates.
     */
    private int bias;
    
    /** 
    * Initialises. 
    * weight default to null.
    * randomiseWeights default to true.
    * weightFilled default to 1.
    * learningRate default to 1
    * depth default to 10.
    * attributeIterator to null.
    * checkAttributesFlags to true
    * bias to 0
    */
    LinearPerceptron() {
        
        weight = null;
        randomiseWeights = true;
        weightFilled = 1;
        learningRate = 1;
        depth = 10;
        attributeIterator = null;
        checkAttributesFlag = true;
        bias = 0;
    }
    
    /** 
    * Builds the classifier using the Online update method.
    * @param ins    Instances
    * @throws       Exception
    */
    @Override
    public void buildClassifier(Instances ins) throws Exception {
         
        //Check data for continuos values
        if(checkAttributesFlag)
            checkAttributes(ins);
        
        if(attributeIterator == null)
            initAttributeIndexs(ins);
        
        weight = initaliseWeights(attributeIterator.length);
        
        //Classify Instances
        weight = this.onlineClassification(ins, weight);
    }
    
    /** 
    * Checks that the data attributes are continuous
    * @param ins    The Instances
    * @throws linear.perceptron.WEKAValueException
    */
    protected static void checkAttributes(Instances ins) throws WEKAValueException{
        
        for(int i = 0; i < ins.numAttributes()-1; i++) {
            
            if(i != ins.classIndex()) {
                
                if(!ins.attribute(i).isNumeric()) {
                    throw new WEKAValueException("Exception:"
                            + "Attributes must be continous");
                }
            }
        }
    }
    
    /**
     * Initalises the attribute iterator to the size of the attributes 
     * in the instances object.
     * @param ins   The instances that the attribute iterator is initalised to
     */
    public void initAttributeIndexs(Instances ins) {
        attributeIterator = new int[ins.numAttributes()-1];
        //Set attribute indexes from 0...numAttributes-1
        for(int i = 0; i < attributeIterator.length; i++)
            attributeIterator[i] = i;
    }
    
    /**
     * Initalises the weights to a size specicifed
     * @param size  The size of the weights to initalise
     * @return 
     */
    protected double[] initaliseWeights(int size) {
        
        double[] newWeight;
        
        //Initalise the weights
        if(this.randomiseWeights) {
            newWeight = LinearPerceptron.initWeightRandom(size);
        }
        else {
            newWeight = LinearPerceptron.initWeightNumber(size, this.weightFilled);
        }
        
        return newWeight;
    }
    
    /** 
    * Creates a random array
    * @param size   The size of the weight array
    * @return       The randomly generated double[] weight 
    */
    protected static double[] initWeightRandom(int size) {
        
        double[] newWeight = new double[size];
        
        for(int i = 0; i < newWeight.length; i++)
            newWeight[i] = Math.random() - 0.5;

        return newWeight;
    }
    
    /** 
    * Creates a double array filled with a given number
    * @param size   The size of the weight array
    * @param number The number for the array to be filled
    * @return       The generated double[] weight 
    */
    protected static double[] initWeightNumber(int size, double number) {
        
        double[] newWeight = new double[size];
        
        for(int i = 0; i < newWeight.length; i++) {
            newWeight[i] = number;
        }
        return newWeight;
    }
    
    /** 
    * Classifies the Instances by generating a new weight using the 
    * Online Update method. 
    * The Online Update method updates the weight every time an instance is
    * incorrectly classified. The loop will not stop altering the weight
    * until a full loop has passed with correct classifications, or until
    * the maximum number of loops have passed defined by the depth attribute.
    * @param ins    The Instances object that will used for training.
    * @param weight The weight that will be updated
    * @return       double[] The new weight 
    */
    protected double[] onlineClassification(Instances ins, double[] weight) {
                
        double[] tempWeight;
        double result;
        int depthCounter = 0;
        
        do {
            //Create a copy of the weight at the begining of the loop
            tempWeight = weight.clone();
            for(Instance in : ins) {
                
                //Classify the Instance
                result = bias; //
                for(int i = 0; i < attributeIterator.length; i++)
                    result += weight[i] * in.value(attributeIterator[i]);
                
                //Alter the weight based on the previous classification
                int c = classification(in.classValue()) - classification(result);
                for(int i = 0; i < attributeIterator.length; i++)
                    weight[i] += 0.5 * this.learningRate * c * in.value(attributeIterator[i]);
            }
            //Increment the depth counter after each full iteration
            depthCounter++;
        //Repeat until weight has not altered or maximum depth has been reached
        } while(!Arrays.equals(tempWeight, weight) && depthCounter < this.depth); 
        
        //Return the weight
        return weight;
    }
    
    /** 
    * Returns the correct learning rate, positive or negative
    * @param i      The double that was calculated when building the classifier
    * @return       The negative or positive learning
    */
    protected static int classification(double i) {
        if(i > 0)
            return 1;
        return -1;
    }
    
    /** 
    * Classify the instance
    * @param in     Instance that is to be classified
    * @return       The classification result
    * @throws       Exception
    */
    @Override
    public double classifyInstance(Instance in) throws Exception {
        
        return classify(in, weight);
    }
    
    /** 
    * Classifies an instance using the weight generated by buildClassifer()
    * @param  in        Instance that is to be classified.
    * @param  weight    double[] The weight that is needed to classify 
    * the instance.
    * @return           The classification result
    */
    protected double classify(Instance in, double[] weight) {
        
        double result = bias;
        for(int i = 0; i < attributeIterator.length; i++) {
            result += weight[i] * in.value(attributeIterator[i]);
        }
        
        //Return class value index
        if(result > 0)
            return 1;
        return 0;
    }
    
    /** Prints the weight of the classifier */
    public void printWeight() {
        System.out.println("\nWeight: ");
        for(int i = 0; i < this.weight.length; i++)
            System.out.println("[  " + this.weight[i] + "\t]");
    }
    
        public double[] getWeight() {
        return this.weight;
    }
    
    public boolean getRandomiseWeights() {
        return this.randomiseWeights;
    }
    
    public int getWeightFilled() {
        return this.weightFilled;
    }
    
    public int getLearningRate() {
        return this.learningRate;
    }
    
    public int getDepth() {
        return this.depth;
    }
    
    public int[] getAttributeIterator() {
        return attributeIterator;
    }
    
    public boolean getCheckAttributesFlag() {
        return checkAttributesFlag;
    }
    
    public int getBias() {
        return bias;
    }
    
    public void setWeight(double[] w) {
        this.weight = w;
    }
    
    /** 
    * Sets flag to randomise the weights.
    * @param b  Boolean
    */
    public void setRandomiseWeightsFlag(boolean b) {
        this.randomiseWeights = b;
    }
    
    /** 
    * Sets the number that the weights will be filled with
    * @param w  Integer
    */
    public void setWeightFilled(int w) {
        this.weightFilled = w;
    }
    
    /** 
    * Sets the learning rate for the weights
    * @param r  Integer
    */
    public void setLearningRate(int r) {
        this.learningRate = r;
    }
    
    /** 
    * Sets the maximum depth for the update methods
    * @param d  Integer
    */
    public void setDepth(int d) {
        this.depth = d;
    }

    
    public void setAttributeIterator(int[] a) {
        attributeIterator = a;
    }
    
    public void setCheckAttributesFlag(boolean b) {
        checkAttributesFlag = b;
    }
    
    public void setBias(int b) {
        bias = b;
    }
    
    @Override
    public double[] distributionForInstance(Instance instnc) throws Exception {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public Capabilities getCapabilities() {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }
}
