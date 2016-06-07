package linear.perceptron;

//Java
import java.io.FileReader;

//Weka
import weka.core.Instance;
import weka.core.Instances;

/** 
 * Main
 * @author James Rogers
 * @version 1.0 2015.
 */
public class Main {

    /** 
    * Loads the arff file and creates an Instances Object
    * @param path    The string file path to the arff file
    * @return       Instances Object loaded from arff file
    */
    public static Instances loadInstances(String path) {
        FileReader reader;
        Instances instances = null;
        try{
            reader = new FileReader(path);
            instances = new Instances(reader);
        }
        catch(Exception e){
            System.out.println("Error: "+e);
        } 
        return instances;
    }
    
    /** 
    * Linear Perceptron method.
    * Create test and train data.
    * Create LinearPerceptron Object and classify data
    * @throws       Exception
    */
    public static void linearPerceptron() throws Exception {
        
        //Create training instance and set class index
        Instances train = loadInstances("../../data/adult/adult_train.arff");
        //Instances train = loadInstances("../../data/q1-train.arff");
        train.setClassIndex(train.numAttributes()-1);
        
        //Create linear peceptron classfier and build it
        LinearPerceptron perceptron = new LinearPerceptron();
        perceptron.buildClassifier(train);
        
        //Create testing instances and classify each instance
        System.out.println("\nLoading in test data");
        Instances test = loadInstances("../../data/adult/adult_test.arff");
        //Instances test = loadInstances("../../data/q1-test.arff");
        test.setClassIndex(test.numAttributes()-1);
        
        System.out.println("Classifying test data");
        
        double result;
        int errorCounter = 0;
        float errorRate;
        
        for(Instance in : test) {
            result = perceptron.classifyInstance(in);
            if(result != in.classValue())
                errorCounter++;
        }
        errorRate = (errorCounter * 100) / test.numInstances();
        
        System.out.println("Classification complete");
        System.out.println("Error rate: " + errorRate + "%");    
    }
    
    /** 
    * Enhanced Linear Perceptron method.
    * Create test and train data.
    * Create EnhancedLinearPerceptron Object and classify data
    * @throws       Exception
    */
    public static void enhancedLinearPerceptron() throws Exception {
                
        //Create training instance and set class index
        Instances train = loadInstances("../../data/adult/adult_train.arff");
        //Instances train = loadInstances("../../data/q1-train.arff");
        
        train.setClassIndex(train.numAttributes()-1);
        
        //Create linear peceptron classfier and build it
        EnhancedLinearPerceptron perceptron = new EnhancedLinearPerceptron();
        perceptron.setDepth(100);
        perceptron.setFolds(50);
        perceptron.setRandomiseWeightsFlag(true);
        perceptron.setWeightFilled(1);
        perceptron.setCrossValidationFlag(true);
        perceptron.setStandardiseFlag(true);
        perceptron.buildClassifier(train);
        
        //Create testing instances and classify each instance
        System.out.println("\nLoading in test data");
        Instances test = loadInstances("../../data/adult/adult_test.arff");
        //Instances test = loadInstances("../../data/q1-test.arff");
        test.setClassIndex(test.numAttributes()-1);
        
        System.out.println("Classifying test data");
        
        double result;
        int errorCounter = 0;
        float errorRate;
        
        for(Instance in : test) {
            result = perceptron.classifyInstance(in);
            if(result != in.classValue())
                errorCounter++;
        }
        errorRate = (errorCounter * 100) / test.numInstances();
        
        System.out.println("Classification complete");
        System.out.println("Error rate: " + errorRate + "%");
    }
    
    /** 
    * Random Linear Perceptron method.
    * Create test and train data.
    * Create LinearPerceptron Object and classify data
    * @throws       Exception
    */
    public static void randomLinearPerceptron() throws Exception {
        
        //Create training instance and set class index
        Instances train = loadInstances("../../data/adult/adult_train.arff");
        //Instances train = loadInstances("../../data/q1-train.arff");
        train.setClassIndex(train.numAttributes()-1);
        
        //Create linear peceptron classfier and build it
        RandomLinearPerceptron p = new RandomLinearPerceptron();
        p.setNumEnsembles(500);
        //p.setNumAttributes(5);
        p.setDepth(100);
        p.setRandomiseWeights(true);
        p.setPerceptronClassifier(true);
        p.setCrossValidationFlag(false);
        p.setFolds(10);
        p.buildClassifier(train);
        
        //Create testing instances and classify each instance
        System.out.println("\nLoading in test data");
        Instances test = loadInstances("../../data/adult/adult_test.arff");
        //Instances test = loadInstances("../../data/q1-test.arff");
        test.setClassIndex(test.numAttributes()-1);
        
        System.out.println("Classifying test data");
        
        double result;
        int errorCounter = 0;
        float errorRate;
        
        for(Instance in : test) {
            result = p.classifyInstance(in);
            if(result != in.classValue())
                errorCounter++;
        }
        errorRate = (errorCounter * 100) / test.numInstances();
        
        System.out.println("Classification complete");
        System.out.println("Error rate: " + errorRate + "%");
    }
    
    
    public static void main(String[] args) throws Exception {
        
        //Linear Perceptron
        System.out.println("\nLinear Perceptron");
        linearPerceptron();
              
        //EnhancedLinearPerceptron
        System.out.println("\n\n\nEnhanced Linear Perceptron\n");
        enhancedLinearPerceptron();
        
        //RandomLinearPerceptron
        System.out.println("\n\n\nRandom Linear Perceptron");
        randomLinearPerceptron();
    
    }
}
