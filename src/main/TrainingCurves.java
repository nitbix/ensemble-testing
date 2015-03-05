package main;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.sql.Connection;
import java.util.ArrayList;
import java.util.List;
import java.util.Properties;

import org.encog.ensemble.EnsembleAggregator;
import org.encog.ensemble.EnsembleMLMethodFactory;
import org.encog.ensemble.EnsembleTrainFactory;
import org.encog.ensemble.aggregator.WeightedAveraging.WeightMismatchException;
import org.encog.neural.data.basic.BasicNeuralDataSet;

import techniques.AdaBoostET.RequiresWeightedAggregatorException;
import techniques.EvaluationTechnique;
import helpers.ArgParser;
import helpers.ArgParser.BadArgument;
import helpers.DBConnect;
import helpers.DataLoader;
import helpers.DataMapper;
import helpers.Evaluator;
import helpers.ChainParams;
import helpers.FileLoader;
import helpers.ProblemDescription;

public class TrainingCurves {

	Evaluator ev;
	static DataLoader dataLoader;
	static ProblemDescription problem;
	
	private static int trainingSetSize;
	private static double activationThreshold;
	private static EnsembleTrainFactory etf;
	private static List<EnsembleMLMethodFactory> mlfs;
	private static EnsembleAggregator agg;
	private static String etType;
	private static int maxIterations;
	private static int maxLoops;
	
	private static List<Integer> sizes;
	private static List<Integer> dataSetSizes;
	private static List<Double> trainingErrors;
	private static int nFolds;
	private static double selectionError;
	private static int targetRunCount = 0;
	
	private static Connection sqlConnection;
	private static String dbconn,dbuser,dbpass;
	private static DBConnect reconnectCallback;
	public static void loop() throws WeightMismatchException, RequiresWeightedAggregatorException {
		List<Integer> one = new ArrayList<Integer>();
		one.add(1);
		for(EnsembleMLMethodFactory mlf: mlfs)
		{
			ChainParams labeler = new ChainParams("", "", "", "", "", 0);
			EvaluationTechnique et = null;
			try {
				et = ArgParser.technique("CURVES",one,trainingSetSize,labeler,mlf,etf,agg,dataLoader,maxIterations,maxLoops);
			} catch (BadArgument e) {
				help();
			}
			et.init(dataLoader,0);
			DataMapper dataMapper = dataLoader.getMapper();
			BasicNeuralDataSet testSet = dataLoader.getTestSet();
			BasicNeuralDataSet trainingSet = dataLoader.getTrainingSet();
			for (int i=0; i < maxIterations; i++) {
				et.trainStep();
				double trainMSE = et.trainError();
				double testMSE = et.testError();
				double trainMisc = et.getMisclassification(testSet, dataMapper);
				double testMisc = et.getMisclassification(trainingSet, dataMapper);
				System.out.println(i + " " + trainMSE + " " + testMSE
									 + " " + trainMisc + " " + testMisc);
			}
		}
	}
	
	public static void main(String[] args) throws WeightMismatchException, RequiresWeightedAggregatorException {
		FileLoader fileLoader = new FileLoader();
		if (args.length != 8 && args.length != 1) {
			help();
		} 
		try {
			if(args.length == 8)
			{
				problem = ArgParser.problem(args[0]);
				trainingSetSize = ArgParser.intSingle(args[1]);
				activationThreshold = ArgParser.doubleSingle(args[2]);
				etf = ArgParser.ETF(args[3]);
				mlfs = ArgParser.MLFS(args[4]);
				maxIterations = ArgParser.intSingle(args[5]);
				maxLoops = ArgParser.intSingle(args[6]);
			} else if (args.length == 1) {
				Properties problemPropFile = new Properties();
				try {
					problemPropFile.load(fileLoader.openOrFind(args[0]));
				} catch (FileNotFoundException e) {
					System.err.println("Could not find " + args[0]);
					help();
				} catch (IOException e) {
					help();
				}
				problem = ArgParser.problem(problemPropFile.getProperty("problem"));
				nFolds = ArgParser.intSingle(problemPropFile.getProperty("folds"));
				activationThreshold = ArgParser.doubleSingle(problemPropFile.getProperty("neural_invalidation_threshold"));
				dataSetSizes = ArgParser.intList(problemPropFile.getProperty("dataset_resampling_sizes"));
				trainingErrors = ArgParser.doubleList(problemPropFile.getProperty("training_errors"));
				etf = ArgParser.ETF(problemPropFile.getProperty("ensemble_training"));
				maxIterations = ArgParser.intSingle(problemPropFile.getProperty("max_training_iterations"));
				if(problemPropFile.containsKey("max_retrain_loops"))
				{
					maxLoops = ArgParser.intSingle(problemPropFile.getProperty("max_retrain_loops"));			
				}
				mlfs = ArgParser.MLFS(problemPropFile.getProperty("member_types"));
				selectionError = ArgParser.doubleSingle(problemPropFile.getProperty("selection_error"));
				if (nFolds < 2) {
					throw new BadArgument();
				};
				trainingSetSize = dataSetSizes.get(0);
				//OMGHACK
				dataLoader = problem.getDataLoader(activationThreshold,nFolds);
				maxLoops = maxIterations;
			}
			//defaults
			agg = ArgParser.AGG("averaging");
		}
		catch (FileNotFoundException e)
		{
			System.err.println("Could not create dataLoader - data file not found");
		}
		catch (IOException e)
		{
			System.err.println("Could not create dataLoader - IOException" + e.toString());
		}
		catch (helpers.ProblemDescriptionLoader.BadArgument e) 
		{
			System.err.println("Could not create dataLoader - perhaps the mapper_type property is wrong");
			e.printStackTrace();
		}
		catch (BadArgument e)
		{
			help();
		}
		loop();
		System.exit(0);
	}

	private static void help() {
		System.err.println("Usage: TrainingCurves <problem> <trainingSetSize> <activationThreshold> <training> <membertypes> <maxIterations> <maxLoops>");
		System.exit(2);
	}
}
