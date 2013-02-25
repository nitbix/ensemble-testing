package main;

import java.util.ArrayList;
import java.util.List;

import org.encog.ensemble.EnsembleAggregator;
import org.encog.ensemble.EnsembleMLMethodFactory;
import org.encog.ensemble.EnsembleTrainFactory;
import org.encog.neural.data.basic.BasicNeuralDataSet;

import techniques.EvaluationTechnique;
import helpers.ArgParser;
import helpers.ArgParser.BadArgument;
import helpers.DataLoader;
import helpers.DataMapper;
import helpers.Evaluator;
import helpers.ChainParams;
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
	
	public static void loop() {
		List<Integer> one = new ArrayList<Integer>();
		one.add(1);
		for(EnsembleMLMethodFactory mlf: mlfs)
		{
			ChainParams labeler = new ChainParams("", "", "", "", "", 0);
			EvaluationTechnique et = null;
			try {
				et = ArgParser.technique(etType,one,trainingSetSize,labeler,mlf,etf,agg);
			} catch (BadArgument e) {
				help();
			}
			DataMapper dataMapper = dataLoader.getMapper();
			BasicNeuralDataSet testSet = dataLoader.getTestSet(1);
			BasicNeuralDataSet trainingSet = dataLoader.getTrainingSet(1);
			et.init(dataLoader,1);
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
	
	public static void main(String[] args) {
		if (args.length != 7) {
			help();
		} 
		try {
			etType = args[0];
			problem = ArgParser.problem(args[1]);
			trainingSetSize = ArgParser.intSingle(args[2]);
			activationThreshold = ArgParser.doubleSingle(args[3]);
			etf = ArgParser.ETF(args[4]);
			mlfs = ArgParser.MLFS(args[5]);
			agg = ArgParser.AGG("averaging");
			maxIterations = ArgParser.intSingle(args[6]);
		} catch (BadArgument e) {
			help();
		}
		
		try {
			dataLoader = problem.getDataLoader(activationThreshold,trainingSetSize);
		} catch (helpers.ProblemDescriptionLoader.BadArgument e) {
			System.err.println("Could not get dataLoader - perhaps the mapper_type property is wrong");
			e.printStackTrace();
		}
		loop();
		System.exit(0);
	}

	private static void help() {
		System.err.println("Usage: TrainingCurves <technique> <problem> <trainingSetSize> <activationThreshold> <training> <membertypes> <maxIterations>");
		System.exit(2);
	}
}
