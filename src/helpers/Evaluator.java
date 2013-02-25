package helpers;

import java.sql.SQLException;
import java.sql.Statement;
import java.util.Calendar;

import org.encog.neural.data.basic.BasicNeuralDataSet;

import techniques.EvaluationTechnique;

public class Evaluator {

	private EvaluationTechnique technique;
	private DataLoader dataLoader;
	
	Evaluator(EvaluationTechnique technique, DataMapper mapper, int inputCols, int inputs, String dataFile, boolean inputsReversed, int nFolds, double targetTrainingError, double selectionError, int fold) {
		this.setTechnique(technique);
		dataLoader = new DataLoader(mapper,inputCols,inputs,inputsReversed,nFolds);
		dataLoader.readData(dataFile);
		this.technique.init(dataLoader,fold);
		this.technique.setParams(targetTrainingError, selectionError);
		this.technique.train(false);
	}
	
	public Evaluator(EvaluationTechnique technique, DataLoader dataLoader, double targetTrainingError, double selectionError, boolean verbose, int fold) {
		this.setTechnique(technique);
		this.dataLoader = dataLoader;
		this.technique.init(dataLoader,fold);
		this.technique.setParams(targetTrainingError, selectionError);
		this.technique.train(verbose);
	}
	
	public void makeLine(boolean isTest, double training_error, ChainParams chainPars, BasicNeuralDataSet dataSet, Statement sqlStatement, long chainId) throws SQLException {
		DataMapper dataMapper = dataLoader.getMapper();
		PerfResults perf = this.technique.testPerformance(dataSet, dataMapper,false);
		Calendar cal = Calendar.getInstance();
		long runId = cal.getTimeInMillis();
		sqlStatement.executeUpdate("INSERT INTO runs SET chain = " + chainId +
				", ml_technique = " + chainPars.getMLF() +
				", training_error = " + training_error +
				", dataset_size = " +
				", misclassified_samples = " + this.technique.getMisclassificationCount(dataSet,dataMapper) +
				", is_test = " + Boolean.toString(isTest) +
				", macro_accuracy = " + perf.getAccuracy(PerfResults.AveragingMethod.MACRO) +
				", macro_precision = " + perf.getPrecision(PerfResults.AveragingMethod.MACRO) +
				", macro_recall = " + perf.getRecall(PerfResults.AveragingMethod.MACRO) +
				", macro_f1 = " + perf.FScore(1.0, PerfResults.AveragingMethod.MACRO) +
				", micro_accuracy = " + perf.getAccuracy(PerfResults.AveragingMethod.MICRO) +
				", micro_precision = " + perf.getPrecision(PerfResults.AveragingMethod.MICRO) +
				", micro_recall = " + perf.getRecall(PerfResults.AveragingMethod.MICRO) +
				", micro_f1 = " + perf.FScore(1.0, PerfResults.AveragingMethod.MICRO) +
				", misclassification = " + this.technique.getMisclassification(dataSet,dataMapper) +
				", ensemble_size = " + technique.getCurrentSize() +
				", id = " + runId +
				", chain = " + chainId +
				";"
		);
		int outputs = dataSet.getIdealSize();
		for (int output = 0; output < outputs; output ++)
		{
			sqlStatement.executeUpdate("INSERT INTO class_details SET run = " + runId +
					", class = " + dataMapper.getClassLabel(output) +
					", is_test = " + Boolean.toString(isTest) +
					", tp = " + perf.getTP(output) + 
					", tn = " + perf.getTN(output) +
					", fp = " + perf.getFP(output) +
					", fn = " + perf.getFN(output) +
					";"
			);
		}
	}
	
	public void getResults (ChainParams prefix, double te, int fold, Statement sqlStatement, long chainId) throws SQLException {
		while(technique.hasStepsLeft()) {
			makeLine(false,te,prefix,this.dataLoader.getTrainingSet(fold), sqlStatement, chainId);
			makeLine(true,te,prefix,this.dataLoader.getTestSet(fold), sqlStatement, chainId);
			technique.step(false);
		}
	}

	public EvaluationTechnique getTechnique() {
		return technique;
	}

	public void setTechnique(EvaluationTechnique technique) {
		this.technique = technique;
	}
	
}
