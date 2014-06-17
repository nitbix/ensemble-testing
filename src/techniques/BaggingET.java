package techniques;

import java.util.List;

import org.encog.ensemble.Ensemble.TrainingAborted;
import org.encog.ensemble.EnsembleAggregator;
import org.encog.ensemble.EnsembleMLMethodFactory;
import org.encog.ensemble.EnsembleTrainFactory;
import org.encog.ensemble.aggregator.WeightedAveraging.WeightMismatchException;
import org.encog.ensemble.bagging.Bagging;
import org.encog.ml.data.MLData;

import helpers.DataLoader;
import helpers.ChainParams;

public class BaggingET extends EvaluationTechnique {

	private int dataSetSize;

	public BaggingET(List<Integer> sizes, int dataSetSize, int maxIterations, ChainParams fullLabel, EnsembleMLMethodFactory mlMethod, EnsembleTrainFactory trainFactory, EnsembleAggregator aggregator) {
		this.sizes = sizes;
		this.dataSetSize = dataSetSize;
		this.label = fullLabel;
		this.mlMethod = mlMethod;
		this.trainFactory = trainFactory;
		this.aggregator = aggregator;
		this.maxIterations = maxIterations;

	}

	@Override
	public void init(DataLoader dataLoader,int fold)
	{
		ensemble = new Bagging(sizes.get(currentSizeIndex),dataSetSize,mlMethod,trainFactory,aggregator);
		dataLoader.setFold(fold);
		setTrainingSet(dataLoader.getTrainingSet());
		setSelectionSet(dataLoader.getCVSet());
		ensemble.setTrainingData(trainingSet);
	}

	@Override
	public MLData compute(MLData input) throws WeightMismatchException {
		return ensemble.compute(input);
	}

	@Override
	public void trainStep() {
		((Bagging) ensemble).trainStep();
	}

	@Override
	public double trainError() {
		return ensemble.getMember(0).getTraining().getError();
	}
	
	@Override
	public void step(boolean verbose) throws TrainingAborted
	{
		if (currentSizeIndex < sizes.size() -1) {
			for (int i = sizes.get(currentSizeIndex++); i < sizes.get(currentSizeIndex); i++) {
				ensemble.addNewMember();
				try {
					ensemble.trainMember(i, trainToError, selectionError, maxIterations, selectionSet, verbose);
				}
				catch (TrainingAborted e) {
					System.out.println("Training aborted on E_t = " + trainToError + ", member no " + i);
				}
			}
		} else {
			this.hasStepsLeft = false;
		}
	}

}
