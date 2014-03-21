package techniques;

import java.util.List;

import org.encog.ensemble.EnsembleAggregator;
import org.encog.ensemble.EnsembleMLMethodFactory;
import org.encog.ensemble.EnsembleTrainFactory;
import org.encog.ensemble.EnsembleWeightedAggregator;
import org.encog.ensemble.adaboost.AdaBoost;
import org.encog.ensemble.aggregator.WeightedAveraging.WeightMismatchException;
import org.encog.ensemble.data.EnsembleDataSet;
import org.encog.ml.data.MLData;

import helpers.DataLoader;
import helpers.ChainParams;

public class AdaBoostET extends EvaluationTechnique {

	public static class RequiresWeightedAggregatorException extends Exception {

		/**
		 * This happens if you pass a non-weighted aggregator to AdaBoost
		 */
		private static final long serialVersionUID = -621777903130247977L;
	}
	private int dataSetSize;
	private DataLoader dataLoader;
	private int fold;

	public AdaBoostET(List<Integer> sizes, int dataSetSize, int maxIterations, ChainParams fullLabel, EnsembleMLMethodFactory mlMethod, EnsembleTrainFactory trainFactory, EnsembleAggregator aggregator) {
		this.sizes = sizes;
		this.dataSetSize = dataSetSize;
		this.label = fullLabel;
		this.mlMethod = mlMethod;
		this.trainFactory = trainFactory;
		this.aggregator = aggregator;
		this.maxIterations = maxIterations;
	}

	@Override
	public void init(DataLoader dataLoader, int fold) throws RequiresWeightedAggregatorException {
		this.dataLoader = dataLoader;
		this.fold = fold;
		dataLoader.setFold(fold);
		if (!(aggregator instanceof EnsembleWeightedAggregator))
		{
			throw new RequiresWeightedAggregatorException();
		}
		ensemble = new AdaBoost(sizes.get(currentSizeIndex),dataSetSize,mlMethod,trainFactory,(EnsembleWeightedAggregator) aggregator);
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
		System.err.println("Can't to this in AdaBoost");
	}

	@Override
	public double trainError() {
		return ensemble.getMember(0).getTraining().getError();
	}

	private void resize(int size, boolean verbose) {
		((AdaBoost)ensemble).resize(size,trainToError,selectionError,maxIterations,(EnsembleDataSet) selectionSet,verbose);
	}
	
	@Override
	public void step(boolean verbose) {
		currentSizeIndex++;
		if (currentSizeIndex < sizes.size()) {
			this.resize(sizes.get(currentSizeIndex),false);
		} else {
			this.hasStepsLeft = false;
		}
	}
	
}
